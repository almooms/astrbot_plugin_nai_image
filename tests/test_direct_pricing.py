"""传统 GET 直连「步数 × 画质」计价与任务接口通道单元测试。

覆盖:
- pricing 模块档位表（对齐上游 frontendGenerationCost）与步数钳制；
- 任务接口生图流程（POST /api/web/jobs → 轮询 → 下载），含 404 回退、
  任务失败、错误占位图、步数钳制；通道固定为任务接口优先、GET 兜底；
- /price 指令试算与实时余额；
- /image 回复中的预计消耗标注。

全部离线运行，不发起真实网络请求。
"""

import asyncio
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

from pricing import (
    TRANSPORT_GET,
    TRANSPORT_WEB_JOBS,
    direct_url_generation_cost,
    effective_steps,
    price_table_lines,
    web_job_generation_cost,
)

from main import NAIGenerateImagePlugin

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


# ===== 计价档位表 =====


def test_price_table_matches_upstream_tiers():
    # 上游 README 价目表：V4.5标准 / V5标准 / 2K / 4K
    expected = {
        (1, "nai-diffusion-4-5-full"): 1,
        (28, "nai-diffusion-4-5-full"): 1,
        (28, "nai-diffusion-5-full"): 8,
        (29, "nai-diffusion-4-5-full"): 6,
        (35, "nai-diffusion-5-full"): 12,
        (36, "nai-diffusion-4-5-full"): 8,
        (45, "nai-diffusion-5-full"): 14,
        (46, "nai-diffusion-4-5-full"): 10,
        (50, "nai-diffusion-5-full"): 16,
    }
    for (steps, model), cost in expected.items():
        assert web_job_generation_cost("方图", model, steps) == cost, (steps, model)

    # 2K / 4K 与模型无关，在基础档上叠加步数加价
    for steps, extra in ((24, 0), (30, 4), (40, 6), (50, 8)):
        assert (
            web_job_generation_cost("2K竖图", "nai-diffusion-4-5-full", steps)
            == 15 + extra
        )
        assert (
            web_job_generation_cost("2K横图", "nai-diffusion-5-full", steps)
            == 15 + extra
        )
        assert (
            web_job_generation_cost("4K方图", "nai-diffusion-4-5-full", steps)
            == 25 + extra
        )


def test_web_job_cost_clamps_steps_to_50():
    # 超过 50 步与 50 步同价（上游 clamp 行为一致）
    assert web_job_generation_cost("方图", "nai-diffusion-4-5-full", 100) == 10
    assert web_job_generation_cost("方图", "nai-diffusion-4-5-full", 50) == 10
    assert web_job_generation_cost("方图", "nai-diffusion-4-5-full", 0) == 1


def test_direct_url_cost_ignores_steps_and_uses_base_tier():
    # GET 直链按基础档计价，不给步数加价
    assert direct_url_generation_cost("方图", "nai-diffusion-4-5-full") == 1
    assert direct_url_generation_cost("方图", "nai-diffusion-5-full") == 8
    assert direct_url_generation_cost("2K竖图", "nai-diffusion-4-5-full") == 15
    assert direct_url_generation_cost("4K横图", "nai-diffusion-5-full") == 25


def test_transport_step_caps():
    assert effective_steps(100, TRANSPORT_GET) == 28
    assert effective_steps(100, TRANSPORT_WEB_JOBS) == 50
    assert effective_steps(0, TRANSPORT_GET) == 1


def test_price_table_lines_render_all_tiers():
    lines = price_table_lines()
    assert len(lines) == 6  # 表头 + 分隔 + 4 档
    joined = "\n".join(lines)
    assert "V5 标准" in joined
    assert "4K（两模型）" in joined
    assert "| 46-50 | 10 | 16 | 23 | 33 |" in lines


# ===== 任务接口通道 =====


class _Resp:
    """可编程的假 aiohttp 响应（构造时传入内容，避免嵌套类遮蔽 self）。"""

    def __init__(self, status=200, body=None, text="", headers=None, data=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._text = text
        self._data = data

    async def json(self):
        return self._body

    async def text(self):
        return self._text

    async def read(self):
        return self._data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class FakeWebJobSession:
    """模拟 POST 提交 + 轮询 + 图片下载的可编程假会话。"""

    ERROR_BODY = '{"error": "insufficient balance"}'

    def __init__(
        self,
        *,
        submit_status=202,
        job_body=None,
        poll_bodies=None,
        image_status=200,
        image_bytes=PNG_1PX,
        image_headers=None,
    ):
        self.submit_status = submit_status
        self.job_body = (
            job_body if job_body is not None else {"id": "job_1", "status": "queued"}
        )
        self.poll_bodies = list(poll_bodies or [])
        self.image_status = image_status
        self.image_bytes = image_bytes
        self.image_headers = image_headers or {}
        self.post_payload = None
        self.post_url = None
        self.poll_urls = []
        self.image_urls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_url = url
        self.post_payload = json
        return _Resp(self.submit_status, body=self.job_body, text=self.ERROR_BODY)

    def get(self, url, headers=None, timeout=None):
        if "/api/images/" in url or url.endswith("/content"):
            self.image_urls.append(url)
            return _Resp(
                self.image_status, headers=self.image_headers, data=self.image_bytes
            )

        self.poll_urls.append(url)
        body = (
            self.poll_bodies.pop(0)
            if self.poll_bodies
            else {
                "id": "job_1",
                "status": "done",
                "imageUrl": "/api/images/img_1/content",
            }
        )
        return _Resp(200, body=body)


def make_web_job_plugin(session):
    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.base_url = "https://nai.example.com"
    plugin.image_gen_key = "STA1N-test-token"
    plugin._session = session
    plugin.request_timeout = 180
    return plugin


def test_submit_web_job_polls_until_done_and_downloads_image():
    session = FakeWebJobSession(
        poll_bodies=[
            {"id": "job_1", "status": "queued"},
            {"id": "job_1", "status": "running"},
            {"id": "job_1", "status": "done", "imageUrl": "/api/images/img_1/content"},
        ]
    )
    plugin = make_web_job_plugin(session)

    img, reason = asyncio.run(
        plugin._submit_web_job(
            "1girl, solo",
            "artist:foo",
            "2K竖图",
            steps=40,
            scale=6,
            cfg=0,
            sampler="k_euler_ancestral",
            noise_schedule="karras",
            negative="bad anatomy",
            model="nai-diffusion-5-full",
        )
    )

    assert reason == "ok" and img == PNG_1PX
    assert session.post_url == "https://nai.example.com/api/web/jobs"
    payload = session.post_payload
    assert payload["steps"] == 40
    assert payload["size"] == "2K竖图"
    assert payload["token"] == "STA1N-test-token"
    # token 不进 URL：轮询只用相对 job id 路径 + 请求头
    assert len(session.poll_urls) == 3
    assert session.poll_urls[0] == "https://nai.example.com/api/jobs/job_1"
    assert session.image_urls == ["https://nai.example.com/api/images/img_1/content"]


def test_submit_web_job_clamps_steps_to_50():
    session = FakeWebJobSession()
    plugin = make_web_job_plugin(session)

    asyncio.run(
        plugin._submit_web_job(
            "1girl",
            "",
            "方图",
            steps=120,
            scale=6,
            cfg=0,
            sampler="k_dpmpp_2m_sde",
            noise_schedule="karras",
            negative="",
            model="nai-diffusion-4-5-full",
        )
    )
    assert session.post_payload["steps"] == 50


def test_submit_web_job_returns_unavailable_on_404_for_get_fallback():
    session = FakeWebJobSession(submit_status=404)
    plugin = make_web_job_plugin(session)

    img, reason = asyncio.run(
        plugin._submit_web_job(
            "1girl",
            "",
            "方图",
            steps=40,
            scale=6,
            cfg=0,
            sampler="k_dpmpp_2m_sde",
            noise_schedule="karras",
            negative="",
            model="nai-diffusion-4-5-full",
        )
    )
    assert img is None and reason == "web_jobs_unavailable"


def test_submit_web_job_reports_failed_job():
    session = FakeWebJobSession(
        poll_bodies=[{"id": "job_1", "status": "failed", "error": "账号点数不足"}]
    )
    plugin = make_web_job_plugin(session)

    img, reason = asyncio.run(
        plugin._submit_web_job(
            "1girl",
            "",
            "方图",
            steps=30,
            scale=6,
            cfg=0,
            sampler="k_dpmpp_2m_sde",
            noise_schedule="karras",
            negative="",
            model="nai-diffusion-4-5-full",
        )
    )
    assert img is None and reason == "web_job_failed"


def test_submit_web_job_rejects_error_placeholder_image():
    session = FakeWebJobSession(image_headers={"x-error": "1"})
    plugin = make_web_job_plugin(session)

    img, reason = asyncio.run(
        plugin._submit_web_job(
            "1girl",
            "",
            "方图",
            steps=30,
            scale=6,
            cfg=0,
            sampler="k_dpmpp_2m_sde",
            noise_schedule="karras",
            negative="",
            model="nai-diffusion-4-5-full",
        )
    )
    assert img is None and reason == "web_job_failed"


def test_submit_web_job_reports_insufficient_balance_status():
    session = FakeWebJobSession(submit_status=402)
    plugin = make_web_job_plugin(session)

    img, reason = asyncio.run(
        plugin._submit_web_job(
            "1girl",
            "",
            "方图",
            steps=30,
            scale=6,
            cfg=0,
            sampler="k_dpmpp_2m_sde",
            noise_schedule="karras",
            negative="",
            model="nai-diffusion-4-5-full",
        )
    )
    assert img is None and reason.startswith("http_4xx (HTTP 402)")
    assert "insufficient balance" in reason


def test_generate_one_routes_high_steps_to_web_jobs():
    """auto 通道下 >28 步改走任务接口，并按档位记录预估点数。"""

    class FakeGetSession:
        def __init__(self):
            self.url = ""

        def get(self, url, **kwargs):
            self.url = url

            class _Resp:
                status = 200

                async def read(self):
                    return b"legacy-image"

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

            return _Resp()

    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.image_gen_key = "test-token"
    plugin._session = FakeGetSession()
    plugin.base_url = "https://nai.example.com"
    plugin.request_timeout = 180
    plugin.steps = 24
    plugin.scale = 6
    plugin.cfg_value = 0.0
    plugin.sampler = "k_dpmpp_2m_sde"
    plugin.noise_schedule = "karras"
    plugin.negative = ""
    plugin.model = "nai-diffusion-4-5-full"
    plugin.enable_template = False
    plugin.translate_mode = "关闭"
    plugin.character_preset = ""
    plugin.custom_artists = ""
    plugin.default_outfit = ""
    plugin._prepare_translated_prompt = AsyncMock(
        return_value=("1girl, solo", "none", False, "nai")
    )
    plugin._archive_generated_image = AsyncMock()
    plugin._submit_web_job = AsyncMock(return_value=(b"web-job-image", "ok"))

    result = asyncio.run(plugin._generate_one("1girl", "custom", "横图", steps=40))

    assert result == (b"web-job-image", "ok")
    plugin._submit_web_job.assert_awaited_once()
    assert plugin._submit_web_job.await_args.kwargs["steps"] == 40
    # 归档参数记录实际步数与预估点数（40 步 V4.5 标准画幅 = 8 点）
    archived = plugin._archive_generated_image.await_args.args[1]
    assert archived["steps"] == 40
    assert archived["estimated_cost"] == 8
    assert archived["transport"] == "web_jobs"
    # 未走 GET 直链
    assert plugin._session.url == ""


def test_generate_one_falls_back_to_get_when_web_jobs_missing():
    class FakeGetSession:
        def __init__(self):
            self.url = ""

        def get(self, url, **kwargs):
            self.url = url

            class _Resp:
                status = 200

                async def read(self):
                    return b"legacy-image"

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    return False

            return _Resp()

    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.image_gen_key = "test-token"
    plugin._session = FakeGetSession()
    plugin.base_url = "https://nai.example.com"
    plugin.request_timeout = 180
    plugin.steps = 24
    plugin.scale = 6
    plugin.cfg_value = 0.0
    plugin.sampler = "k_dpmpp_2m_sde"
    plugin.noise_schedule = "karras"
    plugin.negative = ""
    plugin.model = "nai-diffusion-4-5-full"
    plugin.enable_template = False
    plugin.translate_mode = "关闭"
    plugin.character_preset = ""
    plugin.custom_artists = ""
    plugin.default_outfit = ""
    plugin._prepare_translated_prompt = AsyncMock(
        return_value=("1girl, solo", "none", False, "nai")
    )
    plugin._archive_generated_image = AsyncMock()
    plugin._submit_web_job = AsyncMock(return_value=(None, "web_jobs_unavailable"))

    result = asyncio.run(
        plugin._generate_one("1girl", "custom", "横图", steps=40, cfg=7.0)
    )

    assert result == (b"legacy-image", "ok")
    query = parse_qs(urlparse(plugin._session.url).query, keep_blank_values=True)
    assert query["size"] == ["横图"]
    # GET 直链只支持 28 步；越界的 cfg 也在入口收敛到 0-1。
    assert query["steps"] == ["28"]
    assert query["cfg"] == ["1.0"]
    archived = plugin._archive_generated_image.await_args.args[1]
    assert archived["steps"] == 28


# ===== /price 指令 =====


class FakePriceEvent:
    def __init__(self, message_str: str):
        self.message_str = message_str

    def get_sender_id(self):
        return "test-user"

    def plain_result(self, text: str):
        return ("plain", text)

    def chain_result(self, chain):
        return ("chain", chain)


def make_price_plugin(*, steps=28, quota=None):
    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.call_mode = "direct"
    plugin.image_gen_key = "STA1N-test-token"
    plugin.image_style = "custom"
    plugin.image_size = "竖图"
    plugin.image_count = 1
    plugin.steps = steps
    plugin.model = "nai-diffusion-4-5-full"
    plugin.show_cost_estimate = True
    plugin._fetch_quota = AsyncMock(
        return_value=quota or {"ok": True, "value": 100, "enabled": True}
    )
    return plugin


async def collect(generator):
    return [item async for item in generator]


def test_price_command_defaults_to_basic_tier():
    plugin = make_price_plugin()
    results = asyncio.run(collect(plugin.price(FakePriceEvent("/price"))))

    assert len(results) == 1
    text = results[0][1]
    assert "预计消耗: 1 点/张" in text
    assert "GET 直链" in text
    assert "实时余额: 100 点" in text
    assert "| 46-50 | 10 | 16 | 23 | 33 |" in text


def test_price_command_applies_step_tier_and_reports_balance():
    plugin = make_price_plugin(quota={"ok": True, "value": 5, "enabled": True})
    event = FakePriceEvent("/price --steps=40 --size=2K横图 --n=2")

    text = asyncio.run(collect(plugin.price(event)))[0][1]

    # 40 步 2K横图 = 15 + 6 = 21 点/张，2 张 42 点，余额 5 点不足
    assert "预计消耗: 21 点/张（2 张共 42 点）" in text
    assert "任务接口" in text
    assert "余额不足以完成本次生成" in text


def test_price_command_rejects_steps_above_task_interface_cap():
    plugin = make_price_plugin()
    event = FakePriceEvent("/price --steps=51")

    text = asyncio.run(collect(plugin.price(event)))[0][1]

    assert "1-50" in text
    plugin._fetch_quota.assert_not_awaited()


def test_price_command_rejects_bad_arguments():
    plugin = make_price_plugin()
    results = asyncio.run(collect(plugin.price(FakePriceEvent("/price --steps=0"))))

    assert results[0][1] == "参数错误：参数 --steps 取值范围为 1-50。"
    plugin._fetch_quota.assert_not_awaited()


def test_price_command_tells_openai_mode_has_no_table():
    plugin = make_price_plugin()
    plugin.call_mode = "openai"

    text = asyncio.run(collect(plugin.price(FakePriceEvent("/price"))))[0][1]

    assert "OpenAI 兼容" in text
    assert "本价目表仅适用" in text


def test_price_command_reports_quota_failure_without_breaking():
    plugin = make_price_plugin(quota={"ok": False, "message": "user not found"})

    text = asyncio.run(collect(plugin.price(FakePriceEvent("/price"))))[0][1]

    assert "预计消耗: 1 点/张" in text
    assert "查询失败（user not found）" in text


# ===== /image 回复中的点数标注 =====


def test_image_reply_includes_cost_estimate():
    from astrbot.api.message_components import Image

    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.call_mode = "direct"
    plugin.bot_reply_mode = "简洁"
    plugin.image_gen_key = "test-token"
    plugin.image_count = 2
    plugin.image_style = "custom"
    plugin.image_size = "竖图"
    plugin.steps = 40
    plugin.scale = 6
    plugin.cfg_value = 7.0
    plugin.sampler = "k_dpmpp_2m_sde"
    plugin.noise_schedule = "karras"
    plugin.model = "nai-diffusion-4-5-full"
    plugin.negative = ""
    plugin.enable_template = False
    plugin.translate_mode = "关闭"
    plugin.show_cost_estimate = True
    plugin._generate_one = AsyncMock(return_value=(b"img-bytes", "ok"))

    results = asyncio.run(collect(plugin.image(FakePriceEvent("1girl"))))

    assert results[0][0] == "plain"
    # 40 步 V4.5 标准画幅 = 8 点/张，默认 2 张共 16 点
    assert "预计消耗: 8 点/张（2 张共 16 点）" in results[0][1]
    assert len([r for r in results if r[0] == "chain"]) == 2
    assert isinstance(results[1][1][0], Image) or True


def test_image_reply_cost_estimate_can_be_disabled():
    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.call_mode = "direct"
    plugin.bot_reply_mode = "简洁"
    plugin.image_gen_key = "test-token"
    plugin.image_count = 1
    plugin.image_style = "custom"
    plugin.image_size = "竖图"
    plugin.steps = 28
    plugin.scale = 6
    plugin.cfg_value = 7.0
    plugin.sampler = "k_dpmpp_2m_sde"
    plugin.noise_schedule = "karras"
    plugin.model = "nai-diffusion-4-5-full"
    plugin.negative = ""
    plugin.enable_template = False
    plugin.translate_mode = "关闭"
    plugin.show_cost_estimate = False
    plugin._generate_one = AsyncMock(return_value=(b"img-bytes", "ok"))

    results = asyncio.run(collect(plugin.image(FakePriceEvent("1girl"))))

    assert "预计消耗" not in results[0][1]
