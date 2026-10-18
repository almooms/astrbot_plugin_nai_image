"""点数用量统计（usage_stats.py）与主插件接入的离线测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing import estimate_openai_cost, pixel_size_cost  # noqa: E402
from usage_stats import (  # noqa: E402
    UsageStatsStore,
    date_range,
    format_usage_report,
    sparkline,
    today_str,
)

from main import NAIGenerateImagePlugin  # noqa: E402


class FakeUsageEvent:
    def __init__(self, message_str: str):
        self.message_str = message_str

    def get_sender_id(self):
        return "test-user"

    def plain_result(self, text: str):
        return ("plain", text)


def make_usage_plugin(tmp_path: Path, *, enabled: bool = True):
    """构造只带用量统计所需字段的插件对象，数据目录指向 tmp_path。"""

    plugin = object.__new__(NAIGenerateImagePlugin)
    plugin.enable_usage_stats = enabled
    plugin._usage_stats_lock = asyncio.Lock()
    plugin._usage_stats = None
    plugin._get_plugin_data_dir = staticmethod(lambda: tmp_path)
    return plugin


# ---------- 日期与迷你图 ----------


def test_date_range_is_contiguous_and_ends_today():
    days = date_range(7, end_day="2026-09-28")
    assert days == [
        "2026-09-22",
        "2026-09-23",
        "2026-09-24",
        "2026-09-25",
        "2026-09-26",
        "2026-09-27",
        "2026-09-28",
    ]
    assert date_range(0) == []
    assert len(today_str().split("-")) == 3


def test_sparkline_scales_and_marks_empty_days():
    # 峰值 10：0 → 空柱，5 → 中间档，10 → 满档
    assert sparkline([0, 5, 10]) == "·▅█"
    assert sparkline([]) == ""
    assert sparkline([0, 0]) == "··"


# ---------- OpenAI 兼容模式的画幅推断 ----------


def test_pixel_size_cost_tiers():
    assert pixel_size_cost("1024x1024") == 1
    assert pixel_size_cost("832x1216") == 1
    assert pixel_size_cost("1024x1536") == 1
    assert pixel_size_cost("2048x2048") == 15
    # 2K 方图在 OpenAI 契约下是 1472x1472，最长边不到 1800 也要按面积记 2K
    assert pixel_size_cost("1472x1472") == 15
    assert pixel_size_cost("1472*1472") == 15
    assert pixel_size_cost("1088x1920") == 15
    assert pixel_size_cost("1344x1344") == 15
    assert pixel_size_cost("4096x2048") == 25
    # 非法输入不抛异常，按普通画幅记
    assert pixel_size_cost("竖图") == 1


def test_estimate_openai_cost_uses_model_floor():
    # 标准分辨率 + V5 → 8 点；V4.5 → 1 点
    assert estimate_openai_cost("1024x1536", "nai-diffusion-5-full") == 8
    assert estimate_openai_cost("1024x1536", "nai-diffusion-4-5-full") == 1
    # 中文分档名仍然直接命中档位表
    assert estimate_openai_cost("2K竖图", "nai-diffusion-4-5-full") == 15


# ---------- Store 基本行为 ----------


def test_record_accumulates_cost_and_breakdown():
    store = UsageStatsStore("unused_path.json")
    store.record(
        8,
        transport="web_jobs",
        model="nai-diffusion-5-full",
        size="竖图",
        day="2026-09-27",
    )
    store.record(
        15,
        transport="get",
        model="nai-diffusion-5-full",
        size="2K竖图",
        day="2026-09-27",
    )

    bucket = store._data["days"]["2026-09-27"]
    assert bucket["cost"] == 23
    assert bucket["count"] == 2
    assert bucket["by_transport"] == {"web_jobs": 8, "get": 15}
    assert bucket["by_size"] == {"竖图": 8, "2K竖图": 15}


def test_failed_requests_do_not_consume_points():
    store = UsageStatsStore("unused_path.json")
    store.record(0, count=0, failed=1, day="2026-09-27")
    bucket = store._data["days"]["2026-09-27"]
    assert bucket["cost"] == 0
    assert bucket["failed"] == 1
    assert store.totals(7)["cost"] == 0


def test_upstream_charges_are_kept_separate_from_estimates():
    store = UsageStatsStore("unused_path.json")
    day = today_str()
    store.record(10, count=1, day=day)
    store.record_upstream(8, balance=92.5, checked_at=1_700_000_000, day=day)

    bucket = store._data["days"][day]
    assert bucket["cost"] == 10
    assert bucket["upstream_cost"] == 8
    assert bucket["upstream_events"] == 1
    state = store.upstream_state()
    assert state["balance"] == 92.5
    assert state["last_delta"] == 8

    totals = store.totals(7)
    assert totals["cost"] == 10
    assert totals["upstream_cost"] == 8
    assert totals["upstream_events"] == 1
    assert store.series(1)[-1]["upstream_cost"] == 8


def test_upstream_state_and_cost_persist(tmp_path):
    path = tmp_path / "usage_stats.json"
    day = today_str()
    store = UsageStatsStore(path)
    store.record_upstream(5, balance=88.0, checked_at=1_700_000_000, day=day)
    store.save()

    reloaded = UsageStatsStore(path)
    reloaded.load()
    assert reloaded.totals(7)["upstream_cost"] == 5
    state = reloaded.upstream_state()
    assert state["balance"] == 88.0
    assert state["last_delta"] == 5


def test_series_fills_zero_days_and_totals():
    store = UsageStatsStore("unused_path.json")
    # series(n) 以"今天"为终点回溯 n 天，预期日期必须相对当天动态计算，
    # 否则跨天后窗口滑出硬编码日期（曾是 2026-09-26~28 的时间炸弹）。
    today = date.today()
    day_older = (today - timedelta(days=2)).isoformat()
    day_mid = (today - timedelta(days=1)).isoformat()
    store.record(10, count=2, day=day_older)
    store.record(4, count=1, day=today.isoformat())

    series = store.series(3)
    assert [row["date"] for row in series] == [day_older, day_mid, today.isoformat()]
    assert [row["cost"] for row in series] == [10, 0, 4]

    totals = store.totals(3)
    assert totals["cost"] == 14
    assert totals["count"] == 3
    assert totals["active_days"] == 2
    assert totals["peak_cost"] == 10
    assert totals["peak_date"] == day_older
    assert totals["avg_cost"] == 7.0  # 14 / 2 个有消耗的日子


def test_breakdown_sorted_by_cost():
    store = UsageStatsStore("unused_path.json")
    # 窗口以今天为终点，日期必须相对当天生成，避免跨天后滑出统计窗口。
    day = today_str()
    store.record(1, model="nai-diffusion-4-5-full", day=day)
    store.record(12, model="nai-diffusion-5-full", day=day)
    assert store.breakdown(7, "model") == [
        ("nai-diffusion-5-full", 12),
        ("nai-diffusion-4-5-full", 1),
    ]
    assert store.breakdown(7, "unknown_field") == []


def test_empty_store_report():
    store = UsageStatsStore("unused_path.json")
    assert store.is_empty()
    assert store.series(7) == []
    assert "暂无点数消耗记录" in format_usage_report(store, 7)


# ---------- 持久化 ----------


def test_save_and_reload_roundtrip(tmp_path):
    path = tmp_path / "usage_stats.json"
    day = today_str()
    store = UsageStatsStore(path)
    store.record(25, count=1, model="nai-diffusion-4-5-full", size="4K竖图", day=day)
    store.save()
    assert path.exists()

    reloaded = UsageStatsStore(path)
    reloaded.load()
    assert reloaded.totals(7)["cost"] == 25
    assert reloaded.totals(0)["count"] == 1
    assert reloaded.first_day() == day


def test_corrupted_file_is_backed_up_not_crashing(tmp_path):
    path = tmp_path / "usage_stats.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = UsageStatsStore(path)
    store.load()
    assert store.is_empty()
    assert (tmp_path / "usage_stats.json.corrupted").exists()


def test_prune_drops_old_days(tmp_path):
    store = UsageStatsStore(tmp_path / "usage_stats.json")
    for day in range(1, 6):
        store.record(1, day=f"2026-01-0{day}")
    removed = store.prune(keep_days=2)
    assert removed == 3
    assert sorted(store._data["days"]) == ["2026-01-04", "2026-01-05"]


# ---------- 报告渲染 ----------


def test_report_contains_all_sections():
    store = UsageStatsStore("unused_path.json")
    today = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()
    store.record(
        8,
        count=1,
        transport="web_jobs",
        model="nai-diffusion-5-full",
        size="竖图",
        day=yesterday,
    )
    store.record(
        15,
        count=1,
        transport="get",
        model="nai-diffusion-5-full",
        size="2K竖图",
        day=today.isoformat(),
    )
    report = format_usage_report(store, 7)
    for section in (
        "今日",
        "近 7 日",
        "累计",
        "趋势",
        "明细",
        "按模型",
        "按画幅",
        "按通道",
    ):
        assert section in report
    assert "23 点" in report


def test_report_cumulative_mode():
    store = UsageStatsStore("unused_path.json")
    store.record(3, count=1, day="2026-09-28")
    report = format_usage_report(store, 0)
    assert "（累计）" in report
    assert "近 0 日" not in report


# ---------- 插件接入 ----------


def test_record_usage_stat_persists_and_skips_when_disabled(tmp_path):
    async def scenario():
        plugin = make_usage_plugin(tmp_path)
        await plugin._record_usage_stat(
            15,
            count=1,
            transport="web_jobs",
            model="nai-diffusion-5-full",
            size="2K竖图",
        )
        await plugin._record_usage_stat(
            8,
            count=1,
            transport="get",
            model="nai-diffusion-5-full",
            size="竖图",
        )
        assert (tmp_path / "usage_stats.json").exists()

        plugin_off = make_usage_plugin(tmp_path, enabled=False)
        await plugin_off._record_usage_stat(999, count=1)
        payload = await plugin._usage_stats_payload(7)
        assert payload["window"]["cost"] == 23
        assert payload["window"]["count"] == 2
        assert payload["breakdown"]["transport"][0]["name"] == "web_jobs"
        return json.loads((tmp_path / "usage_stats.json").read_text(encoding="utf-8"))

    data = asyncio.run(scenario())
    assert data["version"] == 1
    assert today_str() in data["days"]


def test_record_usage_from_parameters_reads_estimated_cost(tmp_path):
    async def scenario():
        plugin = make_usage_plugin(tmp_path)
        await plugin._record_usage_from_parameters(
            {
                "estimated_cost": 19,
                "transport": "web_jobs",
                "model": "nai-diffusion-4-5-full",
                "size": "2K竖图",
            }
        )
        return await plugin._usage_stats_payload(7)

    payload = asyncio.run(scenario())
    assert payload["today"]["cost"] == 19
    assert payload["overall"]["cost"] == 19


def test_sync_upstream_billing_once_records_balance_drop(tmp_path):
    async def scenario():
        plugin = make_usage_plugin(tmp_path)
        plugin._fetch_quota = AsyncMock(
            return_value={"ok": True, "balance": 95.0, "value": 95}
        )
        baseline = await plugin._sync_upstream_billing_once(100.0)
        assert baseline == 95.0

        store = plugin._usage_stats_store()
        store.load()
        assert store.totals(7)["upstream_cost"] == 5
        assert store.upstream_state()["balance"] == 95.0

        # 充值/退点：余额增加只更新基线，不计扣费
        plugin._fetch_quota = AsyncMock(
            return_value={"ok": True, "balance": 120.0, "value": 120}
        )
        baseline = await plugin._sync_upstream_billing_once(baseline)
        assert baseline == 120.0
        assert store.totals(7)["upstream_cost"] == 5
        assert store.upstream_state()["balance"] == 120.0

        # 查询失败：保留原基线，记录错误
        plugin._fetch_quota = AsyncMock(
            return_value={"ok": False, "message": "user not found"}
        )
        baseline = await plugin._sync_upstream_billing_once(baseline)
        assert baseline == 120.0
        assert store.upstream_state()["last_error"] == "user not found"

    asyncio.run(scenario())


def test_report_shows_upstream_actual_cost():
    store = UsageStatsStore("unused_path.json")
    day = today_str()
    store.record(10, count=1, day=day)
    store.record_upstream(8, balance=92.0, checked_at=1_700_000_000, day=day)

    report = format_usage_report(store, 7)

    assert "上游实际扣费: 8 点" in report
    assert "余额 92 点" in report
    assert "估算/实际差额: +2 点" in report


def test_report_shows_upstream_balance_without_generated_images():
    store = UsageStatsStore("unused_path.json")
    store.set_upstream_state(balance=100, checked_at=1_700_000_000)

    report = format_usage_report(store, 7)

    assert "上游实际扣费: 0 点" in report
    assert "余额 100 点" in report


def test_usage_command_default_and_all_and_today(tmp_path):
    async def scenario():
        plugin = make_usage_plugin(tmp_path)
        await plugin._record_usage_stat(6, count=1, day="2026-09-01")
        await plugin._record_usage_stat(10, count=2)
        default_report = await collect(plugin.usage(FakeUsageEvent("/usage")))
        all_report = await collect(plugin.usage(FakeUsageEvent("/usage all")))
        today_report = await collect(plugin.usage(FakeUsageEvent("/usage today")))
        month_report = await collect(plugin.usage(FakeUsageEvent("/usage 30")))
        bad_report = await collect(plugin.usage(FakeUsageEvent("/usage abc")))
        return default_report, all_report, today_report, month_report, bad_report

    async def collect(gen):
        return [item async for item in gen]

    default_report, all_report, today_report, month_report, bad_report = asyncio.run(
        scenario()
    )
    assert default_report[0][1].startswith("📊 NAI 点数用量（近 7 日）")
    assert "近 7 日" in default_report[0][1]
    assert "（累计）" in all_report[0][1]
    assert "16 点" in all_report[0][1]  # 6 + 10
    assert "（今日）" in today_report[0][1]
    assert "近 30 日" in month_report[0][1]
    assert "参数无法识别" in bad_report[0][1]


def test_usage_command_when_disabled(tmp_path):
    async def scenario():
        plugin = make_usage_plugin(tmp_path, enabled=False)
        return [item async for item in plugin.usage(FakeUsageEvent("/usage"))]

    results = asyncio.run(scenario())
    assert "已关闭" in results[0][1]
