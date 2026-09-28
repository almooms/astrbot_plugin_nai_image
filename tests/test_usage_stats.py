"""点数用量统计（usage_stats.py）与主插件接入的离线测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import NAIGenerateImagePlugin  # noqa: E402
from pricing import estimate_openai_cost, pixel_size_cost  # noqa: E402
from usage_stats import (  # noqa: E402
    UsageStatsStore,
    date_range,
    format_usage_report,
    sparkline,
    today_str,
)


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
    assert pixel_size_cost("2048x2048") == 15
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
    store.record(8, transport="web_jobs", model="nai-diffusion-5-full", size="竖图",
                 day="2026-09-27")
    store.record(15, transport="get", model="nai-diffusion-5-full", size="2K竖图",
                 day="2026-09-27")

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


def test_series_fills_zero_days_and_totals():
    store = UsageStatsStore("unused_path.json")
    store.record(10, count=2, day="2026-09-26")
    store.record(4, count=1, day="2026-09-28")

    series = store.series(3)
    assert [row["date"] for row in series] == ["2026-09-26", "2026-09-27", "2026-09-28"]
    assert [row["cost"] for row in series] == [10, 0, 4]

    totals = store.totals(3)
    assert totals["cost"] == 14
    assert totals["count"] == 3
    assert totals["active_days"] == 2
    assert totals["peak_cost"] == 10
    assert totals["peak_date"] == "2026-09-26"
    assert totals["avg_cost"] == 7.0  # 14 / 2 个有消耗的日子


def test_breakdown_sorted_by_cost():
    store = UsageStatsStore("unused_path.json")
    store.record(1, model="nai-diffusion-4-5-full", day="2026-09-28")
    store.record(12, model="nai-diffusion-5-full", day="2026-09-28")
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
    store = UsageStatsStore(path)
    store.record(25, count=1, model="nai-diffusion-4-5-full", size="4K竖图",
                 day="2026-09-28")
    store.save()
    assert path.exists()

    reloaded = UsageStatsStore(path)
    reloaded.load()
    assert reloaded.totals(7)["cost"] == 25
    assert reloaded.totals(0)["count"] == 1
    assert reloaded.first_day() == "2026-09-28"


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
    store.record(8, count=1, transport="web_jobs", model="nai-diffusion-5-full",
                 size="竖图", day="2026-09-27")
    store.record(15, count=1, transport="get", model="nai-diffusion-5-full",
                 size="2K竖图", day="2026-09-28")
    report = format_usage_report(store, 7)
    for section in ("今日", "近 7 日", "累计", "趋势", "明细", "按模型", "按画幅", "按通道"):
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
