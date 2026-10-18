"""点数用量统计。

按本地日期聚合每次成功生图的点数消耗，持久化在插件数据目录的
``usage_stats.json``，供聊天指令 ``/usage`` 与测试面板图表读取。

统计口径:

- 只统计**成功出图**的调用，点数按 :mod:`pricing` 的档位表估算；
- 传统 GET 直连 / 任务接口走上游真实档位，OpenAI 兼容模式按像素画幅推断
  档位（上游可能另行计价，记为估算值）；
- 失败请求单独记 ``failed`` 次数，**不计入点数**（上游任务失败会退点）。

存储结构::

    {
      "version": 1,
      "updated_at": 1690000000.0,
      "days": {
        "2026-09-28": {
          "cost": 24, "count": 3, "failed": 0,
          "upstream_cost": 20, "upstream_events": 2,
          "by_transport": {"get": 1, "web_jobs": 2},
          "by_model": {"nai-diffusion-4-5-full": 24},
          "by_size": {"竖图": 8, "2K竖图": 16}
        }
      },
      "upstream": {
        "balance": 943.0,
        "checked_at": 1690000000.0,
        "last_delta": 8,
        "last_error": ""
      }
    }

``upstream_cost`` 只记录上游余额两次查询之间的减少量，来源是真实的网页
余额变化，单位与 ``cost`` 相同但统计口径不同（包含同一密钥在网页或其他
客户端的用量），因此不能相加。

类本身是纯同步的（便于单测），异步封装与并发锁由 ``main.py`` 负责。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

STATS_VERSION = 1
# 默认统计文件名（相对插件数据目录）。
STATS_FILE_NAME = "usage_stats.json"
# 迷你柱状图的字符梯度，索引 0 表示有消耗的最低档。
BAR_BLOCKS = "▁▂▃▄▅▆▇█"
# 空柱（当日无消耗）。
BAR_EMPTY = "·"
# 最长保留天数，超出部分在写入时裁剪，避免文件无限增长。
MAX_TRACKED_DAYS = 730

# 可聚合的维度 → 存储中的键名。
BREAKDOWN_FIELDS = ("transport", "model", "size")
_FIELD_KEY = {
    "transport": "by_transport",
    "model": "by_model",
    "size": "by_size",
}

_DAY_FORMAT = "%Y-%m-%d"


def today_str(now: datetime | None = None) -> str:
    """返回本地时区的日期串 ``YYYY-MM-DD``。"""

    return (now or datetime.now()).strftime(_DAY_FORMAT)


def _parse_day(day: str) -> datetime.date:
    return datetime.strptime(day, _DAY_FORMAT).date()


def date_range(days: int, end_day: str | None = None) -> list[str]:
    """返回以 ``end_day``（默认今天）结尾、长度为 ``days`` 的连续日期列表。

    ``days <= 0`` 时返回空列表。
    """

    if days <= 0:
        return []
    end = _parse_day(end_day) if end_day else datetime.now().date()
    start = end - timedelta(days=days - 1)
    return [
        (start + timedelta(days=offset)).strftime(_DAY_FORMAT) for offset in range(days)
    ]


def _new_bucket() -> dict[str, Any]:
    return {
        "cost": 0,
        "count": 0,
        "failed": 0,
        # 上游余额同步得到的实际扣费，与插件按价目表估算的 cost 分开累计，
        # 避免把估算值和余额变化重复计入同一个总数。
        "upstream_cost": 0,
        "upstream_events": 0,
        "by_transport": {},
        "by_model": {},
        "by_size": {},
    }


def _new_upstream_state() -> dict[str, Any]:
    return {
        "balance": None,
        "checked_at": 0.0,
        "last_delta": 0,
        "last_error": "",
    }


class UsageStatsStore:
    """按日期聚合的点数用量仓库（同步实现，读写由调用方串行化）。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {
            "version": STATS_VERSION,
            "updated_at": 0.0,
            "days": {},
            "upstream": _new_upstream_state(),
        }
        self._loaded = False

    # ---------- 持久化 ----------

    def load(self) -> dict[str, Any]:
        """从磁盘载入；文件缺失或损坏时返回空数据（损坏文件改名备份）。"""

        if self._loaded:
            return self._data
        self._loaded = True
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._data
        except OSError:
            return self._data
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self._backup_corrupted()
            return self._data
        if not isinstance(parsed, dict) or not isinstance(parsed.get("days"), dict):
            self._backup_corrupted()
            return self._data
        upstream_state = _new_upstream_state()
        raw_upstream = parsed.get("upstream")
        if isinstance(raw_upstream, dict):
            try:
                if raw_upstream.get("balance") is not None:
                    upstream_state["balance"] = float(raw_upstream["balance"])
                upstream_state["checked_at"] = float(
                    raw_upstream.get("checked_at") or 0.0
                )
                upstream_state["last_delta"] = int(raw_upstream.get("last_delta") or 0)
            except (TypeError, ValueError):
                upstream_state = _new_upstream_state()
            upstream_state["last_error"] = str(raw_upstream.get("last_error") or "")[
                :200
            ]
        self._data = {
            "version": STATS_VERSION,
            "updated_at": float(parsed.get("updated_at") or 0.0),
            "days": {
                str(day): {**_new_bucket(), **bucket}
                for day, bucket in parsed["days"].items()
                if isinstance(bucket, dict)
            },
            "upstream": upstream_state,
        }
        return self._data

    def _backup_corrupted(self) -> None:
        try:
            self.path.replace(self.path.with_suffix(".json.corrupted"))
        except OSError:
            pass

    def save(self) -> None:
        """原子写入磁盘（临时文件 + 替换），并裁剪超期数据。"""

        self.prune()
        self._data["version"] = STATS_VERSION
        self._data["updated_at"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f".{self.path.name}.tmp")
        temp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self.path)

    def prune(self, keep_days: int = MAX_TRACKED_DAYS) -> int:
        """删除超过 ``keep_days`` 天的记录，返回删除天数。"""

        days = self._data["days"]
        if keep_days > 0 and len(days) > keep_days:
            stale = sorted(days)[: len(days) - keep_days]
            for day in stale:
                days.pop(day, None)
            return len(stale)
        return 0

    # ---------- 写入 ----------

    def record(
        self,
        cost: int,
        *,
        count: int = 1,
        failed: int = 0,
        transport: str = "",
        model: str = "",
        size: str = "",
        day: str | None = None,
    ) -> dict[str, Any]:
        """累计一次生成结果，返回当天的聚合桶。

        ``cost`` 为点数（失败传 0），``count`` 为成功张数，``failed`` 为失败次数。
        三个维度只按点数累计，用于占比分析。
        """

        self.load()
        day = day or today_str()
        bucket = self._data["days"].setdefault(day, _new_bucket())
        cost = max(0, int(cost))
        bucket["cost"] += cost
        bucket["count"] += max(0, int(count))
        bucket["failed"] += max(0, int(failed))
        if cost:
            for field, value in (
                ("transport", transport),
                ("model", model),
                ("size", size),
            ):
                if not value:
                    continue
                key = _FIELD_KEY[field]
                bucket[key][value] = bucket[key].get(value, 0) + cost
        return bucket

    def record_upstream(
        self,
        cost: int,
        *,
        balance: float | None = None,
        checked_at: float | None = None,
        last_error: str = "",
        day: str | None = None,
    ) -> dict[str, Any]:
        """记录一次上游余额同步得到的实际扣费。

        ``cost`` 是相邻两次余额查询之间的减少量，单独累计到
        ``upstream_cost``，不会混入插件按价目表估算的 ``cost``。

        Args:
            cost: 本次检测到的余额减少量（点），负数按 0 处理。
            balance: 本次查询到的上游余额，未知时传 None。
            checked_at: 本次查询的时间戳（秒），None 时使用当前时间。
            last_error: 查询失败时的错误摘要，成功传空串。
            day: 计入的日期（YYYY-MM-DD），默认今天。

        Returns:
            当天的聚合桶。
        """

        self.load()
        day = day or today_str()
        bucket = self._data["days"].setdefault(day, _new_bucket())
        cost = max(0, int(round(cost)))
        if cost:
            bucket["upstream_cost"] += cost
            bucket["upstream_events"] += 1
        self.set_upstream_state(
            balance=balance,
            checked_at=checked_at,
            last_delta=cost,
            last_error=last_error,
        )
        return bucket

    def set_upstream_state(
        self,
        *,
        balance: float | None = None,
        checked_at: float | None = None,
        last_delta: int | None = None,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        """更新上游余额同步状态；None 表示保留原值。"""

        state = self._data.setdefault("upstream", _new_upstream_state())
        if balance is not None:
            state["balance"] = float(balance)
        if checked_at is not None:
            state["checked_at"] = float(checked_at)
        if last_delta is not None:
            state["last_delta"] = int(last_delta)
        if last_error is not None:
            state["last_error"] = str(last_error)[:200]
        return state

    def upstream_state(self) -> dict[str, Any]:
        """返回上游余额同步状态快照。"""

        self.load()
        return dict(self._data.get("upstream") or _new_upstream_state())

    def reset(self) -> None:
        """清空全部统计。"""

        self._data["days"] = {}
        self._data["upstream"] = _new_upstream_state()
        self._data["updated_at"] = time.time()

    # ---------- 查询 ----------

    def series(self, days: int = 30) -> list[dict[str, Any]]:
        """返回连续日期序列（含 0 值填充），按日期升序。

        ``days <= 0`` 表示从最早记录到今天。
        """

        self.load()
        if not self._data["days"]:
            return []
        if days and days > 0:
            day_keys = date_range(days)
        else:
            start = _parse_day(min(self._data["days"]))
            end = datetime.now().date()
            span = (end - start).days + 1
            day_keys = [
                (start + timedelta(days=offset)).strftime(_DAY_FORMAT)
                for offset in range(max(1, span))
            ]
        result = []
        for day in day_keys:
            bucket = self._data["days"].get(day)
            result.append(
                {
                    "date": day,
                    "cost": int(bucket["cost"]) if bucket else 0,
                    "count": int(bucket["count"]) if bucket else 0,
                    "failed": int(bucket["failed"]) if bucket else 0,
                    "upstream_cost": int(bucket["upstream_cost"]) if bucket else 0,
                    "upstream_events": (
                        int(bucket["upstream_events"]) if bucket else 0
                    ),
                }
            )
        return result

    def totals(self, days: int = 0) -> dict[str, Any]:
        """返回窗口内的汇总（``days<=0`` 表示累计全部）。"""

        rows = self.series(days)
        cost = sum(row["cost"] for row in rows)
        count = sum(row["count"] for row in rows)
        failed = sum(row["failed"] for row in rows)
        upstream_cost = sum(row["upstream_cost"] for row in rows)
        upstream_events = sum(row["upstream_events"] for row in rows)
        active = sum(1 for row in rows if row["cost"])
        return {
            "days_in_window": len(rows),
            "active_days": active,
            "cost": cost,
            "count": count,
            "failed": failed,
            "upstream_cost": upstream_cost,
            "upstream_events": upstream_events,
            "avg_cost": round(cost / active, 1) if active else 0.0,
            "peak_cost": max((row["cost"] for row in rows), default=0),
            "peak_date": max(rows, key=lambda row: row["cost"])["date"] if rows else "",
        }

    def breakdown(self, days: int, field: str) -> list[tuple[str, int]]:
        """返回窗口内某个维度的点数占比，按点数降序。"""

        key = _FIELD_KEY.get(field)
        if not key:
            return []
        rows = self.series(days)
        totals: dict[str, int] = {}
        for row in rows:
            bucket = self._data["days"].get(row["date"])
            if not bucket:
                continue
            for name, value in bucket.get(key, {}).items():
                totals[name] = totals.get(name, 0) + int(value)
        return sorted(totals.items(), key=lambda item: item[1], reverse=True)

    def first_day(self) -> str:
        self.load()
        return min(self._data["days"]) if self._data["days"] else ""

    def is_empty(self) -> bool:
        self.load()
        return not self._data["days"]


# ---------- 文本报告 ----------


def sparkline(values: Iterable[int]) -> str:
    """把数值序列渲染为迷你柱状图字符串（0 用 ``·`` 表示）。"""

    values = list(values)
    if not values:
        return ""
    peak = max(values)
    if peak <= 0:
        return BAR_EMPTY * len(values)
    last_index = len(BAR_BLOCKS) - 1
    out = []
    for value in values:
        if value <= 0:
            out.append(BAR_EMPTY)
            continue
        ratio = value / peak
        index = int(round(ratio * last_index))
        out.append(BAR_BLOCKS[max(0, min(last_index, index))])
    return "".join(out)


def format_usage_report(store: UsageStatsStore, days: int = 7) -> str:
    """渲染聊天指令用的中文用量报告。"""

    upstream = store.upstream_state()
    if (
        store.is_empty()
        and upstream.get("balance") is None
        and not upstream.get("last_error")
    ):
        return (
            "📊 暂无点数消耗记录。\n"
            "每次成功生图后会按上游档位表累计点数，生成一张后再来查即可。"
        )

    window = store.totals(days)
    overall = store.totals(0)
    today_row = store.series(1)[-1] if store.series(1) else {"cost": 0, "count": 0}
    first_day = store.first_day()
    if days <= 0:
        window_label = "累计"
    elif days == 1:
        window_label = "今日"
    else:
        window_label = f"近 {days} 日"
    since = f"（自 {first_day} 起）" if first_day else ""

    lines = [
        f"📊 NAI 点数用量（{window_label}）",
        f"今日: {today_row['cost']} 点 / {today_row['count']} 张"
        + (
            f" · 上游 {today_row['upstream_cost']} 点"
            if today_row.get("upstream_cost")
            else ""
        ),
    ]
    if days <= 0:
        lines.append(
            f"累计: {overall['cost']} 点 / {overall['count']} 张"
            f"（{overall['active_days']} 天有消耗"
            + (f"，{since.strip('（）')}" if since else "")
            + "）"
        )
    else:
        # days == 1 时窗口就是「今日」，上面已列过，不重复输出。
        if days > 1:
            lines.append(
                f"{window_label}: {window['cost']} 点 / {window['count']} 张"
                f"（{window['active_days']} 天有消耗，日均 {window['avg_cost']} 点）"
            )
        lines.append(f"累计: {overall['cost']} 点 / {overall['count']} 张{since}")
    if window["failed"]:
        lines.append(f"失败: {window['failed']} 次（上游失败不扣点，仅作排查参考）")

    # 上游余额同步：与插件估算值分开展示，避免把网页/其他客户端的用量
    # 混进插件自己的点数台账。
    upstream_cost = int(window["upstream_cost"] or 0)
    if (
        upstream_cost
        or upstream.get("balance") is not None
        or upstream.get("last_error")
    ):
        upstream_text = f"上游实际扣费: {upstream_cost} 点（{window_label}）"
        if upstream.get("balance") is not None:
            upstream_text += f" · 余额 {upstream['balance']:g} 点"
        if upstream.get("checked_at"):
            synced_at = datetime.fromtimestamp(upstream["checked_at"]).strftime(
                "%m-%d %H:%M"
            )
            upstream_text += f" · 最近同步 {synced_at}"
        if upstream.get("last_error"):
            upstream_text += f" · 同步异常: {upstream['last_error'][:60]}"
        lines.append(upstream_text)
        if window["cost"] and upstream_cost:
            lines.append(
                f"估算/实际差额: {window['cost'] - upstream_cost:+d} 点"
                "（上游实际包含同一密钥在网页或其他客户端的用量）"
            )

    rows = store.series(days)
    lines.append("")
    lines.append(f"趋势: {sparkline(row['cost'] for row in rows)}")
    if any(row["upstream_cost"] for row in rows):
        lines.append(f"上游趋势: {sparkline(row['upstream_cost'] for row in rows)}")
    if window["peak_cost"]:
        lines.append(f"峰值: {window['peak_cost']} 点（{window['peak_date']}）")

    active_rows = [
        row for row in reversed(rows) if row["cost"] or row["upstream_cost"]
    ][:10]
    if active_rows:
        lines.append("")
        lines.append("明细（近 10 个有消耗的日子）:")
        for row in active_rows:
            detail = f"  {row['date'][5:]}  {row['cost']} 点 / {row['count']} 张"
            if row["upstream_cost"]:
                detail += f" · 上游 {row['upstream_cost']} 点"
            lines.append(detail)

    breakdown_lines = []
    for field, label in (("model", "模型"), ("size", "画幅"), ("transport", "通道")):
        items = store.breakdown(days, field)[:4]
        if not items:
            continue
        text = " · ".join(f"{name} {value} 点" for name, value in items)
        breakdown_lines.append(f"按{label}: {text}")
    if breakdown_lines:
        lines.append("")
        lines.extend(breakdown_lines)

    return "\n".join(lines)
