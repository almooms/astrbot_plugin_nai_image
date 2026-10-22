"""提示词转译：默认关闭，关闭时转译专属配置整块隐藏且完全不生效。"""

import json
import shutil
from pathlib import Path

import pytest

from main import NAIGenerateImagePlugin, TRANSLATE_GATED_KEYS

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "_conf_schema.json"


def _plugin(mode: str | None = None, **attrs):
    """构造一个只带转译相关属性的插件实例（不跑 __init__）。"""

    plugin = object.__new__(NAIGenerateImagePlugin)
    if mode is not None:
        plugin.translate_mode = mode
    plugin.translate_provider = attrs.get("translate_provider", "")
    plugin.interrogate_provider = attrs.get("interrogate_provider", "")
    return plugin


def _schema_copy(tmp_path) -> Path:
    target = tmp_path / "_conf_schema.json"
    shutil.copyfile(SCHEMA_PATH, target)
    return target


def _attach_schema(plugin, path: Path) -> None:
    plugin._schema_path = lambda: path  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 默认值：转译默认关闭
# ---------------------------------------------------------------------------
def test_translate_defaults_to_off_in_schema():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["enable_translate"]["default"] == "关闭"


def test_translate_defaults_to_off_when_key_missing():
    from prompt_processing import normalize_translate_mode

    assert normalize_translate_mode(None) == "关闭"


# ---------------------------------------------------------------------------
# 关闭时：转译专属配置从设置页隐藏
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mode", "expected"),
    [("关闭", "开启"), ("开启", "开启"), ("自动", "自动")],
)
def test_translate_condition_value(mode, expected):
    assert _plugin(mode)._translate_condition_value() == expected


def test_sync_hides_translate_entries_when_off(tmp_path):
    plugin = _plugin("关闭")
    schema_path = _schema_copy(tmp_path)
    _attach_schema(plugin, schema_path)

    plugin._sync_translate_visibility()

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for key in TRANSLATE_GATED_KEYS:
        # 关闭时条件值与「关闭」不相等 → 前端判定为隐藏
        assert schema[key]["condition"]["enable_translate"] == "开启"
        assert schema[key]["condition"]["enable_translate"] != schema["enable_translate"][
            "default"
        ]
    # 服装缓存时长仍保留自己的开关条件
    assert schema["outfit_cache_ttl_seconds"]["condition"]["enable_outfit_cache"] is True


def test_sync_shows_translate_entries_when_auto(tmp_path):
    plugin = _plugin("自动")
    schema_path = _schema_copy(tmp_path)
    _attach_schema(plugin, schema_path)

    plugin._sync_translate_visibility()

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    for key in TRANSLATE_GATED_KEYS:
        assert schema[key]["condition"]["enable_translate"] == "自动"


def test_sync_does_not_touch_other_schema_keys(tmp_path):
    plugin = _plugin("关闭")
    schema_path = _schema_copy(tmp_path)
    _attach_schema(plugin, schema_path)
    before = json.loads(schema_path.read_text(encoding="utf-8"))

    plugin._sync_translate_visibility()

    after = json.loads(schema_path.read_text(encoding="utf-8"))
    for key in before:
        if key in TRANSLATE_GATED_KEYS:
            continue
        assert before[key] == after[key]


def test_sync_is_noop_when_condition_already_matches(tmp_path, monkeypatch):
    plugin = _plugin("自动")
    schema_path = _schema_copy(tmp_path)
    _attach_schema(plugin, schema_path)
    plugin._sync_translate_visibility()

    mtime_before = schema_path.stat().st_mtime_ns
    plugin._sync_translate_visibility()
    assert schema_path.stat().st_mtime_ns == mtime_before


def test_sync_failure_is_silent(tmp_path):
    plugin = _plugin("关闭")
    _attach_schema(plugin, tmp_path / "不存在" / "_conf_schema.json")

    plugin._sync_translate_visibility()  # 不抛异常

    assert plugin._translate_visibility_warned is True


# ---------------------------------------------------------------------------
# 关闭时：转译供应商不参与任何请求
# ---------------------------------------------------------------------------
class _ExplodingContext:
    """任何 provider 访问都应视为错误——关闭时不该碰上下文。"""

    def __getattr__(self, name):  # pragma: no cover - 只作为断言
        raise AssertionError(f"转译关闭时不应访问 context.{name}")


def test_translate_provider_not_resolved_when_off():
    plugin = _plugin("关闭", translate_provider="some-provider")
    plugin.context = _ExplodingContext()

    assert plugin._resolve_translate_provider_id() is None


def test_translate_provider_still_resolved_when_on():
    class _Ctx:
        def get_provider_by_id(self, pid):
            return object() if pid == "some-provider" else None

        def get_using_provider(self):
            return None

    plugin = _plugin("开启", translate_provider="some-provider")
    plugin.context = _Ctx()

    assert plugin._resolve_translate_provider_id() == "some-provider"


# ---------------------------------------------------------------------------
# 图片反推：关闭转译时不再借用转译供应商
# ---------------------------------------------------------------------------
def test_interrogate_does_not_borrow_translate_provider_when_off():
    plugin = _plugin("关闭", translate_provider="translate-prov")
    assert plugin._resolve_interrogate_provider_id() == ""


def test_interrogate_borrows_translate_provider_when_on():
    plugin = _plugin("自动", translate_provider="translate-prov")
    assert plugin._resolve_interrogate_provider_id() == "translate-prov"


@pytest.mark.parametrize("mode", ["关闭", "开启", "自动"])
def test_interrogate_own_provider_always_wins(mode):
    plugin = _plugin(
        mode, translate_provider="translate-prov", interrogate_provider="vision-prov"
    )
    assert plugin._resolve_interrogate_provider_id() == "vision-prov"
