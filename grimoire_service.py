from __future__ import annotations

import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

_PLUGIN_NAME = "astrbot_plugin_nai_image"
_PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATHS = [
    _PLUGIN_DIR.parent.parent / "plugin_data" / _PLUGIN_NAME / "grimoire.db",
    _PLUGIN_DIR / "data" / "grimoire.db",
]

_MACRO_RE = re.compile(r"[\[【]法典[:：]([^\]】]+)[\]】]")

# NAI5 的模型 slug 形如 nai-diffusion-5-full / nai-diffusion-5-curated。
# 不能用 ``"-5-" in name`` 这种子串判断：V4.5 的 slug 是
# ``nai-diffusion-4-5-full``，同样含 ``-5-``，会被误判成 NAI5，
# 导致 V4.5 的 18505 条词库一条都检索不到。
_NAI5_PATTERNS = (
    re.compile(r"diffusion[-_]5(?:$|[-_])"),
    re.compile(r"(?:^|[-_])nai[-_]?5(?:$|[-_])"),
)

# 词库解压后的最低可接受条目数；低于它视为解压不完整，需要重新解压。
_EXPECTED_MIN_ROWS = 19000
_extract_lock = threading.Lock()
# 进程内缓存已确认可用的库路径，避免每次检索都去开库计数。
_resolved_db_path: Path | None = None


def normalize_model_ver(model_name: str | None) -> str:
    """将模型名称归一化为词库 model_ver ('NAI4.5' 或 'NAI5')。"""
    m = str(model_name or "").strip().lower()
    if m in ("5", "v5", "nai5"):
        return "NAI5"
    if any(pattern.search(m) for pattern in _NAI5_PATTERNS):
        return "NAI5"
    return "NAI4.5"


def allowed_model_vers(model_name: str | None) -> list[str]:
    """当前模型可用的词库版本，按优先级排列（第一个是模型自身那一卷）。

    V5 是新的底模，能直接用 V4.5 的标签与画风，所以 V5 检索时要把 V4.5
    那 18,505 条一起算进来；反向不成立——V4.5 用不了 V5 专属的部分词条，
    因此 V4.5 只查自己这一卷。
    """

    if not model_name:
        return []
    locked = normalize_model_ver(model_name)
    if locked == "NAI5":
        return ["NAI5", "NAI4.5"]
    return [locked]


def _candidate_db_paths() -> list[Path]:
    """按优先级返回数据库落点：AstrBot 插件数据目录优先，其次插件内置 data/。"""
    paths: list[Path] = []
    data_dir = _plugin_data_dir()
    if data_dir is not None:
        paths.append(data_dir / "grimoire.db")
    for path in DEFAULT_DB_PATHS:
        if path not in paths:
            paths.append(path)
    return paths


def _plugin_data_dir() -> Path | None:
    """通过 AstrBot 官方接口解析插件数据目录，与插件内其它数据保持一致。"""
    try:
        from astrbot.api.star import StarTools

        return Path(StarTools.get_data_dir(_PLUGIN_NAME))
    except Exception:
        return None


def _is_valid_db(path: Path) -> bool:
    """库能打开、表存在且条目数正常，才算一个可用词库。"""
    try:
        if not path.is_file() or path.stat().st_size < 1_000_000:
            return False
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute("SELECT COUNT(*) FROM grimoire_entries").fetchone()[0]
        finally:
            conn.close()
        return rows >= _EXPECTED_MIN_ROWS
    except Exception:
        return False


def _extract_db(gz_path: Path, target: Path) -> bool:
    """解压到临时文件，校验通过后再原子替换，避免留下半截库。"""
    import gzip
    import shutil

    # 可能已被其它线程/进程解压完成，直接复用。
    if _is_valid_db(target):
        return True

    tmp_path = target.parent / (
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    ok = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        if _is_valid_db(tmp_path):
            os.replace(tmp_path, target)
            ok = True
    except Exception:
        ok = False
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
    # 并发竞争时另一路可能已经把目标写好，此时同样视为成功。
    return ok or _is_valid_db(target)


def get_grimoire_db_path() -> Path | None:
    """返回可用词库路径；缺失或损坏时自动重新解压（幂等、并发安全）。"""
    global _resolved_db_path

    # 已确认过的库直接复用；只在文件消失时才重新解析。
    if _resolved_db_path is not None and _resolved_db_path.is_file():
        return _resolved_db_path

    candidates = _candidate_db_paths()
    for path in candidates:
        if _is_valid_db(path):
            _resolved_db_path = path
            return path

    gz_path = _PLUGIN_DIR / "data" / "grimoire.db.gz"
    if not gz_path.is_file():
        return None

    with _extract_lock:
        # 等锁期间可能已被其它线程解压完成。
        for path in candidates:
            if _is_valid_db(path):
                _resolved_db_path = path
                return path
        for target in candidates:
            if _extract_db(gz_path, target):
                _resolved_db_path = target
                return target
    return None


def extract_clean_positive_tags(raw_tags: str) -> str:
    """从法典原始词条中提取干净的正向 Tag（剥离负面提示词、Steps设置行等）。"""
    lines = []
    for line in (raw_tags or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("负面", "negative", "Steps:", "Sampler:", "Prompt Guidance", "设置")):
            continue
        s = re.sub(r"^char\d+[:：]\s*", "", s, flags=re.IGNORECASE)
        lines.append(s.rstrip(",， "))
    return ", ".join(lines)


def search_grimoire(
    query: str = "",
    limit: int = 12,
    model_ver: str | None = None,
) -> list[dict[str, Any]]:
    """检索内置 NovelAI 法典词库，按当前生图模型版本过滤。

    V5 会同时检索 V5 与 V4.5 两卷（新底模兼容旧词条），同名时优先返回 V5 的
    那一条；V4.5 只查自己这一卷（用不了 V5 专属词条）。
    不做 sfw/nsfw 分级过滤：词条本身就是用户点名要查的内容，多一个分级开关
    只会让模型或用户在不知情时漏掉整卷词库。
    """
    db_path = get_grimoire_db_path()
    if not db_path:
        return []

    q = (query or "").strip()
    limit = max(1, min(int(limit or 12), 50))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    allowed_vers = allowed_model_vers(model_ver)
    preferred_ver = allowed_vers[0] if allowed_vers else None

    where_clauses: list[str] = []
    params: list[Any] = []
    if allowed_vers:
        placeholders = ", ".join("?" for _ in allowed_vers)
        where_clauses.append(f"model_ver IN ({placeholders})")
        params.extend(allowed_vers)

    tokens = [t.strip() for t in re.split(r"[\s,，/]+", q) if t.strip()]
    for tok in tokens[:5]:
        where_clauses.append("(title LIKE ? OR category LIKE ? OR tags LIKE ?)")
        like_tok = f"%{tok}%"
        params.extend([like_tok, like_tok, like_tok])

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    if tokens:
        first_tok = tokens[0]
        # 同名同分时，模型自己那一卷排在前面（V5 模型优先 V5 词条，再回退 V4.5）。
        version_bonus_sql = ""
        score_params = [
            first_tok,
            f"{first_tok}%",
            f"%{first_tok}%",
            f"%{first_tok}%",
        ]
        if preferred_ver:
            version_bonus_sql = "\n                 + CASE WHEN model_ver = ? THEN 15 ELSE 0 END"
            score_params.append(preferred_ver)
        sql = f"""
            SELECT id, volume, rating, model_ver, category, title, tags,
                (CASE WHEN title = ? THEN 100
                      WHEN title LIKE ? THEN 60
                      WHEN title LIKE ? THEN 40
                      ELSE 0 END
                 + CASE WHEN category LIKE ? THEN 20 ELSE 0 END{version_bonus_sql}) AS relevance
            FROM grimoire_entries
            {where_sql}
            ORDER BY relevance DESC, length(title) ASC, id ASC
            LIMIT ?
        """
        rows = cur.execute(sql, score_params + params + [limit]).fetchall()
    else:
        sql = f"""
            SELECT id, volume, rating, model_ver, category, title, tags, 10 AS relevance
            FROM grimoire_entries
            {where_sql}
            ORDER BY id ASC
            LIMIT ?
        """
        rows = cur.execute(sql, params + [limit]).fetchall()

    conn.close()
    results = []
    for r in rows:
        raw_tags = r["tags"] or ""
        results.append({
            "id": r["id"],
            "volume": r["volume"],
            "rating": r["rating"],
            "model_ver": r["model_ver"],
            "category": r["category"],
            "title": r["title"],
            "tags": raw_tags,
            "clean_tags": extract_clean_positive_tags(raw_tags),
        })
    return results


def expand_grimoire_macros(prompt: str, model_ver: str | None = None) -> str:
    """展开提示词中的 [法典:词条名] 或 【法典:词条名】 宏（严格限定在当前模型版本词库内）。"""
    if not prompt or ("法典:" not in prompt and "法典：" not in prompt):
        return prompt

    def _repl(m: re.Match) -> str:
        kw = m.group(1).strip()
        hits = search_grimoire(kw, limit=1, model_ver=model_ver)
        if hits:
            return hits[0]["clean_tags"]
        return kw

    return _MACRO_RE.sub(_repl, prompt)


def build_grimoire_translation_hint(
    natural_text: str, max_matches: int = 5, model_ver: str | None = None
) -> str:
    """从自然语言描述中提取命中的法典词条，供转译模型参考。

版本规则与检索一致：V5 模型同时参考 V5 与 V4.5 词条，V4.5 只参考自己那卷。
"""
    if not natural_text or len(natural_text.strip()) < 2:
        return ""
    db_path = get_grimoire_db_path()
    if not db_path:
        return ""

    text = natural_text.strip()
    cn_chunks = re.findall(r"[\u4e00-\u9fff]{2,12}", text)
    if not cn_chunks:
        return ""

    allowed_vers = allowed_model_vers(model_ver) or ["NAI4.5"]
    locked_ver = allowed_vers[0]

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    matched_rows = []
    seen_ids = set()

    placeholders = ", ".join("?" for _ in allowed_vers)
    rows = cur.execute(
        f"""
        SELECT id, volume, category, title, tags
        FROM grimoire_entries
        WHERE model_ver IN ({placeholders})
          AND length(title) BETWEEN 2 AND 14
          AND instr(?, title) > 0
        ORDER BY length(title) DESC, id ASC
        LIMIT 12
        """,
        (*allowed_vers, text),
    ).fetchall()

    for r in rows:
        t = r["title"]
        if t in ("其他", "杂项", "正面", "侧面", "背面", "室内", "室外", "日常", "内衣", "泳装", "睡衣"):
            continue
        if r["id"] not in seen_ids:
            seen_ids.add(r["id"])
            matched_rows.append(r)
            if len(matched_rows) >= max_matches:
                break

    conn.close()
    if not matched_rows:
        return ""

    ver_label = "NAI5 + 兼容的 NAI4.5" if locked_ver == "NAI5" else locked_ver
    lines = [f"\n\n【NovelAI ({ver_label}) 法典命中词条参考（若与用户意图一致，请优先采用或融合以下经过实测的标签）】："]
    for r in matched_rows:
        clean = extract_clean_positive_tags(r["tags"])
        if len(clean) > 260:
            clean = clean[:260] + "..."
        lines.append(f"- 《{r['title']}》({r['volume']} / {r['category']}): {clean}")
    return "\n".join(lines)
