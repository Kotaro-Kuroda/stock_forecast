from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.domain import SCREEN_UNIVERSES

STORE_PATH = Path(__file__).resolve().parent / "custom_universes.json"
ALLOWED_UNIVERSES = {"米国主要", "日本主要"}


def _default_store() -> dict[str, list[dict[str, str]]]:
    return {k: [] for k in ALLOWED_UNIVERSES}


def _load_store() -> dict[str, list[dict[str, str]]]:
    if not STORE_PATH.exists():
        return _default_store()
    try:
        raw = json.loads(STORE_PATH.read_text(encoding="utf-8"))
        out = _default_store()
        for key in ALLOWED_UNIVERSES:
            rows = raw.get(key, [])
            if isinstance(rows, list):
                cleaned: list[dict[str, str]] = []
                for r in rows:
                    symbol = str(r.get("symbol", "")).strip().upper()
                    name = str(r.get("name", "")).strip() or symbol
                    if symbol:
                        cleaned.append({"symbol": symbol, "name": name})
                out[key] = cleaned
        return out
    except Exception:
        return _default_store()


def _save_store(store: dict[str, list[dict[str, str]]]) -> None:
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def list_custom_rows(universe: str) -> list[dict[str, str]]:
    if universe not in ALLOWED_UNIVERSES:
        raise ValueError("登録対象ユニバースは 米国主要 / 日本主要 のみです。")
    store = _load_store()
    return list(store.get(universe, []))


def add_custom_row(universe: str, symbol: str, name: str) -> dict[str, Any]:
    if universe not in ALLOWED_UNIVERSES:
        raise ValueError("登録対象ユニバースは 米国主要 / 日本主要 のみです。")
    s = symbol.strip().upper()
    n = name.strip() or s
    if not s:
        raise ValueError("symbol が空です。")
    store = _load_store()
    rows = store.get(universe, [])
    replaced = False
    for row in rows:
        if row["symbol"] == s:
            row["name"] = n
            replaced = True
            break
    if not replaced:
        rows.append({"symbol": s, "name": n})
    store[universe] = sorted(rows, key=lambda x: x["symbol"])
    _save_store(store)
    return {"ok": True, "replaced": replaced, "rows": store[universe]}


def remove_custom_row(universe: str, symbol: str) -> dict[str, Any]:
    if universe not in ALLOWED_UNIVERSES:
        raise ValueError("登録対象ユニバースは 米国主要 / 日本主要 のみです。")
    s = symbol.strip().upper()
    if not s:
        raise ValueError("symbol が空です。")
    store = _load_store()
    rows = store.get(universe, [])
    before = len(rows)
    rows = [r for r in rows if r["symbol"] != s]
    removed = len(rows) < before
    store[universe] = rows
    _save_store(store)
    return {"ok": True, "removed": removed, "rows": rows}


def get_universe_items(universe: str) -> list[tuple[str, str]]:
    base = list(SCREEN_UNIVERSES.get(universe, []))
    if universe not in ALLOWED_UNIVERSES:
        return base
    custom = list_custom_rows(universe)
    seen = {s.upper() for s, _ in base}
    merged = list(base)
    for row in custom:
        symbol = row["symbol"].upper()
        if symbol not in seen:
            merged.append((symbol, row["name"]))
            seen.add(symbol)
    return merged

