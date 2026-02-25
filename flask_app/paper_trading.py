from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .forecast_service import get_yf, screen_candidates
except ImportError:
    from forecast_service import get_yf, screen_candidates


@dataclass
class Trade:
    action: str
    symbol: str
    name: str
    qty: float
    price_jpy: float
    price_native: float
    currency: str
    fx_rate: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "name": self.name,
            "qty": round(self.qty, 6),
            "price": round(self.price_jpy, 4),
            "price_jpy": round(self.price_jpy, 4),
            "price_native": round(self.price_native, 4),
            "currency": self.currency,
            "fx_rate": round(self.fx_rate, 4),
            "reason": self.reason,
        }


class PaperTrader:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.state = self._load()

    def _default_state(self) -> dict[str, Any]:
        return {
            "cash": 1_000_000.0,
            "positions": {},
            "trades": [],
            "last_universe": "日本主要",
            "last_rows": [],
            "usd_jpy": 150.0,
        }

    @staticmethod
    def _detect_currency(symbol: str) -> str:
        if symbol.upper().endswith(".T"):
            return "JPY"
        return "USD"

    @staticmethod
    def _to_jpy(price_native: float, currency: str, usd_jpy: float) -> float:
        if currency == "JPY":
            return float(price_native)
        return float(price_native) * float(usd_jpy)

    @staticmethod
    def _extract_close(data: pd.DataFrame) -> float | None:
        if data.empty:
            return None
        close_data = None
        if isinstance(data.columns, pd.MultiIndex):
            matches = [c for c in data.columns if c[0] == "Close"]
            if matches:
                close_data = data[matches[0]]
        elif "Close" in data.columns:
            close_data = data["Close"]
        if close_data is None:
            return None
        if isinstance(close_data, pd.DataFrame):
            if close_data.shape[1] == 0:
                return None
            close_data = close_data.iloc[:, 0]
        close_series = pd.Series(close_data).dropna().astype(float)
        if close_series.empty:
            return None
        return float(close_series.iloc[-1])

    def _fetch_usd_jpy(self) -> float:
        try:
            data = get_yf().download("JPY=X", period="5d", progress=False, auto_adjust=True)
            fx = self._extract_close(data)
            if fx and fx > 0:
                return fx
        except Exception:
            pass
        return float(self.state.get("usd_jpy", 150.0))

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_state()

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def reset(self, initial_cash: float = 1_000_000.0) -> dict[str, Any]:
        self.state = self._default_state()
        self.state["cash"] = float(initial_cash)
        self._save()
        return self.status()

    def status(self) -> dict[str, Any]:
        usd_jpy = float(self.state.get("usd_jpy", 150.0))
        positions = []
        total_market_value = 0.0
        for symbol, pos in self.state["positions"].items():
            currency = str(pos.get("currency", self._detect_currency(symbol)))
            fx_rate = usd_jpy if currency == "USD" else 1.0
            qty = float(pos.get("qty", 0.0))
            avg_price_native = float(pos.get("avg_price_native", pos.get("avg_price", 0.0)))
            last_price_native = float(pos.get("last_price_native", pos.get("last_price", avg_price_native)))
            avg_price_jpy = float(pos.get("avg_price_jpy", self._to_jpy(avg_price_native, currency, fx_rate)))
            last_price_jpy = float(pos.get("last_price_jpy", self._to_jpy(last_price_native, currency, fx_rate)))
            market_value = qty * last_price_jpy
            total_market_value += market_value
            positions.append(
                {
                    "symbol": symbol,
                    "name": pos.get("name", symbol),
                    "currency": currency,
                    "fx_rate": round(fx_rate, 4),
                    "qty": round(qty, 6),
                    "avg_price": round(avg_price_jpy, 4),
                    "last_price": round(last_price_jpy, 4),
                    "avg_price_jpy": round(avg_price_jpy, 4),
                    "last_price_jpy": round(last_price_jpy, 4),
                    "avg_price_native": round(avg_price_native, 4),
                    "last_price_native": round(last_price_native, 4),
                    "market_value": round(market_value, 2),
                    "pnl": round((last_price_jpy - avg_price_jpy) * qty, 2),
                }
            )

        equity = float(self.state["cash"]) + total_market_value
        return {
            "cash": round(float(self.state["cash"]), 2),
            "equity": round(equity, 2),
            "usd_jpy": round(usd_jpy, 4),
            "positions": positions,
            "trades": self.state["trades"][-30:],
            "last_universe": self.state.get("last_universe", "日本主要"),
        }

    def step(
        self,
        universe: str,
        top_n: int,
        rf_feature_flags: dict,
        max_positions: int,
        buy_threshold_pct: float,
        stop_loss_pct: float,
        position_size_pct: float,
        sector: str = "すべて",
    ) -> dict[str, Any]:
        rows = screen_candidates(
            universe=universe,
            top_n=max(top_n, 10),
            rf_feature_flags=rf_feature_flags,
            sector=sector,
        )
        self.state["last_universe"] = universe
        self.state["last_rows"] = rows
        self.state["usd_jpy"] = self._fetch_usd_jpy()
        usd_jpy = float(self.state["usd_jpy"])
        row_map = {r["symbol"]: r for r in rows}

        trades: list[Trade] = []
        positions = self.state["positions"]

        # Exit logic
        for symbol in list(positions.keys()):
            pos = positions[symbol]
            currency = str(pos.get("currency", self._detect_currency(symbol)))
            fx_rate = usd_jpy if currency == "USD" else 1.0
            market_price_native = float(
                row_map.get(symbol, {}).get(
                    "last_price",
                    pos.get("last_price_native", pos.get("last_price", pos.get("avg_price_native", pos.get("avg_price", 0.0)))),
                )
            )
            market_price_jpy = self._to_jpy(market_price_native, currency, fx_rate)
            pos["currency"] = currency
            pos["fx_rate"] = fx_rate
            pos["last_price_native"] = market_price_native
            pos["last_price_jpy"] = market_price_jpy
            pos["last_price"] = market_price_jpy
            avg_price_native = float(pos.get("avg_price_native", pos.get("avg_price", 0.0)))
            avg_price_jpy = float(pos.get("avg_price_jpy", self._to_jpy(avg_price_native, currency, fx_rate)))
            pos["avg_price_native"] = avg_price_native
            pos["avg_price_jpy"] = avg_price_jpy
            pos["avg_price"] = avg_price_jpy
            drawdown = (market_price_native / max(avg_price_native, 1e-9) - 1.0) * 100.0
            expected_return = float(row_map.get(symbol, {}).get("expected_return", 0.0))
            should_exit = drawdown <= -abs(stop_loss_pct) or expected_return < 0
            if should_exit:
                qty = float(pos["qty"])
                proceeds = qty * market_price_jpy
                self.state["cash"] += proceeds
                trades.append(
                    Trade(
                        "SELL",
                        symbol,
                        pos.get("name", symbol),
                        qty,
                        market_price_jpy,
                        market_price_native,
                        currency,
                        fx_rate,
                        "risk_or_negative_signal",
                    )
                )
                del positions[symbol]

        # Buy logic
        status = self.status()
        equity = float(status["equity"])
        max_pos_value = equity * max(0.01, min(position_size_pct / 100.0, 0.5))
        buy_threshold = float(buy_threshold_pct)

        ranked = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:top_n]
        for row in ranked:
            if len(positions) >= max_positions:
                break
            symbol = row["symbol"]
            if symbol in positions:
                currency = str(positions[symbol].get("currency", self._detect_currency(symbol)))
                fx_rate = usd_jpy if currency == "USD" else 1.0
                last_price_native = float(row["last_price"])
                positions[symbol]["last_price_native"] = last_price_native
                positions[symbol]["last_price_jpy"] = self._to_jpy(last_price_native, currency, fx_rate)
                positions[symbol]["last_price"] = positions[symbol]["last_price_jpy"]
                continue
            expected_return = float(row["expected_return"])
            if expected_return < buy_threshold:
                continue
            currency = self._detect_currency(symbol)
            fx_rate = usd_jpy if currency == "USD" else 1.0
            price_native = float(row["last_price"])
            price_jpy = self._to_jpy(price_native, currency, fx_rate)
            alloc = min(max_pos_value, float(self.state["cash"]))
            if alloc < price_jpy:
                continue
            qty = alloc / max(price_jpy, 1e-9)
            self.state["cash"] -= qty * price_jpy
            positions[symbol] = {
                "name": row.get("name", symbol),
                "currency": currency,
                "fx_rate": fx_rate,
                "qty": qty,
                "avg_price_native": price_native,
                "last_price_native": price_native,
                "avg_price_jpy": price_jpy,
                "last_price_jpy": price_jpy,
                "avg_price": price_jpy,
                "last_price": price_jpy,
            }
            trades.append(
                Trade(
                    "BUY",
                    symbol,
                    row.get("name", symbol),
                    qty,
                    price_jpy,
                    price_native,
                    currency,
                    fx_rate,
                    "positive_signal",
                )
            )

        for t in trades:
            self.state["trades"].append(t.as_dict())
        self._save()
        result = self.status()
        result["executed_trades"] = [t.as_dict() for t in trades]
        return result
