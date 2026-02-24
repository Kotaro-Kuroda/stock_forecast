from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from forecast_service import screen_candidates
from paper_trading import PaperTrader


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AutoTradingAgent:
    def __init__(self, state_path: Path, paper_trader: PaperTrader) -> None:
        self.state_path = state_path
        self.paper_trader = paper_trader
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = self._load()

    def _default_state(self) -> dict[str, Any]:
        return {
            "running": False,
            "mode": "paper",  # paper | alpaca_paper | alpaca_live | ibkr_paper | ibkr_live
            "config": {
                "universe": "米国主要",
                "sector": "すべて",
                "top_n": 5,
                "max_positions": 5,
                "buy_threshold_pct": 1.0,
                "stop_loss_pct": 5.0,
                "position_size_pct": 15.0,
                "interval_sec": 900,
                "rf_feature_flags": {},
            },
            "cycle_count": 0,
            "last_run_at": None,
            "last_error": "",
            "last_result": {},
            "logs": [],
        }

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            # Process restart時はスレッドが復元できないため、安全側で停止状態に戻す。
            state["running"] = False
            return state
        except Exception:
            return self._default_state()

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _append_log(self, level: str, message: str) -> None:
        log = {"time": now_iso(), "level": level, "message": message}
        self.state["logs"].append(log)
        self.state["logs"] = self.state["logs"][-200:]

    @staticmethod
    def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
        return {
            "universe": str(config.get("universe", "米国主要")),
            "sector": str(config.get("sector", "すべて")),
            "top_n": max(1, min(30, int(config.get("top_n", 5)))),
            "max_positions": max(1, min(30, int(config.get("max_positions", 5)))),
            "buy_threshold_pct": float(config.get("buy_threshold_pct", 1.0)),
            "stop_loss_pct": abs(float(config.get("stop_loss_pct", 5.0))),
            "position_size_pct": max(1.0, min(50.0, float(config.get("position_size_pct", 15.0)))),
            "interval_sec": max(10, int(config.get("interval_sec", 900))),
            "rf_feature_flags": dict(config.get("rf_feature_flags", {})),
        }

    @staticmethod
    def _is_us_equity_symbol(symbol: str) -> bool:
        upper = symbol.upper()
        if upper.endswith(".T"):
            return False
        if upper.endswith("-USD"):
            return False
        return upper.isalnum()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        m = (mode or "").strip().lower()
        if m in {"paper"}:
            return "paper"
        if m in {"alpaca", "alpaca_paper"}:
            return "alpaca_paper"
        if m in {"alpaca_live", "live"}:
            return "alpaca_live"
        if m in {"ibkr_paper"}:
            return "ibkr_paper"
        if m in {"ibkr_live"}:
            return "ibkr_live"
        raise ValueError(
            "mode は paper / alpaca_paper / alpaca_live / ibkr_paper / ibkr_live のいずれかを指定してください。"
        )

    def _alpaca_headers(self) -> dict[str, str]:
        key = os.getenv("ALPACA_API_KEY", "").strip()
        secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not key or not secret:
            raise RuntimeError(
                "Alpaca APIキーが未設定です。環境変数 ALPACA_API_KEY / ALPACA_SECRET_KEY を設定してください。"
            )
        return {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _alpaca_base_url(mode: str) -> str:
        if mode == "alpaca_live":
            return os.getenv("ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets").rstrip("/")
        return os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")

    def _alpaca_request(
        self, mode: str, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        url = f"{self._alpaca_base_url(mode)}{path}"
        resp = requests.request(
            method=method.upper(),
            url=url,
            headers=self._alpaca_headers(),
            json=payload,
            timeout=20,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca APIエラー {resp.status_code}: {resp.text}")
        try:
            return resp.json()
        except Exception:
            return {}

    def _alpaca_account(self, mode: str) -> dict[str, Any]:
        return self._alpaca_request(mode, "GET", "/v2/account")

    def _alpaca_positions(self, mode: str) -> list[dict[str, Any]]:
        data = self._alpaca_request(mode, "GET", "/v2/positions")
        return data if isinstance(data, list) else []

    def _alpaca_submit_market_notional_buy(
        self, mode: str, symbol: str, notional_usd: float
    ) -> dict[str, Any]:
        return self._alpaca_request(
            mode,
            "POST",
            "/v2/orders",
            {
                "symbol": symbol,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "notional": round(float(notional_usd), 2),
            },
        )

    def _alpaca_submit_market_sell(self, mode: str, symbol: str, qty: float) -> dict[str, Any]:
        return self._alpaca_request(
            mode,
            "POST",
            "/v2/orders",
            {
                "symbol": symbol,
                "side": "sell",
                "type": "market",
                "time_in_force": "day",
                "qty": str(round(float(qty), 6)),
            },
        )

    @staticmethod
    def _ibkr_host() -> str:
        return os.getenv("IBKR_HOST", "127.0.0.1")

    @staticmethod
    def _ibkr_port(mode: str) -> int:
        if mode == "ibkr_live":
            return int(os.getenv("IBKR_PORT_LIVE", "7496"))
        return int(os.getenv("IBKR_PORT_PAPER", "7497"))

    @staticmethod
    def _ibkr_client_id() -> int:
        return int(os.getenv("IBKR_CLIENT_ID", "101"))

    def _get_ib(self, mode: str):
        try:
            from ib_insync import IB  # type: ignore
        except Exception as exc:
            raise RuntimeError("ib_insync が利用できません。`uv sync` を実行してください。") from exc
        ib = IB()
        ib.connect(
            self._ibkr_host(),
            self._ibkr_port(mode),
            clientId=self._ibkr_client_id(),
            timeout=8,
        )
        return ib

    @staticmethod
    def _is_jp_symbol(symbol: str) -> bool:
        return symbol.upper().endswith(".T")

    @staticmethod
    def _is_supported_for_ibkr(symbol: str) -> bool:
        upper = symbol.upper()
        if upper.endswith("-USD"):
            return False
        return upper.isalnum() or upper.endswith(".T")

    @staticmethod
    def _to_symbol_from_ibkr_contract(contract: Any) -> str:
        sym = str(getattr(contract, "symbol", "")).upper()
        cur = str(getattr(contract, "currency", "")).upper()
        exch = str(getattr(contract, "primaryExchange", "")).upper()
        if cur == "JPY" or "TSE" in exch:
            return f"{sym}.T"
        return sym

    def _ibkr_contract(self, symbol: str):
        try:
            from ib_insync import Stock  # type: ignore
        except Exception as exc:
            raise RuntimeError("ib_insync が利用できません。`uv sync` を実行してください。") from exc
        upper = symbol.upper()
        if self._is_jp_symbol(upper):
            code = upper.split(".")[0]
            return Stock(code, "SMART", "JPY", primaryExchange="TSEJ")
        return Stock(upper, "SMART", "USD")

    def _step_live_ibkr(self, mode: str, config: dict[str, Any], rf_feature_flags: dict) -> dict[str, Any]:
        rows = screen_candidates(
            universe=config["universe"],
            top_n=max(config["top_n"], 10),
            rf_feature_flags=rf_feature_flags,
            sector=config.get("sector", "すべて"),
        )
        row_map = {r["symbol"].upper(): r for r in rows}
        ranked = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[: config["top_n"]]

        ib = self._get_ib(mode)
        try:
            values = ib.accountValues()
            by_tag = {}
            for v in values:
                k = (str(getattr(v, "tag", "")), str(getattr(v, "currency", "")))
                by_tag[k] = float(getattr(v, "value", "0") or 0.0)

            equity = by_tag.get(("NetLiquidation", "USD"), by_tag.get(("NetLiquidation", "BASE"), 0.0))
            cash = by_tag.get(("AvailableFunds", "USD"), by_tag.get(("AvailableFunds", "BASE"), 0.0))
            if equity <= 0:
                equity = cash
            alloc = equity * (float(config["position_size_pct"]) / 100.0)

            raw_positions = ib.positions()
            pos_map: dict[str, Any] = {}
            for p in raw_positions:
                symbol = self._to_symbol_from_ibkr_contract(p.contract)
                pos_map[symbol.upper()] = p

            executed = []

            # Exit logic
            for symbol, pos in list(pos_map.items()):
                if symbol not in row_map:
                    continue
                row = row_map[symbol]
                expected = float(row.get("expected_return", 0.0))
                avg_cost = float(getattr(pos, "avgCost", 0.0))
                current = float(row.get("last_price", avg_cost))
                if avg_cost <= 0:
                    avg_cost = current
                drawdown = (current / max(avg_cost, 1e-9) - 1.0) * 100.0
                if expected < 0.0 or drawdown <= -abs(float(config["stop_loss_pct"])):
                    qty = abs(float(getattr(pos, "position", 0.0)))
                    if qty >= 1.0:
                        qty_int = int(math.floor(qty))
                        if qty_int < 1:
                            continue
                        contract = self._ibkr_contract(symbol)
                        from ib_insync import MarketOrder  # type: ignore

                        trade = ib.placeOrder(contract, MarketOrder("SELL", qty_int))
                        ib.sleep(0.2)
                        executed.append(
                            {
                                "action": "SELL",
                                "symbol": symbol,
                                "qty": qty_int,
                                "reason": "risk_or_negative_signal",
                                "status": str(getattr(trade.orderStatus, "status", "")),
                            }
                        )

            # Buy logic
            raw_positions = ib.positions()
            current_symbols = {self._to_symbol_from_ibkr_contract(p.contract).upper() for p in raw_positions}
            for row in ranked:
                if len(current_symbols) >= int(config["max_positions"]):
                    break
                symbol = str(row.get("symbol", "")).upper()
                if not self._is_supported_for_ibkr(symbol):
                    continue
                if symbol in current_symbols:
                    continue
                expected = float(row.get("expected_return", 0.0))
                if expected < float(config["buy_threshold_pct"]):
                    continue
                price = float(row.get("last_price", 0.0))
                if price <= 0:
                    continue
                spend = min(alloc, cash)
                qty = int(math.floor(spend / price))
                if qty < 1:
                    continue
                contract = self._ibkr_contract(symbol)
                from ib_insync import MarketOrder  # type: ignore

                trade = ib.placeOrder(contract, MarketOrder("BUY", qty))
                ib.sleep(0.2)
                executed.append(
                    {
                        "action": "BUY",
                        "symbol": symbol,
                        "qty": qty,
                        "reason": "positive_signal",
                        "status": str(getattr(trade.orderStatus, "status", "")),
                    }
                )
                cash -= qty * price
                current_symbols.add(symbol)

            return {
                "broker": mode,
                "ibkr_endpoint": f"{self._ibkr_host()}:{self._ibkr_port(mode)}",
                "account": {
                    "cash_usd": float(cash),
                    "equity_usd": float(equity),
                },
                "executed_trades": executed,
                "candidate_count": len(rows),
            }
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    def _step_live_alpaca(self, mode: str, config: dict[str, Any], rf_feature_flags: dict) -> dict[str, Any]:
        rows = screen_candidates(
            universe=config["universe"],
            top_n=max(config["top_n"], 10),
            rf_feature_flags=rf_feature_flags,
            sector=config.get("sector", "すべて"),
        )
        row_map = {r["symbol"]: r for r in rows}
        ranked = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[: config["top_n"]]

        account = self._alpaca_account(mode)
        positions = self._alpaca_positions(mode)
        pos_map = {str(p.get("symbol", "")).upper(): p for p in positions}

        equity_usd = float(account.get("equity", account.get("cash", 0.0)))
        cash_usd = float(account.get("cash", 0.0))
        alloc_usd = equity_usd * (float(config["position_size_pct"]) / 100.0)
        executed = []

        # Exit
        for symbol, pos in list(pos_map.items()):
            if not self._is_us_equity_symbol(symbol):
                continue
            expected = float(row_map.get(symbol, {}).get("expected_return", 0.0))
            avg_entry = float(pos.get("avg_entry_price", 0.0))
            current = float(pos.get("current_price", avg_entry))
            drawdown = (current / max(avg_entry, 1e-9) - 1.0) * 100.0
            if expected < 0.0 or drawdown <= -abs(float(config["stop_loss_pct"])):
                qty = float(pos.get("qty", 0.0))
                if qty > 0:
                    order = self._alpaca_submit_market_sell(mode, symbol, qty)
                    executed.append(
                        {
                            "action": "SELL",
                            "symbol": symbol,
                            "qty": qty,
                            "reason": "risk_or_negative_signal",
                            "order_id": order.get("id"),
                        }
                    )

        # Buy
        current_positions = self._alpaca_positions(mode)
        current_map = {str(p.get("symbol", "")).upper(): p for p in current_positions}
        for row in ranked:
            if len(current_map) >= int(config["max_positions"]):
                break
            symbol = str(row.get("symbol", "")).upper()
            if not self._is_us_equity_symbol(symbol):
                continue
            if symbol in current_map:
                continue
            expected = float(row.get("expected_return", 0.0))
            if expected < float(config["buy_threshold_pct"]):
                continue
            spend = min(alloc_usd, cash_usd)
            if spend < 20.0:
                break
            order = self._alpaca_submit_market_notional_buy(mode, symbol, spend)
            cash_usd -= spend
            current_map[symbol] = {"symbol": symbol}
            executed.append(
                {
                    "action": "BUY",
                    "symbol": symbol,
                    "notional_usd": round(spend, 2),
                    "reason": "positive_signal",
                    "order_id": order.get("id"),
                }
            )

        latest_account = self._alpaca_account(mode)
        return {
            "broker": mode,
            "broker_url": self._alpaca_base_url(mode),
            "account": {
                "cash_usd": float(latest_account.get("cash", 0.0)),
                "equity_usd": float(latest_account.get("equity", 0.0)),
            },
            "executed_trades": executed,
            "candidate_count": len(rows),
        }

    def step_once(self, rf_feature_flags: dict | None = None) -> dict[str, Any]:
        config = self._validate_config(self.state.get("config", {}))
        flags = rf_feature_flags if rf_feature_flags is not None else dict(config.get("rf_feature_flags", {}))
        mode = self._normalize_mode(str(self.state.get("mode", "paper")))
        if mode == "paper":
            result = self.paper_trader.step(
                universe=config["universe"],
                top_n=config["top_n"],
                rf_feature_flags=flags,
                max_positions=config["max_positions"],
                buy_threshold_pct=config["buy_threshold_pct"],
                stop_loss_pct=config["stop_loss_pct"],
                position_size_pct=config["position_size_pct"],
                sector=config.get("sector", "すべて"),
            )
            return {"mode": "paper", "result": result}
        if mode in {"alpaca_paper", "alpaca_live"}:
            result = self._step_live_alpaca(mode, config, flags)
            return {"mode": mode, "result": result}
        if mode in {"ibkr_paper", "ibkr_live"}:
            result = self._step_live_ibkr(mode, config, flags)
            return {"mode": mode, "result": result}
        raise ValueError("未知のモードです。")

    def _run_loop(self) -> None:
        while not self.stop_event.is_set():
            interval = int(self.state.get("config", {}).get("interval_sec", 900))
            try:
                out = self.step_once()
                with self.lock:
                    self.state["last_result"] = out
                    self.state["last_run_at"] = now_iso()
                    self.state["cycle_count"] = int(self.state.get("cycle_count", 0)) + 1
                    self.state["last_error"] = ""
                    self._append_log("INFO", f"cycle={self.state['cycle_count']} mode={out.get('mode')}")
                    self._save()
            except Exception as exc:
                with self.lock:
                    self.state["last_error"] = str(exc)
                    self._append_log("ERROR", str(exc))
                    self._save()
            self.stop_event.wait(max(10, interval))

    def start(self, mode: str, config: dict[str, Any]) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        with self.lock:
            if self.state.get("running"):
                return self.status()
            self.state["mode"] = normalized_mode
            self.state["config"] = self._validate_config(config)
            self.state["running"] = True
            self.state["last_error"] = ""
            self._append_log("INFO", f"agent started mode={normalized_mode}")
            self._save()
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.state["running"] = False
            self._append_log("INFO", "agent stopped")
            self._save()
            self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        return self.status()

    def status(self) -> dict[str, Any]:
        with self.lock:
            mode = self._normalize_mode(str(self.state.get("mode", "paper")))
            data = {
                "running": bool(self.state.get("running", False)),
                "mode": mode,
                "config": self._validate_config(self.state.get("config", {})),
                "cycle_count": int(self.state.get("cycle_count", 0)),
                "last_run_at": self.state.get("last_run_at"),
                "last_error": self.state.get("last_error", ""),
                "last_result": self.state.get("last_result", {}),
                "logs": self.state.get("logs", [])[-50:],
            }
        try:
            if data["mode"] == "paper":
                data["paper_status"] = self.paper_trader.status()
            elif data["mode"] in {"alpaca_paper", "alpaca_live"}:
                account = self._alpaca_account(data["mode"])
                data["alpaca_account"] = {
                    "cash_usd": float(account.get("cash", 0.0)),
                    "equity_usd": float(account.get("equity", 0.0)),
                    "buying_power": float(account.get("buying_power", 0.0)),
                }
                data["alpaca_base_url"] = self._alpaca_base_url(data["mode"])
            elif data["mode"] in {"ibkr_paper", "ibkr_live"}:
                data["ibkr_endpoint"] = f"{self._ibkr_host()}:{self._ibkr_port(data['mode'])}"
        except Exception as exc:
            data["account_error"] = str(exc)
        return data
