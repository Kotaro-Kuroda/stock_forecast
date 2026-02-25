from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.domain import DEFAULT_RF_FEATURE_FLAGS, SCREEN_UNIVERSES, SECTOR_OPTIONS
try:
    from .forecast_service import (
        forecast,
        get_company_name,
        optimize_for_symbol,
        resolve_symbol,
        screen_candidates,
    )
    from .paper_trading import PaperTrader
    from .auto_trading_agent import AutoTradingAgent
    from .decision_agent import decide_trade
    from .universe_store import add_custom_row, list_custom_rows, remove_custom_row
except ImportError:
    from forecast_service import forecast, get_company_name, optimize_for_symbol, resolve_symbol, screen_candidates
    from paper_trading import PaperTrader
    from auto_trading_agent import AutoTradingAgent
    from decision_agent import decide_trade
    from universe_store import add_custom_row, list_custom_rows, remove_custom_row

app = Flask(__name__)
paper_trader = PaperTrader(PROJECT_ROOT / "flask_app" / "paper_state.json")
auto_agent = AutoTradingAgent(PROJECT_ROOT / "flask_app" / "auto_agent_state.json", paper_trader)


@app.get("/")
def index():
    return render_template(
        "index.html",
        universes=list(SCREEN_UNIVERSES.keys()),
        sectors=SECTOR_OPTIONS,
        default_flags=DEFAULT_RF_FEATURE_FLAGS,
    )


@app.post("/api/forecast")
def api_forecast():
    payload = request.get_json(force=True) or {}
    try:
        result = forecast(
            raw_symbol=str(payload.get("symbol", "")).strip(),
            period=str(payload.get("period", "2y")).strip() or "2y",
            forecast_days=int(payload.get("forecast_days", 30)),
            lag=int(payload.get("lag", 20)),
            model=str(payload.get("model", "Ensemble")),
            daily=bool(payload.get("daily", False)),
            weekly=bool(payload.get("weekly", True)),
            yearly=bool(payload.get("yearly", True)),
            changepoint_prior_scale=float(payload.get("changepoint_prior_scale", 0.05)),
            seasonality_mode=str(payload.get("seasonality_mode", "additive")),
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
            auto_optimize=bool(payload.get("auto_optimize", False)),
            optimization_trials=int(payload.get("optimization_trials", 30)),
            fast_optimize=bool(payload.get("fast_optimize", True)),
        )
        return jsonify({"ok": True, "data": result.__dict__})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/optimize")
def api_optimize():
    payload = request.get_json(force=True) or {}
    try:
        result = optimize_for_symbol(
            raw_symbol=str(payload.get("symbol", "")).strip(),
            period=str(payload.get("period", "2y")).strip() or "2y",
            model=str(payload.get("model", "Ensemble")),
            lag=int(payload.get("lag", 20)),
            daily=bool(payload.get("daily", False)),
            weekly=bool(payload.get("weekly", True)),
            yearly=bool(payload.get("yearly", True)),
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
            n_trials=int(payload.get("optimization_trials", 30)),
            fast_mode=bool(payload.get("fast_optimize", True)),
        )
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/decision")
def api_decision():
    payload = request.get_json(force=True) or {}
    try:
        result = decide_trade(
            raw_symbol=str(payload.get("symbol", "")).strip(),
            period=str(payload.get("period", "2y")).strip() or "2y",
            forecast_days=int(payload.get("forecast_days", 30)),
            lag=int(payload.get("lag", 20)),
            model=str(payload.get("model", "Ensemble")),
            daily=bool(payload.get("daily", False)),
            weekly=bool(payload.get("weekly", True)),
            yearly=bool(payload.get("yearly", True)),
            changepoint_prior_scale=float(payload.get("changepoint_prior_scale", 0.05)),
            seasonality_mode=str(payload.get("seasonality_mode", "additive")),
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
            auto_optimize=bool(payload.get("auto_optimize", False)),
            optimization_trials=int(payload.get("optimization_trials", 30)),
            fast_optimize=bool(payload.get("fast_optimize", True)),
            min_expected_return_pct=float(payload.get("min_expected_return_pct", 2.0)),
            max_rmse_pct=float(payload.get("max_rmse_pct", 5.0)),
            max_mae_pct=float(payload.get("max_mae_pct", 3.0)),
            use_openai=bool(payload.get("use_openai", False)),
            openai_model=str(payload.get("openai_model", "gpt-4.1-mini")),
        )
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/screen")
def api_screen():
    payload = request.get_json(force=True) or {}
    try:
        rows = screen_candidates(
            universe=str(payload.get("universe", "日本主要")),
            top_n=int(payload.get("top_n", 5)),
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
            sector=str(payload.get("sector", "すべて")),
        )
        return jsonify({"ok": True, "rows": rows})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/universe/list")
def api_universe_list():
    universe = str(request.args.get("universe", "米国主要"))
    try:
        rows = list_custom_rows(universe)
        return jsonify({"ok": True, "universe": universe, "rows": rows})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/universe/add")
def api_universe_add():
    payload = request.get_json(force=True) or {}
    universe = str(payload.get("universe", "米国主要"))
    raw_symbol = str(payload.get("symbol", "")).strip()
    raw_name = str(payload.get("name", "")).strip()
    try:
        resolved = resolve_symbol(raw_symbol)
        if universe == "米国主要" and resolved.endswith(".T"):
            raise ValueError("米国主要には米国株ティッカーを登録してください（例: AAPL）。")
        if universe == "日本主要" and not resolved.endswith(".T"):
            raise ValueError("日本主要には日本株コードを登録してください（例: 7203.T または 7203）。")
        name = raw_name or get_company_name(resolved, raw_symbol)
        out = add_custom_row(universe=universe, symbol=resolved, name=name)
        return jsonify({"ok": True, "universe": universe, "symbol": resolved, "name": name, **out})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/universe/remove")
def api_universe_remove():
    payload = request.get_json(force=True) or {}
    universe = str(payload.get("universe", "米国主要"))
    symbol = str(payload.get("symbol", "")).strip()
    try:
        out = remove_custom_row(universe=universe, symbol=symbol)
        return jsonify({"ok": True, "universe": universe, "symbol": symbol.upper(), **out})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/paper/status")
def api_paper_status():
    try:
        return jsonify({"ok": True, "data": paper_trader.status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/paper/reset")
def api_paper_reset():
    payload = request.get_json(force=True) or {}
    try:
        initial_cash = float(payload.get("initial_cash", 1_000_000))
        return jsonify({"ok": True, "data": paper_trader.reset(initial_cash=initial_cash)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/paper/step")
def api_paper_step():
    payload = request.get_json(force=True) or {}
    try:
        data = paper_trader.step(
            universe=str(payload.get("universe", "日本主要")),
            top_n=int(payload.get("top_n", 5)),
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
            max_positions=int(payload.get("max_positions", 5)),
            buy_threshold_pct=float(payload.get("buy_threshold_pct", 1.0)),
            stop_loss_pct=float(payload.get("stop_loss_pct", 5.0)),
            position_size_pct=float(payload.get("position_size_pct", 15.0)),
            sector=str(payload.get("sector", "すべて")),
        )
        return jsonify({"ok": True, "data": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.get("/api/agent/status")
def api_agent_status():
    try:
        return jsonify({"ok": True, "data": auto_agent.status()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/agent/start")
def api_agent_start():
    payload = request.get_json(force=True) or {}
    try:
        mode = str(payload.get("mode", "paper"))
        if mode in {"alpaca_live", "ibkr_live"} and not bool(payload.get("confirm_live", False)):
            raise ValueError("実取引を開始するには confirm_live=true を指定してください。")
        config = {
            "universe": str(payload.get("universe", "米国主要")),
            "sector": str(payload.get("sector", "すべて")),
            "top_n": int(payload.get("top_n", 5)),
            "max_positions": int(payload.get("max_positions", 5)),
            "buy_threshold_pct": float(payload.get("buy_threshold_pct", 1.0)),
            "stop_loss_pct": float(payload.get("stop_loss_pct", 5.0)),
            "position_size_pct": float(payload.get("position_size_pct", 15.0)),
            "interval_sec": int(payload.get("interval_sec", 900)),
            "rf_feature_flags": payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS),
        }
        data = auto_agent.start(mode=mode, config=config)
        return jsonify({"ok": True, "data": data})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/agent/stop")
def api_agent_stop():
    try:
        return jsonify({"ok": True, "data": auto_agent.stop()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/agent/step")
def api_agent_step():
    payload = request.get_json(force=True) or {}
    try:
        result = auto_agent.step_once(
            rf_feature_flags=payload.get("rf_feature_flags", DEFAULT_RF_FEATURE_FLAGS)
        )
        return jsonify({"ok": True, "data": result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "8000"))
    app.run(debug=True, host=host, port=port)
