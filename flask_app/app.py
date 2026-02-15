from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.domain import DEFAULT_RF_FEATURE_FLAGS, SCREEN_UNIVERSES
from forecast_service import forecast, screen_candidates

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        universes=list(SCREEN_UNIVERSES.keys()),
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
        )
        return jsonify({"ok": True, "data": result.__dict__})
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
        )
        return jsonify({"ok": True, "rows": rows})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
