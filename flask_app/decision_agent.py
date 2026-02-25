from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

try:
    from .forecast_service import ForecastPayload, forecast
except ImportError:
    from forecast_service import ForecastPayload, forecast


@dataclass
class DecisionResult:
    action: str
    should_buy: bool
    confidence: float
    expected_return_pct: float
    backtest_mae_pct: float
    backtest_rmse_pct: float
    trend_slope_pct: float
    reasons: list[str]
    symbol: str
    company_name: str
    model: str
    engine: str = "rule"

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return (numerator / denominator) * 100.0


def _extract_core_metrics(payload: ForecastPayload) -> dict:
    current_price = float(payload.history_values[-1])
    future_price = float(payload.forecast_values[-1])
    expected_return_pct = _safe_pct(future_price - current_price, current_price)
    backtest_mae_pct = _safe_pct(float(payload.backtest_mae), current_price)
    backtest_rmse_pct = _safe_pct(float(payload.backtest_rmse), current_price)
    forecast_arr = np.array(payload.forecast_values, dtype=float)
    if len(forecast_arr) >= 2 and abs(current_price) > 1e-9:
        x = np.arange(len(forecast_arr), dtype=float)
        slope = float(np.polyfit(x, forecast_arr, 1)[0])
        trend_slope_pct = _safe_pct(slope, current_price)
    else:
        trend_slope_pct = 0.0
    return {
        "current_price": current_price,
        "future_price": future_price,
        "expected_return_pct": expected_return_pct,
        "backtest_mae_pct": backtest_mae_pct,
        "backtest_rmse_pct": backtest_rmse_pct,
        "trend_slope_pct": trend_slope_pct,
    }


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    print(api_key)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("openai パッケージが利用できません。`uv sync` を実行してください。") from exc
    return OpenAI(api_key=api_key)


def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("OpenAI 応答を JSON として解釈できませんでした。")


def _decide_with_openai(
    payload: ForecastPayload,
    min_expected_return_pct: float,
    max_rmse_pct: float,
    max_mae_pct: float,
    openai_model: str,
) -> DecisionResult:
    metrics = _extract_core_metrics(payload)
    client = _get_openai_client()
    system_prompt = (
        "You are a conservative trading decision agent. Return ONLY JSON with keys: "
        "action (BUY/HOLD/SELL), confidence (0-100 number), reasons (array of short Japanese strings). "
        "Use risk control. Prefer HOLD when uncertain."
    )
    user_payload = {
        "symbol": payload.symbol,
        "company_name": payload.company_name,
        "model": payload.model,
        "thresholds": {
            "min_expected_return_pct": min_expected_return_pct,
            "max_rmse_pct": max_rmse_pct,
            "max_mae_pct": max_mae_pct,
        },
        "metrics": {
            "expected_return_pct": round(float(metrics["expected_return_pct"]), 4),
            "backtest_mae_pct": round(float(metrics["backtest_mae_pct"]), 4),
            "backtest_rmse_pct": round(float(metrics["backtest_rmse_pct"]), 4),
            "trend_slope_pct": round(float(metrics["trend_slope_pct"]), 6),
            "current_price": round(float(metrics["current_price"]), 6),
            "future_price": round(float(metrics["future_price"]), 6),
        },
    }

    raw_text = ""
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        raw_text = getattr(resp, "output_text", "") or ""
    else:
        comp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
        raw_text = comp.choices[0].message.content or ""

    parsed = _parse_json_from_text(raw_text)
    action = str(parsed.get("action", "HOLD")).upper().strip()
    if action not in {"BUY", "HOLD", "SELL"}:
        action = "HOLD"
    confidence = float(parsed.get("confidence", 50.0))
    reasons = parsed.get("reasons", [])
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    reasons = [str(r) for r in reasons][:5]

    return DecisionResult(
        action=action,
        should_buy=(action == "BUY"),
        confidence=round(max(1.0, min(99.0, confidence)), 1),
        expected_return_pct=round(float(metrics["expected_return_pct"]), 3),
        backtest_mae_pct=round(float(metrics["backtest_mae_pct"]), 3),
        backtest_rmse_pct=round(float(metrics["backtest_rmse_pct"]), 3),
        trend_slope_pct=round(float(metrics["trend_slope_pct"]), 4),
        reasons=reasons if reasons else ["OpenAI が十分な根拠を返さなかったため保守的に判断"],
        symbol=payload.symbol,
        company_name=payload.company_name,
        model=payload.model,
        engine=f"openai:{openai_model}",
    )


def decide_from_forecast(
    payload: ForecastPayload,
    min_expected_return_pct: float = 2.0,
    max_rmse_pct: float = 5.0,
    max_mae_pct: float = 3.0,
) -> DecisionResult:
    metrics = _extract_core_metrics(payload)
    current_price = float(metrics["current_price"])
    expected_return_pct = float(metrics["expected_return_pct"])
    backtest_mae_pct = float(metrics["backtest_mae_pct"])
    backtest_rmse_pct = float(metrics["backtest_rmse_pct"])
    trend_slope_pct = float(metrics["trend_slope_pct"])

    reasons: list[str] = []
    quality_good = backtest_rmse_pct <= max_rmse_pct and backtest_mae_pct <= max_mae_pct
    trend_up = trend_slope_pct > 0
    trend_down = trend_slope_pct < 0

    if expected_return_pct >= min_expected_return_pct and quality_good and trend_up:
        action = "BUY"
        should_buy = True
        reasons.append(f"期待リターン {expected_return_pct:.2f}% が閾値 {min_expected_return_pct:.2f}% 以上")
        reasons.append(f"予測誤差 RMSE={backtest_rmse_pct:.2f}% / MAE={backtest_mae_pct:.2f}% が許容範囲")
        reasons.append(f"予測トレンドが上向き（傾き {trend_slope_pct:.3f}%/step）")
    elif expected_return_pct <= -max(1.0, min_expected_return_pct * 0.7) and trend_down:
        action = "SELL"
        should_buy = False
        reasons.append(f"期待リターン {expected_return_pct:.2f}% がマイナス")
        reasons.append(f"予測トレンドが下向き（傾き {trend_slope_pct:.3f}%/step）")
    else:
        action = "HOLD"
        should_buy = False
        if expected_return_pct < min_expected_return_pct:
            reasons.append(
                f"期待リターン {expected_return_pct:.2f}% が閾値 {min_expected_return_pct:.2f}% 未満"
            )
        if not quality_good:
            reasons.append(
                f"予測誤差が大きい（RMSE={backtest_rmse_pct:.2f}% / MAE={backtest_mae_pct:.2f}%）"
            )
        if abs(trend_slope_pct) < 1e-4:
            reasons.append("予測トレンドが横ばい")
        elif trend_down:
            reasons.append(f"予測トレンドが下向き（傾き {trend_slope_pct:.3f}%/step）")

    # 信頼度: 0-100
    score = 50.0
    score += max(-30.0, min(30.0, expected_return_pct * 2.0))
    score += max(-20.0, min(20.0, -backtest_rmse_pct * 2.5 + 10.0))
    score += max(-15.0, min(15.0, trend_slope_pct * 300.0))
    if action == "BUY":
        score += 10.0
    if action == "SELL":
        score -= 5.0
    confidence = max(1.0, min(99.0, score))

    return DecisionResult(
        action=action,
        should_buy=should_buy,
        confidence=round(confidence, 1),
        expected_return_pct=round(expected_return_pct, 3),
        backtest_mae_pct=round(backtest_mae_pct, 3),
        backtest_rmse_pct=round(backtest_rmse_pct, 3),
        trend_slope_pct=round(trend_slope_pct, 4),
        reasons=reasons[:5],
        symbol=payload.symbol,
        company_name=payload.company_name,
        model=payload.model,
        engine="rule",
    )


def decide_trade(
    raw_symbol: str,
    period: str = "2y",
    forecast_days: int = 30,
    lag: int = 20,
    model: str = "Ensemble",
    daily: bool = False,
    weekly: bool = True,
    yearly: bool = True,
    changepoint_prior_scale: float = 0.05,
    seasonality_mode: str = "additive",
    rf_feature_flags: dict | None = None,
    auto_optimize: bool = False,
    optimization_trials: int = 30,
    fast_optimize: bool = True,
    min_expected_return_pct: float = 2.0,
    max_rmse_pct: float = 5.0,
    max_mae_pct: float = 3.0,
    use_openai: bool = False,
    openai_model: str = "gpt-4.1-mini",
) -> dict:
    payload = forecast(
        raw_symbol=raw_symbol,
        period=period,
        forecast_days=forecast_days,
        lag=lag,
        model=model,
        daily=daily,
        weekly=weekly,
        yearly=yearly,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
        rf_feature_flags=rf_feature_flags,
        auto_optimize=auto_optimize,
        optimization_trials=optimization_trials,
        fast_optimize=fast_optimize,
    )
    openai_error = ""
    if use_openai:
        try:
            print('openai')
            decision = _decide_with_openai(
                payload=payload,
                min_expected_return_pct=min_expected_return_pct,
                max_rmse_pct=max_rmse_pct,
                max_mae_pct=max_mae_pct,
                openai_model=openai_model,
            )
        except Exception as exc:
            openai_error = str(exc)
            decision = decide_from_forecast(
                payload=payload,
                min_expected_return_pct=min_expected_return_pct,
                max_rmse_pct=max_rmse_pct,
                max_mae_pct=max_mae_pct,
            )
            decision.engine = "rule_fallback"
            decision.reasons.insert(0, f"OpenAI判定に失敗したためルールベースへフォールバック: {openai_error}")
    else:
        decision = decide_from_forecast(
            payload=payload,
            min_expected_return_pct=min_expected_return_pct,
            max_rmse_pct=max_rmse_pct,
            max_mae_pct=max_mae_pct,
        )
    return {
        "decision": decision.as_dict(),
        "forecast": payload.__dict__,
        "openai_error": openai_error,
    }
