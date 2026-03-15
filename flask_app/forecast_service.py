from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from shared.domain import ALIAS_SYMBOLS, DEFAULT_RF_FEATURE_FLAGS, SCREEN_UNIVERSES, sector_of
from shared.feature_utils import (
    build_rf_feature_dataset,
    make_rf_feature_vector,
    normalize_rf_feature_flags,
)
try:
    from .universe_store import get_universe_items
except ImportError:
    from universe_store import get_universe_items


def get_optuna():
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise RuntimeError("optuna が利用できません。`uv sync` を実行してください。") from exc
    return optuna


def get_yf():
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "yfinance の読み込みに失敗しました。Python の SSL ライブラリが壊れている可能性があります。"
            " `uv python install 3.11 && rm -rf .venv && uv venv --python 3.11 && uv sync` を実行してください。"
        ) from exc
    return yf


OPTIMIZATION_CACHE_PATH = Path(__file__).resolve().parent / "optimization_cache.json"


def _load_optimization_cache() -> dict[str, dict]:
    if not OPTIMIZATION_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(OPTIMIZATION_CACHE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_optimization_cache(cache: dict[str, dict]) -> None:
    OPTIMIZATION_CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _series_signature(close: pd.Series) -> str:
    if close.empty:
        return "empty"
    last_date = str(close.index[-1].date()) if hasattr(close.index[-1], "date") else str(close.index[-1])
    values = close.values.astype(float)
    sample = values[-min(40, len(values)) :]
    material = f"{len(values)}|{last_date}|{float(values[-1]):.8f}|{float(np.mean(sample)):.8f}|{float(np.std(sample)):.8f}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _build_opt_cache_key(
    symbol: str,
    model: str,
    lag: int,
    daily: bool,
    weekly: bool,
    yearly: bool,
    rf_feature_flags: dict,
    n_trials: int,
    fast_mode: bool,
    close: pd.Series,
) -> str:
    flags = normalize_rf_feature_flags(rf_feature_flags)
    key_material = {
        "symbol": symbol,
        "model": model,
        "lag": int(lag),
        "daily": bool(daily),
        "weekly": bool(weekly),
        "yearly": bool(yearly),
        "flags": flags,
        "n_trials": int(n_trials),
        "fast_mode": bool(fast_mode),
        "series_sig": _series_signature(close),
    }
    return hashlib.sha256(json.dumps(key_material, sort_keys=True).encode("utf-8")).hexdigest()


def _get_cached_optimization(key: str) -> dict | None:
    cache = _load_optimization_cache()
    row = cache.get(key)
    return row if isinstance(row, dict) else None


def _put_cached_optimization(key: str, best: dict) -> None:
    cache = _load_optimization_cache()
    cache[key] = {"best": best}
    if len(cache) > 500:
        cache = dict(list(cache.items())[-500:])
    _save_optimization_cache(cache)


@dataclass
class ForecastPayload:
    symbol: str
    company_name: str
    model: str
    history_dates: list[str]
    history_values: list[float]
    forecast_dates: list[str]
    forecast_values: list[float]
    backtest_dates: list[str]
    backtest_actual: list[float]
    backtest_pred: list[float]
    backtest_mae: float
    backtest_rmse: float
    optimization_note: str = ""
    optimized_params: dict[str, object] | None = None
    ensemble_weights: dict[str, float] | None = None


def extract_close_series(data: pd.DataFrame) -> pd.Series:
    close_data = None
    if isinstance(data.columns, pd.MultiIndex):
        matches = [c for c in data.columns if c[0] == "Close"]
        if matches:
            close_data = data[matches[0]]
    elif "Close" in data.columns:
        close_data = data["Close"]

    if close_data is None:
        raise ValueError("Close 列が存在しません。")

    if isinstance(close_data, pd.DataFrame):
        if close_data.shape[1] == 0:
            raise ValueError("Close データが空です。")
        close_data = close_data.iloc[:, 0]

    close_series = pd.Series(close_data).dropna().astype(float)
    if close_series.empty:
        raise ValueError("Close データが空です。")
    return close_series


def has_recent_data(symbol: str) -> bool:
    try:
        data = get_yf().download(symbol, period="1mo", progress=False, auto_adjust=True)
        return not data.empty
    except Exception:
        return False


def resolve_symbol(raw_symbol: str) -> str:
    text = unicodedata.normalize("NFKC", raw_symbol).strip()
    upper_text = text.upper()
    is_japanese = any(("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff") for ch in text)

    if text.isdigit():
        return f"{text}.T"
    if "." in text:
        return upper_text

    if "-" in upper_text and upper_text.endswith(("USD", "USDT", "JPY")):
        return upper_text
    if upper_text.isalpha() and 2 <= len(upper_text) <= 6:
        crypto_symbol = f"{upper_text}-USD"
        if has_recent_data(crypto_symbol):
            return crypto_symbol

    alias = ALIAS_SYMBOLS.get(upper_text) or ALIAS_SYMBOLS.get(text)
    if alias:
        return alias

    try:
            query_candidates = [text]
            if is_japanese:
                query_candidates.extend([f"{text} 株価", f"{text} 日本"])
            for query in query_candidates:
                search = get_yf().Search(query=query, max_results=20)
                quotes = search.quotes or []
            for quote in quotes[:10]:
                symbol = str(quote.get("symbol", "")).upper()
                if symbol and has_recent_data(symbol):
                    return symbol
    except Exception:
        pass

    if upper_text.isalnum() and 1 <= len(upper_text) <= 6 and has_recent_data(upper_text):
        return upper_text

    raise ValueError("銘柄を解決できませんでした。")


def get_company_name(symbol: str, fallback: str) -> str:
    try:
        ticker = get_yf().Ticker(symbol)
        info = ticker.info or {}
        name = info.get("longName") or info.get("shortName")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return fallback.strip() or symbol


def train_rf(
    series: pd.Series,
    forecast_days: int,
    lag: int,
    rf_feature_flags: dict,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
) -> pd.Series:
    values = series.values.astype(float)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    x, y = build_rf_feature_dataset(values, lag, flags)
    if len(x) == 0:
        raise ValueError("RandomForest 学習に必要なデータが不足しています。")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        n_jobs=-1,
    )
    model.fit(x, y)

    history = list(values)
    preds = []
    for _ in range(forecast_days):
        fv = np.array(make_rf_feature_vector(np.array(history[-lag:], dtype=float), flags)).reshape(1, -1)
        r = float(model.predict(fv)[0])
        v = history[-1] * (1.0 + r)
        preds.append(v)
        history.append(v)

    idx = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.Series(preds, index=idx, name="Forecast")


def train_xgb(
    series: pd.Series,
    forecast_days: int,
    lag: int,
    rf_feature_flags: dict,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
) -> pd.Series:
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ValueError("XGBoost が利用できません。") from exc

    values = series.values.astype(float)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    x, y = build_rf_feature_dataset(values, lag, flags)
    if len(x) == 0:
        raise ValueError("XGBoost 学習に必要なデータが不足しています。")

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(x, y)

    history = list(values)
    preds = []
    for _ in range(forecast_days):
        fv = np.array(make_rf_feature_vector(np.array(history[-lag:], dtype=float), flags)).reshape(1, -1)
        r = float(model.predict(fv)[0])
        v = history[-1] * (1.0 + r)
        preds.append(v)
        history.append(v)

    idx = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    return pd.Series(preds, index=idx, name="Forecast")


def train_prophet(
    series: pd.Series,
    forecast_days: int,
    daily: bool,
    weekly: bool,
    yearly: bool,
    changepoint_prior_scale: float,
    seasonality_mode: str = "additive",
) -> pd.Series:
    try:
        from prophet import Prophet
    except Exception as exc:
        raise ValueError("Prophet が利用できません。") from exc

    if not any([daily, weekly, yearly]):
        raise ValueError("Prophet の季節性を1つ以上選択してください。")

    train_df = pd.DataFrame({"ds": pd.to_datetime(series.index), "y": series.values.astype(float)})
    model = Prophet(
        daily_seasonality=daily,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode,
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    pred = model.predict(future)
    out = pred[["ds", "yhat"]].tail(forecast_days).copy()
    out["ds"] = pd.to_datetime(out["ds"])
    return out.set_index("ds")["yhat"].rename("Forecast")


def weighted_average_three(
    rf_pred: pd.Series,
    prophet_pred: pd.Series,
    xgb_pred: pd.Series,
    w_rf: float,
    w_prophet: float,
    w_xgb: float,
) -> pd.Series:
    aligned = pd.concat(
        [rf_pred.rename("rf"), prophet_pred.rename("prophet"), xgb_pred.rename("xgb")], axis=1
    ).dropna()
    total = max(w_rf + w_prophet + w_xgb, 1e-9)
    return (
        aligned["rf"] * (w_rf / total)
        + aligned["prophet"] * (w_prophet / total)
        + aligned["xgb"] * (w_xgb / total)
    ).rename("Forecast")


def rolling_mae_for_model(
    close: pd.Series,
    model: str,
    lag: int,
    rf_feature_flags: dict,
    daily: bool,
    weekly: bool,
    yearly: bool,
    cps: float,
    seasonality_mode: str,
    xgb_params: dict,
    folds: int = 3,
    window_days: int = 20,
) -> float:
    if len(close) < 200:
        return float("nan")
    maes = []
    n = len(close)
    for i in range(folds, 0, -1):
        test_end = n - (i - 1) * window_days
        test_start = test_end - window_days
        if test_start <= max(80, lag + 20):
            continue
        train = close.iloc[:test_start]
        test = close.iloc[test_start:test_end]
        if model == "rf":
            pred = train_rf(train, len(test), lag, rf_feature_flags)
        elif model == "prophet":
            pred = train_prophet(train, len(test), daily, weekly, yearly, cps, seasonality_mode)
        else:
            pred = train_xgb(
                train,
                len(test),
                lag,
                rf_feature_flags,
                n_estimators=int(xgb_params["n_estimators"]),
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
            )
        pred = pred.reindex(test.index).dropna()
        actual = test.reindex(pred.index).dropna()
        if len(actual) < 5:
            continue
        maes.append(float(np.mean(np.abs(actual.values - pred.reindex(actual.index).values))))
    return float(np.mean(maes)) if maes else float("nan")


def compute_ensemble_weights(
    close: pd.Series,
    lag: int,
    rf_feature_flags: dict,
    daily: bool,
    weekly: bool,
    yearly: bool,
    cps: float,
    seasonality_mode: str,
    xgb_params: dict,
) -> dict:
    rf_mae = rolling_mae_for_model(
        close, "rf", lag, rf_feature_flags, daily, weekly, yearly, cps, seasonality_mode, xgb_params
    )
    prophet_mae = rolling_mae_for_model(
        close, "prophet", lag, rf_feature_flags, daily, weekly, yearly, cps, seasonality_mode, xgb_params
    )
    xgb_mae = rolling_mae_for_model(
        close, "xgb", lag, rf_feature_flags, daily, weekly, yearly, cps, seasonality_mode, xgb_params
    )
    inv_rf = 1.0 / max(rf_mae, 1e-9)
    inv_prophet = 1.0 / max(prophet_mae, 1e-9)
    inv_xgb = 1.0 / max(xgb_mae, 1e-9)
    total = inv_rf + inv_prophet + inv_xgb
    return {"rf": inv_rf / total, "prophet": inv_prophet / total, "xgb": inv_xgb / total}


def optimize_rf_hyperparameters(
    close: pd.Series,
    base_lag: int,
    rf_feature_flags: dict,
    n_trials: int = 30,
    fast_mode: bool = True,
) -> dict:
    optuna = get_optuna()
    series = close.tail(260) if fast_mode and len(close) > 260 else close
    values = series.values.astype(float)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    min_lag = max(5, base_lag - 12)
    max_lag = min(120, base_lag + 12)
    if len(values) < max(80, min_lag + 35):
        raise ValueError("最適化に必要なデータが不足しています。期間を長くしてください。")

    def objective(trial) -> float:
        lag = int(trial.suggest_int("lag", min_lag, max_lag))
        n_estimators = int(trial.suggest_int("n_estimators", 120, 600, step=40))
        max_depth_choice = trial.suggest_categorical("max_depth_choice", ["none", "8", "12", "16", "24"])
        max_depth = None if max_depth_choice == "none" else int(max_depth_choice)
        min_samples_leaf = int(trial.suggest_int("min_samples_leaf", 1, 6))

        x, y = build_rf_feature_dataset(values, lag, flags)
        if len(x) < 50:
            return float("inf")
        tscv = TimeSeriesSplit(n_splits=2 if fast_mode else 3)
        fold_maes = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(x), start=1):
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
            )
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[val_idx])
            mae = float(np.mean(np.abs(y[val_idx] - pred)))
            fold_maes.append(mae)
            trial.report(float(np.mean(fold_maes)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes)) if fold_maes else float("inf")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, int(n_trials) // 5), n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=max(5, int(n_trials)), show_progress_bar=False)

    if study.best_trial is None or not np.isfinite(study.best_value):
        raise ValueError("RandomForest の最適化に失敗しました。")

    best = study.best_trial.params
    best_depth = None if best["max_depth_choice"] == "none" else int(best["max_depth_choice"])
    return {
        "lag": int(best["lag"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": best_depth,
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "cv_mae": round(float(study.best_value), 4),
    }


def optimize_xgb_hyperparameters(
    close: pd.Series,
    base_lag: int,
    rf_feature_flags: dict,
    n_trials: int = 35,
    fast_mode: bool = True,
) -> dict:
    optuna = get_optuna()
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise ValueError("XGBoost が利用できません。") from exc

    series = close.tail(260) if fast_mode and len(close) > 260 else close
    values = series.values.astype(float)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    min_lag = max(5, base_lag - 12)
    max_lag = min(120, base_lag + 12)
    if len(values) < max(80, min_lag + 35):
        raise ValueError("最適化に必要なデータが不足しています。期間を長くしてください。")

    def objective(trial) -> float:
        lag = int(trial.suggest_int("lag", min_lag, max_lag))
        n_estimators = int(trial.suggest_int("n_estimators", 160, 700, step=40))
        max_depth = int(trial.suggest_int("max_depth", 3, 10))
        learning_rate = float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True))
        subsample = float(trial.suggest_float("subsample", 0.7, 1.0))
        colsample_bytree = float(trial.suggest_float("colsample_bytree", 0.7, 1.0))
        reg_lambda = float(trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))

        x, y = build_rf_feature_dataset(values, lag, flags)
        if len(x) < 50:
            return float("inf")
        tscv = TimeSeriesSplit(n_splits=2 if fast_mode else 3)
        fold_maes = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(x), start=1):
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_lambda=reg_lambda,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=4,
            )
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[val_idx])
            mae = float(np.mean(np.abs(y[val_idx] - pred)))
            fold_maes.append(mae)
            trial.report(float(np.mean(fold_maes)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes)) if fold_maes else float("inf")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, int(n_trials) // 5), n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=max(5, int(n_trials)), show_progress_bar=False)

    if study.best_trial is None or not np.isfinite(study.best_value):
        raise ValueError("XGBoost の最適化に失敗しました。")

    best = study.best_trial.params
    return {
        "lag": int(best["lag"]),
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "cv_mae": round(float(study.best_value), 4),
    }


def optimize_prophet_hyperparameters(
    close: pd.Series,
    daily: bool,
    weekly: bool,
    yearly: bool,
    n_trials: int = 20,
    fast_mode: bool = True,
) -> dict:
    series = close.tail(320) if fast_mode and len(close) > 320 else close
    if len(series) < 160:
        raise ValueError("Prophet 最適化にはより長い期間のデータが必要です（2y 以上推奨）。")
    if not any([daily, weekly, yearly]):
        raise ValueError("Prophet の季節性を1つ以上選択してください。")
    optuna = get_optuna()
    values = series.values.astype(float)

    def objective(trial) -> float:
        cps = float(trial.suggest_float("changepoint_prior_scale", 0.005, 0.5, log=True))
        mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        tscv = TimeSeriesSplit(n_splits=2 if fast_mode else 3)
        fold_maes = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(values), start=1):
            train_series = series.iloc[train_idx]
            val_series = series.iloc[val_idx]
            pred = train_prophet(
                train_series,
                forecast_days=len(val_series),
                daily=daily,
                weekly=weekly,
                yearly=yearly,
                changepoint_prior_scale=cps,
                seasonality_mode=mode,
            )
            pred = pred.reindex(val_series.index).dropna()
            actual = val_series.reindex(pred.index).dropna()
            if len(actual) < 5:
                continue
            mae = float(np.mean(np.abs(actual.values - pred.reindex(actual.index).values)))
            fold_maes.append(mae)
            trial.report(float(np.mean(fold_maes)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_maes)) if fold_maes else float("inf")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, int(n_trials) // 5), n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=max(5, int(n_trials)), show_progress_bar=False)

    if study.best_trial is None or not np.isfinite(study.best_value):
        raise ValueError("Prophet の最適化に失敗しました。")

    best = study.best_trial.params
    return {
        "changepoint_prior_scale": float(best["changepoint_prior_scale"]),
        "seasonality_mode": str(best["seasonality_mode"]),
        "cv_mae": round(float(study.best_value), 4),
    }


def optimize_ensemble_hyperparameters(
    close: pd.Series,
    base_lag: int,
    daily: bool,
    weekly: bool,
    yearly: bool,
    rf_feature_flags: dict,
    n_trials: int = 24,
    fast_mode: bool = True,
) -> dict:
    per_model_trials = max(5, int(n_trials))
    with ThreadPoolExecutor(max_workers=3) as executor:
        rf_fut = executor.submit(
            optimize_rf_hyperparameters,
            close,
            base_lag,
            rf_feature_flags,
            per_model_trials,
            fast_mode,
        )
        xgb_fut = executor.submit(
            optimize_xgb_hyperparameters,
            close,
            base_lag,
            rf_feature_flags,
            per_model_trials,
            fast_mode,
        )
        prophet_fut = executor.submit(
            optimize_prophet_hyperparameters,
            close,
            daily,
            weekly,
            yearly,
            per_model_trials,
            fast_mode,
        )
        rf_best = rf_fut.result()
        xgb_best = xgb_fut.result()
        prophet_best = prophet_fut.result()
    return {
        "lag": rf_best["lag"],
        "n_estimators": rf_best["n_estimators"],
        "max_depth": rf_best["max_depth"],
        "min_samples_leaf": rf_best["min_samples_leaf"],
        "xgb_n_estimators": xgb_best["n_estimators"],
        "xgb_max_depth": xgb_best["max_depth"],
        "xgb_learning_rate": xgb_best["learning_rate"],
        "xgb_subsample": xgb_best["subsample"],
        "xgb_colsample_bytree": xgb_best["colsample_bytree"],
        "changepoint_prior_scale": prophet_best["changepoint_prior_scale"],
        "seasonality_mode": prophet_best["seasonality_mode"],
        "rf_cv_mae": rf_best["cv_mae"],
        "xgb_cv_mae": xgb_best["cv_mae"],
        "prophet_cv_mae": prophet_best["cv_mae"],
    }


def optimize_hyperparameters(
    close: pd.Series,
    model: str,
    lag: int,
    daily: bool,
    weekly: bool,
    yearly: bool,
    rf_feature_flags: dict,
    n_trials: int = 30,
    fast_mode: bool = True,
) -> dict:
    flags = normalize_rf_feature_flags(rf_feature_flags)
    if model == "RandomForest":
        return optimize_rf_hyperparameters(close, lag, flags, n_trials=n_trials, fast_mode=fast_mode)
    if model == "XGBoost":
        return optimize_xgb_hyperparameters(close, lag, flags, n_trials=n_trials, fast_mode=fast_mode)
    if model == "Ensemble":
        return optimize_ensemble_hyperparameters(
            close,
            lag,
            daily,
            weekly,
            yearly,
            flags,
            n_trials=n_trials,
            fast_mode=fast_mode,
        )
    return optimize_prophet_hyperparameters(close, daily, weekly, yearly, n_trials=n_trials, fast_mode=fast_mode)


def optimize_for_symbol(
    raw_symbol: str,
    period: str,
    model: str,
    lag: int,
    daily: bool,
    weekly: bool,
    yearly: bool,
    rf_feature_flags: dict | None = None,
    n_trials: int = 30,
    fast_mode: bool = True,
) -> dict:
    symbol = resolve_symbol(raw_symbol)
    data = get_yf().download(symbol, period=period, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError("株価データを取得できませんでした。")
    close = extract_close_series(data)
    trials = max(5, int(n_trials))
    flags = normalize_rf_feature_flags(rf_feature_flags or DEFAULT_RF_FEATURE_FLAGS)
    cache_key = _build_opt_cache_key(
        symbol=symbol,
        model=model,
        lag=lag,
        daily=daily,
        weekly=weekly,
        yearly=yearly,
        rf_feature_flags=flags,
        n_trials=trials,
        fast_mode=fast_mode,
        close=close,
    )
    cached = _get_cached_optimization(cache_key)
    if cached and isinstance(cached.get("best"), dict):
        return {
            "symbol": symbol,
            "company_name": get_company_name(symbol, raw_symbol),
            "model": model,
            "best": cached["best"],
            "n_trials": trials,
            "cache_hit": True,
            "fast_mode": bool(fast_mode),
        }

    best = optimize_hyperparameters(
        close=close,
        model=model,
        lag=lag,
        daily=daily,
        weekly=weekly,
        yearly=yearly,
        rf_feature_flags=flags,
        n_trials=trials,
        fast_mode=fast_mode,
    )
    _put_cached_optimization(cache_key, best)
    return {
        "symbol": symbol,
        "company_name": get_company_name(symbol, raw_symbol),
        "model": model,
        "best": best,
        "n_trials": trials,
        "cache_hit": False,
        "fast_mode": bool(fast_mode),
    }


def run_backtest(
    close: pd.Series,
    model: str,
    lag: int,
    rf_feature_flags: dict,
    daily: bool,
    weekly: bool,
    yearly: bool,
    cps: float,
    seasonality_mode: str,
    rf_params: dict,
    xgb_params: dict,
    ensemble_weights: dict,
) -> tuple[pd.Series, pd.Series]:
    backtest_start = close.index.max() - pd.DateOffset(months=3)
    train = close[close.index < backtest_start]
    test = close[close.index >= backtest_start]
    if len(test) < 20 or len(train) < 60:
        raise ValueError("バックテストに必要なデータが不足しています。")

    horizon = len(test)
    if model == "RandomForest":
        pred = train_rf(
            train,
            horizon,
            lag,
            rf_feature_flags,
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
        )
    elif model == "Prophet":
        pred = train_prophet(train, horizon, daily, weekly, yearly, cps, seasonality_mode)
    elif model == "XGBoost":
        pred = train_xgb(
            train,
            horizon,
            lag,
            rf_feature_flags,
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
        )
    else:
        rf_pred = train_rf(
            train,
            horizon,
            lag,
            rf_feature_flags,
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
        )
        prophet_pred = train_prophet(train, horizon, daily, weekly, yearly, cps, seasonality_mode)
        xgb_pred = train_xgb(
            train,
            horizon,
            lag,
            rf_feature_flags,
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
        )
        pred = weighted_average_three(
            rf_pred,
            prophet_pred,
            xgb_pred,
            ensemble_weights["rf"],
            ensemble_weights["prophet"],
            ensemble_weights["xgb"],
        )

    pred = pred.reindex(test.index).dropna()
    actual = test.reindex(pred.index).dropna()
    pred = pred.reindex(actual.index)
    return actual, pred


def forecast(
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
) -> ForecastPayload:
    symbol = resolve_symbol(raw_symbol)
    data = get_yf().download(symbol, period=period, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError("株価データを取得できませんでした。")

    close = extract_close_series(data)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    optimization_trials = max(5, int(optimization_trials))
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }
    optimization_note = ""
    optimized_params: dict[str, object] | None = None

    if auto_optimize:
        cache_key = _build_opt_cache_key(
            symbol=symbol,
            model=model,
            lag=lag,
            daily=daily,
            weekly=weekly,
            yearly=yearly,
            rf_feature_flags=flags,
            n_trials=optimization_trials,
            fast_mode=fast_optimize,
            close=close,
        )
        cached = _get_cached_optimization(cache_key)
        if cached and isinstance(cached.get("best"), dict):
            best = cached["best"]
            cache_hit = True
        else:
            best = optimize_hyperparameters(
                close=close,
                model=model,
                lag=lag,
                daily=daily,
                weekly=weekly,
                yearly=yearly,
                rf_feature_flags=flags,
                n_trials=optimization_trials,
                fast_mode=fast_optimize,
            )
            _put_cached_optimization(cache_key, best)
            cache_hit = False
        optimized_params = best
        if model == "RandomForest":
            lag = int(best["lag"])
            rf_params = {
                "n_estimators": int(best["n_estimators"]),
                "max_depth": best["max_depth"],
                "min_samples_leaf": int(best["min_samples_leaf"]),
            }
            optimization_note = (
                f"最適化: lag={best['lag']}, n={best['n_estimators']}, "
                f"depth={best['max_depth']}, leaf={best['min_samples_leaf']}, cv_mae={best['cv_mae']}"
            )
        elif model == "XGBoost":
            lag = int(best["lag"])
            xgb_params = {
                "n_estimators": int(best["n_estimators"]),
                "max_depth": int(best["max_depth"]),
                "learning_rate": float(best["learning_rate"]),
                "subsample": float(best["subsample"]),
                "colsample_bytree": float(best["colsample_bytree"]),
            }
            optimization_note = (
                f"最適化: lag={best['lag']}, n={best['n_estimators']}, depth={best['max_depth']}, "
                f"lr={best['learning_rate']}, subsample={best['subsample']}, colsample={best['colsample_bytree']}, cv_mae={best['cv_mae']}"
            )
        elif model == "Prophet":
            changepoint_prior_scale = float(best["changepoint_prior_scale"])
            seasonality_mode = str(best["seasonality_mode"])
            optimization_note = (
                f"最適化: changepoint={best['changepoint_prior_scale']}, mode={best['seasonality_mode']}, cv_mae={best['cv_mae']}"
            )
        else:
            lag = int(best["lag"])
            rf_params = {
                "n_estimators": int(best["n_estimators"]),
                "max_depth": best["max_depth"],
                "min_samples_leaf": int(best["min_samples_leaf"]),
            }
            xgb_params = {
                "n_estimators": int(best["xgb_n_estimators"]),
                "max_depth": int(best["xgb_max_depth"]),
                "learning_rate": float(best["xgb_learning_rate"]),
                "subsample": float(best["xgb_subsample"]),
                "colsample_bytree": float(best["xgb_colsample_bytree"]),
            }
            changepoint_prior_scale = float(best["changepoint_prior_scale"])
            seasonality_mode = str(best["seasonality_mode"])
            optimization_note = (
                "最適化: "
                f"RF(lag={best['lag']}, n={best['n_estimators']}, depth={best['max_depth']}, leaf={best['min_samples_leaf']}) "
                f"XGB(n={best['xgb_n_estimators']}, depth={best['xgb_max_depth']}, lr={best['xgb_learning_rate']}) "
                f"Prophet(cps={best['changepoint_prior_scale']}, mode={best['seasonality_mode']})"
            )
        optimization_note = f"{optimization_note} [trials={optimization_trials}]".strip()
        optimization_note = f"{optimization_note} [{'cache' if cache_hit else 'fresh'}]"
    else:
        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
        }

    if auto_optimize and model in ["Prophet", "XGBoost"]:
        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
        }

    if model == "RandomForest":
        forecast_series = train_rf(
            close,
            forecast_days,
            lag,
            flags,
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
        )
        weights = {"rf": 1.0, "prophet": 0.0, "xgb": 0.0}
    elif model == "Prophet":
        forecast_series = train_prophet(
            close, forecast_days, daily, weekly, yearly, changepoint_prior_scale, seasonality_mode
        )
        weights = {"rf": 0.0, "prophet": 1.0, "xgb": 0.0}
    elif model == "XGBoost":
        forecast_series = train_xgb(
            close,
            forecast_days,
            lag,
            flags,
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
        )
        weights = {"rf": 0.0, "prophet": 0.0, "xgb": 1.0}
    else:
        weights = compute_ensemble_weights(
            close,
            lag,
            flags,
            daily,
            weekly,
            yearly,
            changepoint_prior_scale,
            seasonality_mode,
            xgb_params,
        )
        rf_pred = train_rf(
            close,
            forecast_days,
            lag,
            flags,
            n_estimators=int(rf_params["n_estimators"]),
            max_depth=rf_params["max_depth"],
            min_samples_leaf=int(rf_params["min_samples_leaf"]),
        )
        prophet_pred = train_prophet(
            close, forecast_days, daily, weekly, yearly, changepoint_prior_scale, seasonality_mode
        )
        xgb_pred = train_xgb(
            close,
            forecast_days,
            lag,
            flags,
            n_estimators=int(xgb_params["n_estimators"]),
            max_depth=int(xgb_params["max_depth"]),
            learning_rate=float(xgb_params["learning_rate"]),
            subsample=float(xgb_params["subsample"]),
            colsample_bytree=float(xgb_params["colsample_bytree"]),
        )
        forecast_series = weighted_average_three(
            rf_pred,
            prophet_pred,
            xgb_pred,
            weights["rf"],
            weights["prophet"],
            weights["xgb"],
        )

    back_actual, back_pred = run_backtest(
        close,
        model,
        lag,
        flags,
        daily,
        weekly,
        yearly,
        changepoint_prior_scale,
        seasonality_mode,
        rf_params,
        xgb_params,
        weights,
    )
    mae = float(np.mean(np.abs(back_actual.values - back_pred.values)))
    rmse = float(np.sqrt(np.mean((back_actual.values - back_pred.values) ** 2)))

    payload = ForecastPayload(
        symbol=symbol,
        company_name=get_company_name(symbol, raw_symbol),
        model=model,
        history_dates=[d.strftime("%Y-%m-%d") for d in close.index],
        history_values=[float(v) for v in close.values],
        forecast_dates=[d.strftime("%Y-%m-%d") for d in forecast_series.index],
        forecast_values=[float(v) for v in forecast_series.values],
        backtest_dates=[d.strftime("%Y-%m-%d") for d in back_actual.index],
        backtest_actual=[float(v) for v in back_actual.values],
        backtest_pred=[float(v) for v in back_pred.values],
        backtest_mae=mae,
        backtest_rmse=rmse,
        optimization_note=optimization_note,
        optimized_params=optimized_params,
        ensemble_weights=weights,
    )
    return payload


def screen_candidates(
    universe: str = "日本主要",
    top_n: int = 5,
    rf_feature_flags: dict | None = None,
    sector: str = "すべて",
) -> list[dict]:
    flags = normalize_rf_feature_flags(rf_feature_flags)
    items = get_universe_items(universe)
    sector = str(sector or "すべて")
    if sector != "すべて":
        items = [(s, n) for s, n in items if sector_of(s) == sector]
    scored = []

    for symbol, name in items:
        try:
            data = get_yf().download(symbol, period="1y", progress=False, auto_adjust=True)
            close = extract_close_series(data)
            if len(close) < 140:
                continue
            last_price = float(close.iloc[-1])
            pred = train_rf(close, forecast_days=20, lag=20, rf_feature_flags=flags)
            predicted_price = float(pred.iloc[-1])
            expected_return = (predicted_price / last_price) - 1.0
            momentum_60 = (last_price / float(close.iloc[-60])) - 1.0
            returns = np.log(close / close.shift(1)).dropna()
            volatility_60 = float(returns.iloc[-60:].std() * np.sqrt(252))
            score = (0.6 * expected_return) + (0.4 * momentum_60) - (0.2 * volatility_60)
            scored.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector_of(symbol),
                    "last_price": round(last_price, 2),
                    "predicted_price": round(predicted_price, 2),
                    "expected_return": round(expected_return * 100, 2),
                    "momentum_60": round(momentum_60 * 100, 2),
                    "volatility_60": round(volatility_60 * 100, 2),
                    "score": round(score, 4),
                }
            )
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]
