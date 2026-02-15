import numpy as np

from domain import DEFAULT_RF_FEATURE_FLAGS


def normalize_rf_feature_flags(flags: dict | None) -> dict:
    normalized = dict(DEFAULT_RF_FEATURE_FLAGS)
    if flags:
        for key in normalized.keys():
            if key in flags:
                normalized[key] = bool(flags[key])
    return normalized


def build_rf_feature_dataset(
    values: np.ndarray, lag: int, rf_feature_flags: dict | None
) -> tuple[np.ndarray, np.ndarray]:
    x = []
    y = []
    flags = normalize_rf_feature_flags(rf_feature_flags)
    for i in range(lag, len(values)):
        window = values[i - lag : i]
        x.append(make_rf_feature_vector(window, flags))
        y.append(values[i])
    return np.array(x, dtype=float), np.array(y, dtype=float)


def make_rf_feature_vector(window: np.ndarray, rf_feature_flags: dict) -> list[float]:
    eps = 1e-9
    w = window.astype(float)
    flags = normalize_rf_feature_flags(rf_feature_flags)
    last = w[-1]
    features = list(w)

    if flags["return"]:
        prev = w[-2] if len(w) > 1 else w[-1]
        ret_1 = (last / max(prev, eps)) - 1.0
        ref_5 = w[-6] if len(w) > 6 else w[0]
        ret_5 = (last / max(ref_5, eps)) - 1.0
        features.extend([ret_1, ret_5])

    if flags["ma_ratio"]:
        ma_5 = float(np.mean(w[-5:])) if len(w) >= 5 else float(np.mean(w))
        ma_20 = float(np.mean(w[-20:])) if len(w) >= 20 else float(np.mean(w))
        ma5_ratio = (last / max(ma_5, eps)) - 1.0
        ma20_ratio = (last / max(ma_20, eps)) - 1.0
        features.extend([ma5_ratio, ma20_ratio])

    if flags["momentum"]:
        momentum_20 = (last / max(w[-20] if len(w) >= 20 else w[0], eps)) - 1.0
        features.append(momentum_20)

    if flags["volatility"]:
        vol_20 = (
            float(np.std(np.diff(np.log(np.maximum(w[-21:], eps)))))
            if len(w) >= 21
            else float(np.std(np.diff(np.log(np.maximum(w, eps)))))
        )
        features.append(vol_20)

    if flags["rsi"]:
        features.append(calc_rsi_from_window(w, period=14))

    if flags["macd"]:
        macd, macd_signal = calc_macd_from_window(w, fast=12, slow=26, signal=9)
        features.extend([macd, macd_signal])

    return features


def calc_rsi_from_window(window: np.ndarray, period: int = 14) -> float:
    if len(window) < 3:
        return 50.0
    diffs = np.diff(window[-(period + 1) :]) if len(window) > period else np.diff(window)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = float(np.mean(gains)) if len(gains) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ema_series(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(values, dtype=float)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def calc_macd_from_window(
    window: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[float, float]:
    w = window.astype(float)
    if len(w) < 3:
        return 0.0, 0.0
    fast_span = min(fast, max(2, len(w) - 1))
    slow_span = min(slow, max(fast_span + 1, len(w)))
    fast_ema = ema_series(w, fast_span)
    slow_ema = ema_series(w, slow_span)
    macd_line = fast_ema - slow_ema
    signal_span = min(signal, max(2, len(macd_line)))
    signal_line = ema_series(macd_line, signal_span)
    return float(macd_line[-1]), float(signal_line[-1])
