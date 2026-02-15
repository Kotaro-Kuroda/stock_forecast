import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import font_manager, rcParams
from PyQt6.QtCore import Qt, QEvent, QTimer, QLineF
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.domain import (
    ALIAS_SYMBOLS,
    SCREEN_UNIVERSES,
    DEFAULT_RF_FEATURE_FLAGS,
    CandidateScore,
    ForecastResult,
)
from shared.feature_utils import (
    build_rf_feature_dataset,
    make_rf_feature_vector,
    normalize_rf_feature_flags,
)


class StockForecastApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.has_japanese_plot_font = configure_matplotlib_font()
        self.setWindowTitle("Stock Chart & Forecast (PyQt)")
        self.resize(1000, 700)

        central = QWidget()
        root = QVBoxLayout(central)

        form_layout = QGridLayout()
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(6)
        form_layout.setColumnStretch(1, 2)
        form_layout.setColumnStretch(3, 1)
        form_layout.setColumnStretch(5, 1)
        form_layout.setColumnStretch(7, 1)

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("例: AAPL, TSLA, 7203, BTC, ETH, BTC-USD")
        form_layout.addWidget(QLabel("銘柄名/コード"), 0, 0)
        form_layout.addWidget(self.symbol_input, 0, 1)

        self.period_input = QLineEdit("2y")
        self.period_input.setPlaceholderText("例: 6mo, 1y, 2y, 5y")
        form_layout.addWidget(QLabel("取得期間"), 0, 2)
        form_layout.addWidget(self.period_input, 0, 3)

        self.horizon_input = QSpinBox()
        self.horizon_input.setRange(1, 365)
        self.horizon_input.setValue(30)
        form_layout.addWidget(QLabel("予測日数"), 0, 4)
        form_layout.addWidget(self.horizon_input, 0, 5)

        self.lag_input = QSpinBox()
        self.lag_input.setRange(5, 120)
        self.lag_input.setValue(20)
        form_layout.addWidget(QLabel("ラグ日数"), 0, 6)
        form_layout.addWidget(self.lag_input, 0, 7)

        self.model_input = QComboBox()
        self.model_input.addItems(["RandomForest", "Prophet", "Ensemble", "XGBoost"])
        form_layout.addWidget(QLabel("予測モデル"), 1, 0)
        form_layout.addWidget(self.model_input, 1, 1)
        self.model_input.currentTextChanged.connect(self.on_model_changed)

        self.auto_optimize_input = QCheckBox("実行前に自動最適化")
        self.auto_optimize_input.setChecked(False)
        form_layout.addWidget(QLabel("最適化"), 1, 2)
        form_layout.addWidget(self.auto_optimize_input, 1, 3)

        self.screen_universe_input = QComboBox()
        self.screen_universe_input.addItems(list(SCREEN_UNIVERSES.keys()))
        form_layout.addWidget(QLabel("おすすめ対象"), 1, 4)
        form_layout.addWidget(self.screen_universe_input, 1, 5)

        self.screen_topn_input = QSpinBox()
        self.screen_topn_input.setRange(3, 20)
        self.screen_topn_input.setValue(5)
        form_layout.addWidget(QLabel("おすすめ件数"), 1, 6)
        form_layout.addWidget(self.screen_topn_input, 1, 7)

        self.daily_input = QCheckBox("daily")
        self.daily_input.setChecked(False)
        self.weekly_input = QCheckBox("weekly")
        self.weekly_input.setChecked(True)
        self.yearly_input = QCheckBox("yearly")
        self.yearly_input.setChecked(True)

        seasonality_layout = QHBoxLayout()
        seasonality_layout.addWidget(self.daily_input)
        seasonality_layout.addWidget(self.weekly_input)
        seasonality_layout.addWidget(self.yearly_input)
        form_layout.addWidget(QLabel("季節性"), 2, 0)
        form_layout.addLayout(seasonality_layout, 2, 1, 1, 3)

        self.changepoint_input = QDoubleSpinBox()
        self.changepoint_input.setDecimals(3)
        self.changepoint_input.setRange(0.001, 1.0)
        self.changepoint_input.setSingleStep(0.01)
        self.changepoint_input.setValue(0.05)
        form_layout.addWidget(QLabel("changepoint"), 2, 4)
        form_layout.addWidget(self.changepoint_input, 2, 5)

        self.rf_return_input = QCheckBox("return")
        self.rf_return_input.setChecked(True)
        self.rf_ma_ratio_input = QCheckBox("ma_ratio")
        self.rf_ma_ratio_input.setChecked(True)
        self.rf_momentum_input = QCheckBox("momentum")
        self.rf_momentum_input.setChecked(True)
        self.rf_volatility_input = QCheckBox("volatility")
        self.rf_volatility_input.setChecked(True)
        self.rf_rsi_input = QCheckBox("rsi")
        self.rf_rsi_input.setChecked(True)
        self.rf_macd_input = QCheckBox("macd")
        self.rf_macd_input.setChecked(True)

        rf_feature_layout = QHBoxLayout()
        rf_feature_layout.addWidget(self.rf_return_input)
        rf_feature_layout.addWidget(self.rf_ma_ratio_input)
        rf_feature_layout.addWidget(self.rf_momentum_input)
        rf_feature_layout.addWidget(self.rf_volatility_input)
        rf_feature_layout.addWidget(self.rf_rsi_input)
        rf_feature_layout.addWidget(self.rf_macd_input)
        form_layout.addWidget(QLabel("RF特徴量"), 3, 0)
        form_layout.addLayout(rf_feature_layout, 3, 1, 1, 7)

        root.addLayout(form_layout)

        controls = QHBoxLayout()
        self.run_button = QPushButton("チャート表示 + 予測")
        self.run_button.clicked.connect(self.on_run_clicked)
        controls.addWidget(self.run_button)
        self.optimize_button = QPushButton("最適化のみ実行")
        self.optimize_button.clicked.connect(self.on_optimize_clicked)
        controls.addWidget(self.optimize_button)
        self.screen_button = QPushButton("おすすめ銘柄を抽出")
        self.screen_button.clicked.connect(self.on_screen_clicked)
        controls.addWidget(self.screen_button)
        self.screen_jp_button = QPushButton("日本株おすすめを抽出")
        self.screen_jp_button.clicked.connect(self.on_screen_japan_clicked)
        controls.addWidget(self.screen_jp_button)

        self.status_label = QLabel("銘柄名/コードを入力して実行してください。")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        controls.addWidget(self.status_label, stretch=1)

        root.addLayout(controls)

        self.figure = Figure(figsize=(10, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.screen_table = QTableWidget(0, 8)
        self.screen_table.setHorizontalHeaderLabels(
            ["コード", "銘柄名", "現在値", "予測値(20営業日)", "予測上昇率", "60日モメンタム", "60日ボラ", "総合スコア"]
        )
        self.screen_table.verticalHeader().setVisible(False)
        self.screen_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.screen_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.screen_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.screen_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.tab_widget = QTabWidget()
        self.chart_tab = QWidget()
        chart_layout = QVBoxLayout(self.chart_tab)
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas)
        self.loading_overlay = LoadingOverlay(self.chart_tab)
        self.loading_overlay.setGeometry(self.chart_tab.rect())
        self.loading_overlay.hide()

        self.screen_tab = QWidget()
        screen_layout = QVBoxLayout(self.screen_tab)
        screen_layout.addWidget(self.screen_table)

        self.tab_widget.addTab(self.chart_tab, "チャート/予測")
        self.tab_widget.addTab(self.screen_tab, "おすすめ銘柄")
        root.addWidget(self.tab_widget)

        self.setCentralWidget(central)
        self.on_model_changed(self.model_input.currentText())

    def on_run_clicked(self) -> None:
        raw_symbol = self.symbol_input.text().strip()
        period = self.period_input.text().strip() or "2y"
        horizon = int(self.horizon_input.value())
        lag = int(self.lag_input.value())
        model_name = self.model_input.currentText()
        prophet_daily = self.daily_input.isChecked()
        prophet_weekly = self.weekly_input.isChecked()
        prophet_yearly = self.yearly_input.isChecked()
        changepoint_prior_scale = float(self.changepoint_input.value())
        auto_optimize = self.auto_optimize_input.isChecked()
        rf_feature_flags = self.get_rf_feature_flags()

        if not raw_symbol:
            self.show_error("銘柄名または銘柄コードを入力してください。")
            return

        self.run_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.screen_button.setEnabled(False)
        self.screen_jp_button.setEnabled(False)
        self.status_label.setText("データ取得と予測を実行中...")
        self.set_loading(True, "予測を計算中...")
        QApplication.processEvents()

        try:
            result = self.fetch_and_forecast(
                raw_symbol,
                period=period,
                forecast_days=horizon,
                lag=lag,
                model_name=model_name,
                prophet_daily=prophet_daily,
                prophet_weekly=prophet_weekly,
                prophet_yearly=prophet_yearly,
                changepoint_prior_scale=changepoint_prior_scale,
                auto_optimize=auto_optimize,
                rf_feature_flags=rf_feature_flags,
            )
            self.plot_result(result)
            self.tab_widget.setCurrentWidget(self.chart_tab)
            self.status_label.setText(
                f"表示完了: {result.resolved_symbol} / {result.model_name} / 予測 {horizon} 日 {result.optimization_note}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.show_error(str(exc))
            self.status_label.setText("エラーが発生しました。")
        finally:
            self.run_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.screen_button.setEnabled(True)
            self.screen_jp_button.setEnabled(True)
            self.set_loading(False)

    def fetch_and_forecast(
        self,
        raw_symbol: str,
        period: str = "2y",
        forecast_days: int = 30,
        lag: int = 20,
        model_name: str = "RandomForest",
        prophet_daily: bool = False,
        prophet_weekly: bool = True,
        prophet_yearly: bool = True,
        changepoint_prior_scale: float = 0.05,
        auto_optimize: bool = False,
        rf_feature_flags: dict | None = None,
    ) -> ForecastResult:
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)
        symbol = self.resolve_symbol(raw_symbol)

        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if data.empty:
            raise ValueError(
                "株価データを取得できませんでした。銘柄コードを確認してください。"
            )

        close = self.extract_close_series(data)
        if model_name in ["RandomForest", "Ensemble", "XGBoost"] and len(close) <= lag + 5:
            raise ValueError(
                f"学習データが不足しています。期間を長くするかラグ日数を減らしてください。現在件数: {len(close)}"
            )

        rf_params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 2,
        }
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
        prophet_params = {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_mode": "additive",
        }
        ensemble_weights = {"rf": 1 / 3, "prophet": 1 / 3, "xgb": 1 / 3}
        optimization_note = ""
        if auto_optimize:
            best = self.optimize_hyperparameters(
                close=close,
                model_name=model_name,
                lag=lag,
                prophet_daily=prophet_daily,
                prophet_weekly=prophet_weekly,
                prophet_yearly=prophet_yearly,
                rf_feature_flags=rf_feature_flags,
            )
            if model_name == "RandomForest":
                lag = int(best["lag"])
                rf_params = {
                    "n_estimators": int(best["n_estimators"]),
                    "max_depth": best["max_depth"],
                    "min_samples_leaf": int(best["min_samples_leaf"]),
                }
                optimization_note = (
                    f"(最適化: lag={best['lag']}, n={best['n_estimators']}, "
                    f"depth={best['max_depth']}, leaf={best['min_samples_leaf']})"
                )
            elif model_name == "Prophet":
                prophet_params = {
                    "changepoint_prior_scale": float(best["changepoint_prior_scale"]),
                    "seasonality_mode": str(best["seasonality_mode"]),
                }
                changepoint_prior_scale = float(best["changepoint_prior_scale"])
                optimization_note = (
                    f"(最適化: changepoint={best['changepoint_prior_scale']}, "
                    f"mode={best['seasonality_mode']})"
                )
            elif model_name == "Ensemble":
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
                prophet_params = {
                    "changepoint_prior_scale": float(best["changepoint_prior_scale"]),
                    "seasonality_mode": str(best["seasonality_mode"]),
                }
                changepoint_prior_scale = float(best["changepoint_prior_scale"])
                optimization_note = (
                    f"(最適化: RF lag={best['lag']} n={best['n_estimators']} "
                    f"depth={best['max_depth']} leaf={best['min_samples_leaf']} / "
                    f"XGB n={best['xgb_n_estimators']} depth={best['xgb_max_depth']} "
                    f"lr={best['xgb_learning_rate']} / "
                    f"Prophet changepoint={best['changepoint_prior_scale']} "
                    f"mode={best['seasonality_mode']})"
                )
            else:
                lag = int(best["lag"])
                xgb_params = {
                    "n_estimators": int(best["n_estimators"]),
                    "max_depth": int(best["max_depth"]),
                    "learning_rate": float(best["learning_rate"]),
                    "subsample": float(best["subsample"]),
                    "colsample_bytree": float(best["colsample_bytree"]),
                }
                optimization_note = (
                    f"(最適化: lag={best['lag']}, n={best['n_estimators']}, depth={best['max_depth']}, "
                    f"lr={best['learning_rate']}, subsample={best['subsample']}, colsample={best['colsample_bytree']})"
                )

        if model_name == "Ensemble":
            ensemble_weights = self.compute_ensemble_weights(
                series=close,
                lag=lag,
                rf_params=rf_params,
                xgb_params=xgb_params,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                changepoint_prior_scale=float(prophet_params["changepoint_prior_scale"]),
                seasonality_mode=str(prophet_params["seasonality_mode"]),
                rf_feature_flags=rf_feature_flags,
            )
            optimization_note = (
                f"{optimization_note} [w_rf={ensemble_weights['rf']:.2f}, "
                f"w_prophet={ensemble_weights['prophet']:.2f}, "
                f"w_xgb={ensemble_weights['xgb']:.2f}]"
            ).strip()

        if model_name == "Prophet":
            forecast = self.train_and_forecast_prophet(
                close,
                forecast_days=forecast_days,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                changepoint_prior_scale=float(prophet_params["changepoint_prior_scale"]),
                seasonality_mode=str(prophet_params["seasonality_mode"]),
            )
        elif model_name == "RandomForest":
            forecast = self.train_and_forecast_rf_with_params(
                close,
                forecast_days=forecast_days,
                lag=lag,
                n_estimators=int(rf_params["n_estimators"]),
                max_depth=rf_params["max_depth"],
                min_samples_leaf=int(rf_params["min_samples_leaf"]),
                rf_feature_flags=rf_feature_flags,
            )
        elif model_name == "Ensemble":
            rf_forecast = self.train_and_forecast_rf_with_params(
                close,
                forecast_days=forecast_days,
                lag=lag,
                n_estimators=int(rf_params["n_estimators"]),
                max_depth=rf_params["max_depth"],
                min_samples_leaf=int(rf_params["min_samples_leaf"]),
                rf_feature_flags=rf_feature_flags,
            )
            prophet_forecast = self.train_and_forecast_prophet(
                close,
                forecast_days=forecast_days,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                changepoint_prior_scale=float(prophet_params["changepoint_prior_scale"]),
                seasonality_mode=str(prophet_params["seasonality_mode"]),
            )
            xgb_forecast = self.train_and_forecast_xgb_with_params(
                close,
                forecast_days=forecast_days,
                lag=lag,
                n_estimators=int(xgb_params["n_estimators"]),
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                rf_feature_flags=rf_feature_flags,
            )
            forecast = self.weighted_average_three_predictions(
                rf_forecast,
                prophet_forecast,
                xgb_forecast,
                ensemble_weights["rf"],
                ensemble_weights["prophet"],
                ensemble_weights["xgb"],
            )
        else:
            forecast = self.train_and_forecast_xgb_with_params(
                close,
                forecast_days=forecast_days,
                lag=lag,
                n_estimators=int(xgb_params["n_estimators"]),
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                rf_feature_flags=rf_feature_flags,
            )

        backtest_actual, backtest_pred = self.run_backtest(
            close=close,
            model_name=model_name,
            lag=lag,
            prophet_daily=prophet_daily,
            prophet_weekly=prophet_weekly,
            prophet_yearly=prophet_yearly,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=str(prophet_params["seasonality_mode"]),
            rf_params=rf_params,
            xgb_params=xgb_params,
            ensemble_weights=ensemble_weights,
            rf_feature_flags=rf_feature_flags,
        )
        backtest_mae = float(np.mean(np.abs(backtest_actual.values - backtest_pred.values)))
        backtest_rmse = float(
            np.sqrt(np.mean((backtest_actual.values - backtest_pred.values) ** 2))
        )
        rolling_mae, rolling_rmse = self.evaluate_rolling_backtest(
            close=close,
            model_name=model_name,
            lag=lag,
            prophet_daily=prophet_daily,
            prophet_weekly=prophet_weekly,
            prophet_yearly=prophet_yearly,
            changepoint_prior_scale=float(prophet_params["changepoint_prior_scale"]),
            seasonality_mode=str(prophet_params["seasonality_mode"]),
            rf_params=rf_params,
            xgb_params=xgb_params,
            ensemble_weights=ensemble_weights,
            rf_feature_flags=rf_feature_flags,
        )

        company_name = self.get_company_name(symbol, fallback=raw_symbol)
        return ForecastResult(
            history=close,
            forecast=forecast,
            backtest_actual=backtest_actual,
            backtest_pred=backtest_pred,
            backtest_mae=backtest_mae,
            backtest_rmse=backtest_rmse,
            resolved_symbol=symbol,
            company_name=company_name,
            model_name=model_name,
            optimization_note=optimization_note,
            rolling_mae=rolling_mae,
            rolling_rmse=rolling_rmse,
        )

    def resolve_symbol(self, raw_symbol: str) -> str:
        text = unicodedata.normalize("NFKC", raw_symbol).strip()
        upper_text = text.upper()
        is_japanese = any(
            ("\u3040" <= ch <= "\u30ff") or ("\u4e00" <= ch <= "\u9fff")
            for ch in text
        )

        if text.isdigit():
            return f"{text}.T"

        if "." in text:
            return text.upper()

        # 仮想通貨ティッカー入力（BTC, ETH など）を USD 建てシンボルに変換。
        if "-" in upper_text and upper_text.endswith(("USD", "USDT", "JPY")):
            return upper_text
        if upper_text.isalpha() and 2 <= len(upper_text) <= 6:
            crypto_symbol = f"{upper_text}-USD"
            if self.has_recent_data(crypto_symbol):
                return crypto_symbol

        alias_symbol = ALIAS_SYMBOLS.get(upper_text) or ALIAS_SYMBOLS.get(text)
        if alias_symbol:
            return alias_symbol

        # 銘柄名検索を優先（TESLA のような企業名入力に対応）。
        try:
            query_candidates = [text]
            if is_japanese:
                query_candidates.extend([f"{text} 株価", f"{text} 日本"])

            for query in query_candidates:
                search = yf.Search(query=query, max_results=20)
                quotes = search.quotes or []
                if not quotes:
                    continue

                sorted_quotes = sorted(
                    quotes,
                    key=lambda q: self.quote_score(q, is_japanese=is_japanese),
                    reverse=True,
                )
                for quote in sorted_quotes[:5]:
                    symbol = str(quote.get("symbol", "")).upper()
                    if not symbol:
                        continue
                    if self.has_recent_data(symbol):
                        return symbol
        except Exception:
            pass

        # 最後にティッカー入力らしい形式のみ直接判定。
        if upper_text.isalnum() and 1 <= len(upper_text) <= 6:
            if self.has_recent_data(upper_text):
                return upper_text

        raise ValueError(
            "銘柄を解決できませんでした。例: TSLA, AAPL, 7203, 9984.T, トヨタ のように入力してください。"
        )

    def on_optimize_clicked(self) -> None:
        raw_symbol = self.symbol_input.text().strip()
        period = self.period_input.text().strip() or "2y"
        lag = int(self.lag_input.value())
        model_name = self.model_input.currentText()
        prophet_daily = self.daily_input.isChecked()
        prophet_weekly = self.weekly_input.isChecked()
        prophet_yearly = self.yearly_input.isChecked()
        rf_feature_flags = self.get_rf_feature_flags()

        if not raw_symbol:
            self.show_error("銘柄名または銘柄コードを入力してください。")
            return

        self.run_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.screen_button.setEnabled(False)
        self.screen_jp_button.setEnabled(False)
        self.status_label.setText("ハイパーパラメータ最適化を実行中...")
        self.set_loading(True, "最適化を計算中...")
        QApplication.processEvents()

        try:
            symbol = self.resolve_symbol(raw_symbol)
            data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
            close = self.extract_close_series(data)
            best = self.optimize_hyperparameters(
                close=close,
                model_name=model_name,
                lag=lag,
                prophet_daily=prophet_daily,
                prophet_weekly=prophet_weekly,
                prophet_yearly=prophet_yearly,
                rf_feature_flags=rf_feature_flags,
            )
            if model_name == "RandomForest":
                self.lag_input.setValue(int(best["lag"]))
                self.status_label.setText(
                    f"最適化完了: lag={best['lag']} n={best['n_estimators']} depth={best['max_depth']} leaf={best['min_samples_leaf']} cv_mae={best['cv_mae']}"
                )
            elif model_name == "Prophet":
                self.changepoint_input.setValue(float(best["changepoint_prior_scale"]))
                self.status_label.setText(
                    f"最適化完了: changepoint={best['changepoint_prior_scale']} mode={best['seasonality_mode']} cv_mae={best['cv_mae']}"
                )
            elif model_name == "Ensemble":
                self.lag_input.setValue(int(best["lag"]))
                self.changepoint_input.setValue(float(best["changepoint_prior_scale"]))
                self.status_label.setText(
                    "最適化完了: "
                    f"RF(lag={best['lag']}, n={best['n_estimators']}, depth={best['max_depth']}, leaf={best['min_samples_leaf']}, cv_mae={best['rf_cv_mae']}) "
                    f"+ XGB(n={best['xgb_n_estimators']}, depth={best['xgb_max_depth']}, lr={best['xgb_learning_rate']}, cv_mae={best['xgb_cv_mae']}) "
                    f"+ Prophet(changepoint={best['changepoint_prior_scale']}, mode={best['seasonality_mode']}, cv_mae={best['prophet_cv_mae']})"
                )
            else:
                self.lag_input.setValue(int(best["lag"]))
                self.status_label.setText(
                    "最適化完了: "
                    f"lag={best['lag']} n={best['n_estimators']} depth={best['max_depth']} "
                    f"lr={best['learning_rate']} subsample={best['subsample']} "
                    f"colsample={best['colsample_bytree']} cv_mae={best['cv_mae']}"
                )
        except Exception as exc:  # pylint: disable=broad-except
            self.show_error(str(exc))
            self.status_label.setText("最適化に失敗しました。")
        finally:
            self.run_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.screen_button.setEnabled(True)
            self.screen_jp_button.setEnabled(True)
            self.set_loading(False)

    def on_screen_clicked(self) -> None:
        universe_name = self.screen_universe_input.currentText()
        top_n = int(self.screen_topn_input.value())
        candidates = SCREEN_UNIVERSES.get(universe_name, [])
        rf_feature_flags = self.get_rf_feature_flags()
        if not candidates:
            self.show_error("対象ユニバースが空です。")
            return

        self.run_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.screen_button.setEnabled(False)
        self.screen_jp_button.setEnabled(False)
        self.status_label.setText("おすすめ銘柄を抽出中...")
        self.set_loading(True, "おすすめ銘柄を抽出中...")
        QApplication.processEvents()

        try:
            ranked = self.rank_candidates(candidates, top_n=top_n, rf_feature_flags=rf_feature_flags)
            self.update_screen_table(ranked)
            self.tab_widget.setCurrentWidget(self.screen_tab)
            self.status_label.setText(
                f"抽出完了: {universe_name} から {len(ranked)} 件を表示"
            )
        except Exception as exc:  # pylint: disable=broad-except
            self.show_error(str(exc))
            self.status_label.setText("おすすめ銘柄抽出に失敗しました。")
        finally:
            self.run_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.screen_button.setEnabled(True)
            self.screen_jp_button.setEnabled(True)
            self.set_loading(False)

    def on_screen_japan_clicked(self) -> None:
        top_n = int(self.screen_topn_input.value())
        candidates = SCREEN_UNIVERSES.get("日本主要", [])
        rf_feature_flags = self.get_rf_feature_flags()
        if not candidates:
            self.show_error("日本株ユニバースが空です。")
            return

        self.run_button.setEnabled(False)
        self.optimize_button.setEnabled(False)
        self.screen_button.setEnabled(False)
        self.screen_jp_button.setEnabled(False)
        self.status_label.setText("日本株おすすめを抽出中...")
        self.set_loading(True, "日本株おすすめを抽出中...")
        QApplication.processEvents()

        try:
            ranked = self.rank_candidates(candidates, top_n=top_n, rf_feature_flags=rf_feature_flags)
            self.update_screen_table(ranked)
            self.tab_widget.setCurrentWidget(self.screen_tab)
            self.status_label.setText(
                f"抽出完了: 日本株から {len(ranked)} 件を表示"
            )
            self.screen_universe_input.setCurrentText("日本主要")
        except Exception as exc:  # pylint: disable=broad-except
            self.show_error(str(exc))
            self.status_label.setText("日本株おすすめ抽出に失敗しました。")
        finally:
            self.run_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.screen_button.setEnabled(True)
            self.screen_jp_button.setEnabled(True)
            self.set_loading(False)

    def rank_candidates(
        self, candidates: list[tuple[str, str]], top_n: int, rf_feature_flags: dict
    ) -> list[CandidateScore]:
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)
        scored: list[CandidateScore] = []
        total = len(candidates)
        for idx, (symbol, name) in enumerate(candidates, start=1):
            self.status_label.setText(f"抽出中... {idx}/{total} {symbol}")
            QApplication.processEvents()

            try:
                data = yf.download(symbol, period="1y", progress=False, auto_adjust=True)
                close = self.extract_close_series(data)
                if len(close) < 140:
                    continue

                last_price = float(close.iloc[-1])
                predicted = self.train_and_forecast_rf_with_params(
                    close,
                    forecast_days=20,
                    lag=20,
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    rf_feature_flags=rf_feature_flags,
                )
                predicted_price = float(predicted.iloc[-1])
                expected_return = (predicted_price / last_price) - 1.0

                momentum_60 = (last_price / float(close.iloc[-60])) - 1.0
                returns = np.log(close / close.shift(1)).dropna()
                volatility_60 = float(returns.iloc[-60:].std() * np.sqrt(252))

                score = (0.6 * expected_return) + (0.4 * momentum_60) - (0.2 * volatility_60)
                scored.append(
                    CandidateScore(
                        symbol=symbol,
                        name=name,
                        last_price=last_price,
                        predicted_price=predicted_price,
                        expected_return=expected_return,
                        momentum_60=momentum_60,
                        volatility_60=volatility_60,
                        score=score,
                    )
                )
            except Exception:
                continue

        if not scored:
            raise ValueError("評価できる銘柄がありませんでした。時間をおいて再実行してください。")

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]

    def update_screen_table(self, ranked: list[CandidateScore]) -> None:
        self.screen_table.setRowCount(len(ranked))
        for r, item in enumerate(ranked):
            values = [
                item.symbol,
                item.name,
                f"{item.last_price:.2f}",
                f"{item.predicted_price:.2f}",
                f"{item.expected_return * 100:.2f}%",
                f"{item.momentum_60 * 100:.2f}%",
                f"{item.volatility_60 * 100:.2f}%",
                f"{item.score:.4f}",
            ]
            for c, value in enumerate(values):
                self.screen_table.setItem(r, c, QTableWidgetItem(value))

    def get_rf_feature_flags(self) -> dict:
        return normalize_rf_feature_flags(
            {
                "return": self.rf_return_input.isChecked(),
                "ma_ratio": self.rf_ma_ratio_input.isChecked(),
                "momentum": self.rf_momentum_input.isChecked(),
                "volatility": self.rf_volatility_input.isChecked(),
                "rsi": self.rf_rsi_input.isChecked(),
                "macd": self.rf_macd_input.isChecked(),
            }
        )

    @staticmethod
    def quote_score(quote: dict, is_japanese: bool) -> int:
        symbol = str(quote.get("symbol", "")).upper()
        exchange = str(quote.get("exchange", "")).upper()
        quote_type = str(quote.get("quoteType", "")).upper()

        score = 0
        if quote_type == "EQUITY":
            score += 3

        if is_japanese:
            if symbol.endswith(".T"):
                score += 5
            if any(x in exchange for x in ["JPX", "TSE", "TOKYO"]):
                score += 3
        else:
            if any(x in exchange for x in ["NMS", "NAS", "NYQ", "NYSE"]):
                score += 2
        return score

    @staticmethod
    def has_recent_data(symbol: str) -> bool:
        try:
            data = yf.download(symbol, period="1mo", progress=False, auto_adjust=True)
            return not data.empty
        except Exception:
            return False

    @staticmethod
    def get_company_name(symbol: str, fallback: str) -> str:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            name = info.get("longName") or info.get("shortName")
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            pass
        return fallback.strip() or symbol

    @staticmethod
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

    def run_backtest(
        self,
        close: pd.Series,
        model_name: str,
        lag: int,
        prophet_daily: bool,
        prophet_weekly: bool,
        prophet_yearly: bool,
        changepoint_prior_scale: float,
        seasonality_mode: str,
        rf_params: dict,
        xgb_params: dict,
        ensemble_weights: dict,
        rf_feature_flags: dict,
    ) -> tuple[pd.Series, pd.Series]:
        last_date = close.index.max()
        backtest_start = last_date - pd.DateOffset(months=3)

        train = close[close.index < backtest_start]
        test = close[close.index >= backtest_start]

        if len(test) < 20:
            raise ValueError(
                "直近3ヶ月の評価データが不足しています。取得期間を長くしてください（例: 2y 以上）。"
            )
        if len(train) < 60:
            raise ValueError(
                "バックテスト用の学習データが不足しています。取得期間を長くしてください。"
            )
        if model_name in ["RandomForest", "Ensemble", "XGBoost"] and len(train) <= lag + 5:
            raise ValueError(
                "RandomForest のバックテスト学習データが不足しています。期間を長くするかラグを減らしてください。"
            )

        horizon = len(test)
        if model_name == "Prophet":
            pred = self.train_and_forecast_prophet(
                train,
                forecast_days=horizon,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
            )
        elif model_name == "RandomForest":
            pred = self.train_and_forecast_rf_with_params(
                train,
                forecast_days=horizon,
                lag=lag,
                n_estimators=int(rf_params["n_estimators"]),
                max_depth=rf_params["max_depth"],
                min_samples_leaf=int(rf_params["min_samples_leaf"]),
                rf_feature_flags=rf_feature_flags,
            )
        elif model_name == "Ensemble":
            rf_pred = self.train_and_forecast_rf_with_params(
                train,
                forecast_days=horizon,
                lag=lag,
                n_estimators=int(rf_params["n_estimators"]),
                max_depth=rf_params["max_depth"],
                min_samples_leaf=int(rf_params["min_samples_leaf"]),
                rf_feature_flags=rf_feature_flags,
            )
            prophet_pred = self.train_and_forecast_prophet(
                train,
                forecast_days=horizon,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
            )
            xgb_pred = self.train_and_forecast_xgb_with_params(
                train,
                forecast_days=horizon,
                lag=lag,
                n_estimators=int(xgb_params["n_estimators"]),
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                rf_feature_flags=rf_feature_flags,
            )
            pred = self.weighted_average_three_predictions(
                rf_pred,
                prophet_pred,
                xgb_pred,
                float(ensemble_weights["rf"]),
                float(ensemble_weights["prophet"]),
                float(ensemble_weights["xgb"]),
            )
        else:
            pred = self.train_and_forecast_xgb_with_params(
                train,
                forecast_days=horizon,
                lag=lag,
                n_estimators=int(xgb_params["n_estimators"]),
                max_depth=int(xgb_params["max_depth"]),
                learning_rate=float(xgb_params["learning_rate"]),
                subsample=float(xgb_params["subsample"]),
                colsample_bytree=float(xgb_params["colsample_bytree"]),
                rf_feature_flags=rf_feature_flags,
            )

        pred = pred.reindex(test.index).dropna()
        actual = test.reindex(pred.index).dropna()
        pred = pred.reindex(actual.index)

        if len(actual) < 10:
            raise ValueError("バックテスト結果が不足しています。別の銘柄または期間で試してください。")
        return actual, pred

    def train_and_forecast_rf(self, series: pd.Series, forecast_days: int, lag: int) -> pd.Series:
        return self.train_and_forecast_rf_with_params(
            series=series,
            forecast_days=forecast_days,
            lag=lag,
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            rf_feature_flags=DEFAULT_RF_FEATURE_FLAGS,
        )

    def train_and_forecast_rf_with_params(
        self,
        series: pd.Series,
        forecast_days: int,
        lag: int,
        n_estimators: int,
        max_depth: int | None,
        min_samples_leaf: int,
        rf_feature_flags: dict,
    ) -> pd.Series:
        values = series.values.astype(float)
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)

        x, y = build_rf_feature_dataset(values, lag, rf_feature_flags)
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
            features = np.array(
                make_rf_feature_vector(np.array(history[-lag:], dtype=float), rf_feature_flags)
            ).reshape(1, -1)
            next_price = float(model.predict(features)[0])
            preds.append(next_price)
            history.append(next_price)

        last_date = series.index[-1]
        future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        return pd.Series(preds, index=future_index, name="Forecast")

    def train_and_forecast_xgb_with_params(
        self,
        series: pd.Series,
        forecast_days: int,
        lag: int,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        rf_feature_flags: dict,
    ) -> pd.Series:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ValueError(
                "XGBoost が利用できません。`uv sync` で依存関係をインストールしてください。"
            ) from exc

        values = series.values.astype(float)
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)
        x, y = build_rf_feature_dataset(values, lag, rf_feature_flags)
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
            features = np.array(
                make_rf_feature_vector(np.array(history[-lag:], dtype=float), rf_feature_flags)
            ).reshape(1, -1)
            next_price = float(model.predict(features)[0])
            preds.append(next_price)
            history.append(next_price)

        last_date = series.index[-1]
        future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        return pd.Series(preds, index=future_index, name="Forecast")

    def train_and_forecast_prophet(
        self,
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
            raise ValueError(
                "Prophet が利用できません。`uv sync` で依存関係をインストールしてください。"
            ) from exc

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
        predicted = model.predict(future)

        pred_only = predicted[["ds", "yhat"]].tail(forecast_days).copy()
        pred_only["ds"] = pd.to_datetime(pred_only["ds"])
        pred_only = pred_only.set_index("ds")
        return pred_only["yhat"].rename("Forecast")

    def optimize_hyperparameters(
        self,
        close: pd.Series,
        model_name: str,
        lag: int,
        prophet_daily: bool,
        prophet_weekly: bool,
        prophet_yearly: bool,
        rf_feature_flags: dict,
    ) -> dict:
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)
        if model_name == "RandomForest":
            return self.optimize_rf_hyperparameters(
                close=close, base_lag=lag, rf_feature_flags=rf_feature_flags
            )
        if model_name == "XGBoost":
            return self.optimize_xgb_hyperparameters(
                close=close, base_lag=lag, rf_feature_flags=rf_feature_flags
            )
        if model_name == "Ensemble":
            return self.optimize_ensemble_hyperparameters(
                close=close,
                base_lag=lag,
                daily=prophet_daily,
                weekly=prophet_weekly,
                yearly=prophet_yearly,
                rf_feature_flags=rf_feature_flags,
            )
        return self.optimize_prophet_hyperparameters(
            close=close,
            daily=prophet_daily,
            weekly=prophet_weekly,
            yearly=prophet_yearly,
        )

    def optimize_ensemble_hyperparameters(
        self,
        close: pd.Series,
        base_lag: int,
        daily: bool,
        weekly: bool,
        yearly: bool,
        rf_feature_flags: dict,
    ) -> dict:
        rf_best = self.optimize_rf_hyperparameters(
            close=close, base_lag=base_lag, rf_feature_flags=rf_feature_flags
        )
        xgb_best = self.optimize_xgb_hyperparameters(
            close=close, base_lag=base_lag, rf_feature_flags=rf_feature_flags
        )
        prophet_best = self.optimize_prophet_hyperparameters(
            close=close,
            daily=daily,
            weekly=weekly,
            yearly=yearly,
        )
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

    def optimize_rf_hyperparameters(
        self, close: pd.Series, base_lag: int, rf_feature_flags: dict
    ) -> dict:
        values = close.values.astype(float)
        candidate_lags = sorted(
            set([max(5, base_lag - 10), base_lag, min(120, base_lag + 10), 10, 30])
        )
        candidate_lags = [lag for lag in candidate_lags if lag + 30 < len(values)]
        if not candidate_lags:
            raise ValueError("最適化に必要なデータが不足しています。期間を長くしてください。")

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float("inf")
        best_params: dict | None = None
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)

        for lag in candidate_lags:
            x, y = build_rf_feature_dataset(values, lag, rf_feature_flags)
            if len(x) < 50:
                continue
            for n_estimators in [150, 300, 500]:
                for max_depth in [None, 8, 16]:
                    for min_samples_leaf in [1, 2, 4]:
                        fold_maes = []
                        for train_idx, val_idx in tscv.split(x):
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
                        score = float(np.mean(fold_maes))
                        if score < best_score:
                            best_score = score
                            best_params = {
                                "lag": lag,
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "min_samples_leaf": min_samples_leaf,
                                "cv_mae": round(score, 4),
                            }
        if best_params is None:
            raise ValueError("RandomForest の最適化に失敗しました。")
        return best_params

    def optimize_xgb_hyperparameters(
        self, close: pd.Series, base_lag: int, rf_feature_flags: dict
    ) -> dict:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ValueError("XGBoost が利用できません。`uv sync` を実行してください。") from exc

        values = close.values.astype(float)
        candidate_lags = sorted(
            set([max(5, base_lag - 10), base_lag, min(120, base_lag + 10), 10, 30])
        )
        candidate_lags = [lag for lag in candidate_lags if lag + 30 < len(values)]
        if not candidate_lags:
            raise ValueError("最適化に必要なデータが不足しています。期間を長くしてください。")

        tscv = TimeSeriesSplit(n_splits=3)
        best_score = float("inf")
        best_params: dict | None = None
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)

        for lag in candidate_lags:
            x, y = build_rf_feature_dataset(values, lag, rf_feature_flags)
            if len(x) < 50:
                continue
            for n_estimators in [200, 400]:
                for max_depth in [4, 6, 8]:
                    for learning_rate in [0.03, 0.05, 0.1]:
                        for subsample in [0.8, 1.0]:
                            for colsample_bytree in [0.8, 1.0]:
                                fold_maes = []
                                for train_idx, val_idx in tscv.split(x):
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
                                    model.fit(x[train_idx], y[train_idx])
                                    pred = model.predict(x[val_idx])
                                    mae = float(np.mean(np.abs(y[val_idx] - pred)))
                                    fold_maes.append(mae)
                                score = float(np.mean(fold_maes))
                                if score < best_score:
                                    best_score = score
                                    best_params = {
                                        "lag": lag,
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "learning_rate": learning_rate,
                                        "subsample": subsample,
                                        "colsample_bytree": colsample_bytree,
                                        "cv_mae": round(score, 4),
                                    }
        if best_params is None:
            raise ValueError("XGBoost の最適化に失敗しました。")
        return best_params

    def optimize_prophet_hyperparameters(
        self, close: pd.Series, daily: bool, weekly: bool, yearly: bool
    ) -> dict:
        if len(close) < 160:
            raise ValueError("Prophet 最適化にはより長い期間のデータが必要です（2y 以上推奨）。")
        if not any([daily, weekly, yearly]):
            raise ValueError("Prophet の季節性を1つ以上選択してください。")

        tscv = TimeSeriesSplit(n_splits=3)
        cps_candidates = [0.01, 0.03, 0.05, 0.1, 0.2]
        mode_candidates = ["additive", "multiplicative"]
        values = close.values.astype(float)
        best_score = float("inf")
        best_params: dict | None = None

        for cps in cps_candidates:
            for mode in mode_candidates:
                fold_maes = []
                for train_idx, val_idx in tscv.split(values):
                    train_series = close.iloc[train_idx]
                    val_series = close.iloc[val_idx]
                    pred = self.train_and_forecast_prophet(
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
                if not fold_maes:
                    continue
                score = float(np.mean(fold_maes))
                if score < best_score:
                    best_score = score
                    best_params = {
                        "changepoint_prior_scale": cps,
                        "seasonality_mode": mode,
                        "cv_mae": round(score, 4),
                    }

        if best_params is None:
            raise ValueError("Prophet の最適化に失敗しました。")
        return best_params

    @staticmethod
    def average_predictions(first: pd.Series, second: pd.Series) -> pd.Series:
        aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
        if aligned.empty:
            raise ValueError("アンサンブル予測を作成できませんでした。")
        return ((aligned["first"] + aligned["second"]) / 2.0).rename("Forecast")

    @staticmethod
    def weighted_average_predictions(
        first: pd.Series, second: pd.Series, first_weight: float, second_weight: float
    ) -> pd.Series:
        aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
        if aligned.empty:
            raise ValueError("重み付きアンサンブル予測を作成できませんでした。")
        total = max(first_weight + second_weight, 1e-9)
        fw = first_weight / total
        sw = second_weight / total
        return (aligned["first"] * fw + aligned["second"] * sw).rename("Forecast")

    @staticmethod
    def weighted_average_three_predictions(
        first: pd.Series,
        second: pd.Series,
        third: pd.Series,
        first_weight: float,
        second_weight: float,
        third_weight: float,
    ) -> pd.Series:
        aligned = pd.concat(
            [first.rename("first"), second.rename("second"), third.rename("third")], axis=1
        ).dropna()
        if aligned.empty:
            raise ValueError("3モデルの重み付きアンサンブル予測を作成できませんでした。")
        total = max(first_weight + second_weight + third_weight, 1e-9)
        fw = first_weight / total
        sw = second_weight / total
        tw = third_weight / total
        return (aligned["first"] * fw + aligned["second"] * sw + aligned["third"] * tw).rename(
            "Forecast"
        )

    def compute_ensemble_weights(
        self,
        series: pd.Series,
        lag: int,
        rf_params: dict,
        xgb_params: dict,
        daily: bool,
        weekly: bool,
        yearly: bool,
        changepoint_prior_scale: float,
        seasonality_mode: str,
        rf_feature_flags: dict,
    ) -> dict:
        rf_mae, _ = self.evaluate_rolling_backtest(
            close=series,
            model_name="RandomForest",
            lag=lag,
            prophet_daily=daily,
            prophet_weekly=weekly,
            prophet_yearly=yearly,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            rf_params=rf_params,
            xgb_params=xgb_params,
            ensemble_weights={"rf": 1 / 3, "prophet": 1 / 3, "xgb": 1 / 3},
            rf_feature_flags=rf_feature_flags,
        )
        prophet_mae, _ = self.evaluate_rolling_backtest(
            close=series,
            model_name="Prophet",
            lag=lag,
            prophet_daily=daily,
            prophet_weekly=weekly,
            prophet_yearly=yearly,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            rf_params=rf_params,
            xgb_params=xgb_params,
            ensemble_weights={"rf": 1 / 3, "prophet": 1 / 3, "xgb": 1 / 3},
            rf_feature_flags=rf_feature_flags,
        )
        xgb_mae, _ = self.evaluate_rolling_backtest(
            close=series,
            model_name="XGBoost",
            lag=lag,
            prophet_daily=daily,
            prophet_weekly=weekly,
            prophet_yearly=yearly,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            rf_params=rf_params,
            xgb_params=xgb_params,
            ensemble_weights={"rf": 1 / 3, "prophet": 1 / 3, "xgb": 1 / 3},
            rf_feature_flags=rf_feature_flags,
        )
        inv_rf = 1.0 / max(rf_mae, 1e-9)
        inv_prophet = 1.0 / max(prophet_mae, 1e-9)
        inv_xgb = 1.0 / max(xgb_mae, 1e-9)
        total = inv_rf + inv_prophet + inv_xgb
        return {"rf": inv_rf / total, "prophet": inv_prophet / total, "xgb": inv_xgb / total}

    def evaluate_rolling_backtest(
        self,
        close: pd.Series,
        model_name: str,
        lag: int,
        prophet_daily: bool,
        prophet_weekly: bool,
        prophet_yearly: bool,
        changepoint_prior_scale: float,
        seasonality_mode: str,
        rf_params: dict,
        xgb_params: dict,
        ensemble_weights: dict,
        rf_feature_flags: dict,
        window_days: int = 20,
        folds: int = 3,
    ) -> tuple[float, float]:
        rf_feature_flags = normalize_rf_feature_flags(rf_feature_flags)
        if len(close) < 200:
            return float("nan"), float("nan")
        total = len(close)
        maes = []
        rmses = []
        for fold_idx in range(folds, 0, -1):
            test_end = total - (fold_idx - 1) * window_days
            test_start = test_end - window_days
            train_end = test_start
            if train_end < max(80, lag + 20):
                continue
            train = close.iloc[:train_end]
            test = close.iloc[test_start:test_end]
            if len(test) < 10:
                continue
            if model_name == "RandomForest":
                pred = self.train_and_forecast_rf_with_params(
                    train,
                    forecast_days=len(test),
                    lag=lag,
                    n_estimators=int(rf_params["n_estimators"]),
                    max_depth=rf_params["max_depth"],
                    min_samples_leaf=int(rf_params["min_samples_leaf"]),
                    rf_feature_flags=rf_feature_flags,
                )
            elif model_name == "Prophet":
                pred = self.train_and_forecast_prophet(
                    train,
                    forecast_days=len(test),
                    daily=prophet_daily,
                    weekly=prophet_weekly,
                    yearly=prophet_yearly,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_mode=seasonality_mode,
                )
            elif model_name == "Ensemble":
                rf_pred = self.train_and_forecast_rf_with_params(
                    train,
                    forecast_days=len(test),
                    lag=lag,
                    n_estimators=int(rf_params["n_estimators"]),
                    max_depth=rf_params["max_depth"],
                    min_samples_leaf=int(rf_params["min_samples_leaf"]),
                    rf_feature_flags=rf_feature_flags,
                )
                prophet_pred = self.train_and_forecast_prophet(
                    train,
                    forecast_days=len(test),
                    daily=prophet_daily,
                    weekly=prophet_weekly,
                    yearly=prophet_yearly,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_mode=seasonality_mode,
                )
                xgb_pred = self.train_and_forecast_xgb_with_params(
                    train,
                    forecast_days=len(test),
                    lag=lag,
                    n_estimators=int(xgb_params["n_estimators"]),
                    max_depth=int(xgb_params["max_depth"]),
                    learning_rate=float(xgb_params["learning_rate"]),
                    subsample=float(xgb_params["subsample"]),
                    colsample_bytree=float(xgb_params["colsample_bytree"]),
                    rf_feature_flags=rf_feature_flags,
                )
                pred = self.weighted_average_three_predictions(
                    rf_pred,
                    prophet_pred,
                    xgb_pred,
                    float(ensemble_weights["rf"]),
                    float(ensemble_weights["prophet"]),
                    float(ensemble_weights["xgb"]),
                )
            else:
                pred = self.train_and_forecast_xgb_with_params(
                    train,
                    forecast_days=len(test),
                    lag=lag,
                    n_estimators=int(xgb_params["n_estimators"]),
                    max_depth=int(xgb_params["max_depth"]),
                    learning_rate=float(xgb_params["learning_rate"]),
                    subsample=float(xgb_params["subsample"]),
                    colsample_bytree=float(xgb_params["colsample_bytree"]),
                    rf_feature_flags=rf_feature_flags,
                )
            pred = pred.reindex(test.index).dropna()
            actual = test.reindex(pred.index).dropna()
            pred = pred.reindex(actual.index)
            if len(actual) < 5:
                continue
            err = actual.values - pred.values
            maes.append(float(np.mean(np.abs(err))))
            rmses.append(float(np.sqrt(np.mean(err**2))))
        if not maes:
            return float("nan"), float("nan")
        return float(np.mean(maes)), float(np.mean(rmses))

    def plot_result(self, result: ForecastResult) -> None:
        self.figure.clear()
        axes = self.figure.subplots(2, 1, sharex=False, gridspec_kw={"height_ratios": [2, 1]})
        ax = axes[0]
        bx = axes[1]

        actual_label = "実績終値" if self.has_japanese_plot_font else "Actual Close"
        forecast_label = "予測終値" if self.has_japanese_plot_font else "Forecast Close"
        chart_title = (
            f"{result.resolved_symbol} ({result.company_name}) 株価チャートと将来予測 ({result.model_name})"
            if self.has_japanese_plot_font
            else f"{result.resolved_symbol} ({result.company_name}) Stock Chart & Forecast ({result.model_name})"
        )

        ax.plot(result.history.index, result.history.values, label=actual_label, linewidth=1.5)
        ax.plot(result.forecast.index, result.forecast.values, label=forecast_label, linewidth=2)

        ax.set_title(chart_title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.3)
        ax.legend()

        backtest_title = (
            f"直近3ヶ月バックテスト: 予測 vs 実測  (MAE={result.backtest_mae:.2f}, RMSE={result.backtest_rmse:.2f}) "
            f"/ Rolling(MAE={result.rolling_mae:.2f}, RMSE={result.rolling_rmse:.2f})"
            if self.has_japanese_plot_font
            else f"Last 3 Months Backtest: Pred vs Actual  (MAE={result.backtest_mae:.2f}, RMSE={result.backtest_rmse:.2f}) "
            f"/ Rolling(MAE={result.rolling_mae:.2f}, RMSE={result.rolling_rmse:.2f})"
        )
        backtest_actual_label = "実測終値" if self.has_japanese_plot_font else "Actual Close"
        backtest_pred_label = "予測終値" if self.has_japanese_plot_font else "Predicted Close"

        bx.plot(
            result.backtest_actual.index,
            result.backtest_actual.values,
            label=backtest_actual_label,
            linewidth=1.4,
        )
        bx.plot(
            result.backtest_pred.index,
            result.backtest_pred.values,
            label=backtest_pred_label,
            linewidth=1.8,
        )
        bx.set_title(backtest_title, fontsize=11)
        bx.set_xlabel("Date")
        bx.set_ylabel("Price")
        bx.grid(alpha=0.3)
        bx.legend()

        self.canvas.draw_idle()

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "エラー", message)

    def set_loading(self, is_loading: bool, message: str = "計算中...") -> None:
        if is_loading:
            self.tab_widget.setCurrentWidget(self.chart_tab)
            self.loading_overlay.setGeometry(self.chart_tab.rect())
            self.loading_overlay.start(message)
        else:
            self.loading_overlay.stop()

    def on_model_changed(self, model_name: str) -> None:
        needs_prophet = model_name in ["Prophet", "Ensemble"]
        needs_rf = model_name in ["RandomForest", "Ensemble", "XGBoost"]
        self.daily_input.setEnabled(needs_prophet)
        self.weekly_input.setEnabled(needs_prophet)
        self.yearly_input.setEnabled(needs_prophet)
        self.changepoint_input.setEnabled(needs_prophet)
        self.lag_input.setEnabled(needs_rf)
        self.rf_return_input.setEnabled(needs_rf)
        self.rf_ma_ratio_input.setEnabled(needs_rf)
        self.rf_momentum_input.setEnabled(needs_rf)
        self.rf_volatility_input.setEnabled(needs_rf)
        self.rf_rsi_input.setEnabled(needs_rf)
        self.rf_macd_input.setEnabled(needs_rf)


class CircleProgressIndicator(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.setInterval(70)
        self.timer.timeout.connect(self._rotate)
        self.setFixedSize(56, 56)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _rotate(self) -> None:
        self.angle = (self.angle + 30) % 360
        self.update()

    def start(self) -> None:
        self.timer.start()
        self.show()

    def stop(self) -> None:
        self.timer.stop()
        self.hide()

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(self.width(), self.height()) / 2 - 6

        for i in range(12):
            painter.save()
            painter.translate(center_x, center_y)
            painter.rotate(self.angle - i * 30)
            alpha = max(25, 255 - i * 18)
            painter.setPen(QColor(33, 150, 243, alpha))
            painter.drawLine(QLineF(0.0, -radius, 0.0, -radius + 10.0))
            painter.restore()


class LoadingOverlay(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: rgba(0, 0, 0, 70);")
        self.spinner = CircleProgressIndicator(self)
        self.message_label = QLabel("計算中...", self)
        self.message_label.setStyleSheet("color: white; font-size: 14px;")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parent.installEventFilter(self)

    def start(self, message: str) -> None:
        self.message_label.setText(message)
        self.setGeometry(self.parentWidget().rect())
        self._layout_children()
        self.show()
        self.raise_()
        self.spinner.start()

    def stop(self) -> None:
        self.spinner.stop()
        self.hide()

    def _layout_children(self) -> None:
        w = self.width()
        h = self.height()
        self.spinner.move((w - self.spinner.width()) // 2, (h - self.spinner.height()) // 2 - 10)
        self.message_label.setGeometry((w - 240) // 2, self.spinner.y() + self.spinner.height() + 8, 240, 24)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self.parentWidget() and event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parentWidget().rect())
            self._layout_children()
        return super().eventFilter(obj, event)


def check_dependencies() -> None:
    missing = []
    try:
        import PyQt6  # noqa: F401
    except Exception:
        missing.append("PyQt6")

    try:
        import yfinance  # noqa: F401
    except Exception:
        missing.append("yfinance")

    try:
        import sklearn  # noqa: F401
    except Exception:
        missing.append("scikit-learn")

    try:
        import xgboost  # noqa: F401
    except Exception:
        missing.append("xgboost")

    try:
        import prophet  # noqa: F401
    except Exception:
        missing.append("prophet")

    try:
        import matplotlib  # noqa: F401
    except Exception:
        missing.append("matplotlib")

    if missing:
        package_list = " ".join(missing)
        raise SystemExit(
            "必要ライブラリが不足しています。\n"
            f"以下をインストールしてください: uv add {package_list}"
        )


def configure_matplotlib_font() -> bool:
    jp_font_candidates = [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "MS Gothic",
    ]
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}

    for font_name in jp_font_candidates:
        if font_name in installed_fonts:
            rcParams["font.family"] = [font_name, "DejaVu Sans", "sans-serif"]
            rcParams["axes.unicode_minus"] = False
            return True

    rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    rcParams["axes.unicode_minus"] = False
    return False


def main() -> None:
    check_dependencies()
    app = QApplication(sys.argv)
    window = StockForecastApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
