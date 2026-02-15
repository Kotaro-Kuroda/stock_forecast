# PyQt 株価チャート + 時系列予測アプリ

銘柄名または銘柄コードを入力すると株価チャートを表示し、時系列予測モデルで将来価格を予測するデスクトップアプリです。

## 機能
- 銘柄名/コード入力 (`AAPL`, `MSFT`, `7203`, `9984.T` など)
- `yfinance` から株価終値を取得
- `matplotlib` で実績チャートを表示
- 予測モデルを GUI で切り替え
- `RandomForestRegressor`（ラグ + テクニカル特徴量）
- `Prophet`（トレンド・季節性モデル）
- `Ensemble`（RandomForest / Prophet / XGBoost の重み付き平均）
- `XGBoost`（勾配ブースティング回帰）
- 予測日数/ラグ日数を GUI から調整
- `Prophet` の季節性（daily/weekly/yearly）と `changepoint_prior_scale` を GUI で調整
- 日本株は `銘柄コード(7203)` に加えて `銘柄名(例: トヨタ)` でも検索可能
- 仮想通貨にも対応（`BTC`, `ETH`, `XRP`, `BTC-USD` など）
- 直近3ヶ月バックテストに加えて、ローリングバックテスト指標も表示
- matplotlib ツールバーで拡大・縮小・パン・表示リセットが可能
- `TimeSeriesSplit` ベースのハイパーパラメータ自動最適化に対応
- おすすめ銘柄スクリーニング（米国主要/日本主要/仮想通貨主要）を表示
- 日本株専用の `日本株おすすめを抽出` ボタンに対応
- `チャート/予測` タブと `おすすめ銘柄` タブを切り替えて表示可能
- RF追加特徴量（return/ma_ratio/momentum/volatility/rsi/macd）をGUIでON/OFF可能

## セットアップ（uv）
```bash
cd /Users/kurodakotaro/Documents/Stock/pyqt_stock_forecast
uv venv
source .venv/bin/activate
uv sync
```

## 実行
```bash
cd /Users/kurodakotaro/Documents/Stock/pyqt_stock_forecast
uv run app.py
```

## ファイル構成
- `app.py`: PyQt GUI と画面イベント処理
- `domain.py`: 定数（銘柄ユニバース/エイリアス）とデータクラス
- `feature_utils.py`: RF/XGBoost用の特徴量生成ロジック

## 使い方
1. `銘柄名/コード` に入力
2. `取得期間`, `予測日数`, `ラグ日数` を必要に応じて調整
3. `予測モデル` で `RandomForest` / `Prophet` / `Ensemble` / `XGBoost` を選択
4. `Prophet` 選択時は `季節性` と `changepoint` を調整
5. `RandomForest` / `Ensemble` では `RF特徴量` のチェックで追加特徴量をON/OFF
6. 必要に応じて `実行前に自動最適化` をオン
7. 先に最適化だけしたい場合は `最適化のみ実行` を押す
8. `チャート表示 + 予測` を押す
9. おすすめ機能を使う場合は `おすすめ対象` と `おすすめ件数` を選んで `おすすめ銘柄を抽出` を押す
10. 日本株だけを素早く抽出したい場合は `日本株おすすめを抽出` を押す

## 予測モデルについて
- `RandomForest`
  - 過去 `lag` 日分の終値に加えて、RSI/MACD/モメンタム/ボラティリティ等を特徴量として学習
  - 1日先を逐次予測し、予測値を次ステップ入力に再利用
- `Prophet`
  - 日付と終値からトレンドと季節性を学習
  - 営業日ベースで将来 `forecast_days` 日を予測
  - `季節性`: `daily` / `weekly` / `yearly` の有効化を選択
  - `changepoint_prior_scale`: トレンド変化への追従の強さを調整
- バックテスト（共通）
  - 直近3ヶ月を除いたデータで学習
  - 学習モデルで直近3ヶ月を予測し、実測と比較
  - MAE / RMSE をグラフタイトルに表示
- 自動最適化
  - `RandomForest`: `lag`, `n_estimators`, `max_depth`, `min_samples_leaf`
  - `XGBoost`: `lag`, `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
  - `Prophet`: `changepoint_prior_scale`, `seasonality_mode`
  - `Ensemble`: RandomForest / XGBoost / Prophet を個別最適化し、ローリング誤差から重みを自動推定
  - `TimeSeriesSplit(n_splits=3)` の平均 MAE で最良パラメータを選択
- おすすめ銘柄スクリーニング
  - 各銘柄に対して 1年データから 20営業日先の予測上昇率を計算
  - 60日モメンタムと60日ボラティリティを加えて総合スコア化
  - スコア上位をテーブル表示

## 注意
- 予測は参考値であり、投資判断を保証するものではありません。
- 入力が銘柄名の場合、`yfinance` の検索 API 依存のため一致しない場合があります。
- `Importing plotly failed` と表示される場合は `uv sync` を再実行してください。
- 日本語フォントが見つからない環境では、チャート内テキストは自動で英語表示にフォールバックします。
