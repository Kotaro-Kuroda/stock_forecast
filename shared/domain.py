from dataclasses import dataclass

import pandas as pd

ALIAS_SYMBOLS = {
    "TESLA": "TSLA",
    "テスラ": "TSLA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "NVIDIA": "NVDA",
    "META": "META",
    "TOYOTA": "7203.T",
    "トヨタ": "7203.T",
    "ソニー": "6758.T",
    "SONY": "6758.T",
    "任天堂": "7974.T",
    "NINTENDO": "7974.T",
    "ソフトバンク": "9984.T",
    "SOFTBANK": "9984.T",
    "BTC": "BTC-USD",
    "BITCOIN": "BTC-USD",
    "ビットコイン": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHEREUM": "ETH-USD",
    "イーサリアム": "ETH-USD",
    "XRP": "XRP-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "BNB": "BNB-USD",
}

SCREEN_UNIVERSES: dict[str, list[tuple[str, str]]] = {
    "米国主要": [
        ("AAPL", "Apple"),
        ("MSFT", "Microsoft"),
        ("NVDA", "NVIDIA"),
        ("AMZN", "Amazon"),
        ("META", "Meta"),
        ("GOOGL", "Alphabet"),
        ("TSLA", "Tesla"),
        ("AVGO", "Broadcom"),
        ("JPM", "JPMorgan"),
        ("LLY", "Eli Lilly"),
        ("V", "Visa"),
        ("WMT", "Walmart"),
    ],
    "日本主要": [
        ("7203.T", "Toyota"),
        ("6758.T", "Sony"),
        ("7974.T", "Nintendo"),
        ("9984.T", "SoftBank Group"),
        ("6861.T", "Keyence"),
        ("8035.T", "Tokyo Electron"),
        ("8306.T", "Mitsubishi UFJ"),
        ("9432.T", "NTT"),
        ("6501.T", "Hitachi"),
        ("8058.T", "Mitsubishi Corp"),
        ("4063.T", "Shin-Etsu"),
        ("6098.T", "Recruit"),
        ("7267.T", "Honda"),
        ("8053.T", "Sumitomo Corp"),
        ("2914.T", "Japan Tobacco"),
        ("8766.T", "Tokio Marine"),
        ("6902.T", "Denso"),
        ("8001.T", "Itochu"),
        ("9433.T", "KDDI"),
        ("4502.T", "Takeda"),
        ("7011.T", "Mitsubishi Heavy"),
        ("4578.T", "Otsuka"),
        ("1605.T", "INPEX"),
        ("6301.T", "Komatsu"),
    ],
    "仮想通貨主要": [
        ("BTC-USD", "Bitcoin"),
        ("ETH-USD", "Ethereum"),
        ("XRP-USD", "XRP"),
        ("SOL-USD", "Solana"),
        ("BNB-USD", "BNB"),
        ("DOGE-USD", "Dogecoin"),
        ("ADA-USD", "Cardano"),
    ],
}

SYMBOL_SECTORS: dict[str, str] = {
    # US
    "AAPL": "テクノロジー",
    "MSFT": "テクノロジー",
    "NVDA": "半導体",
    "AMZN": "一般消費財",
    "META": "コミュニケーション",
    "GOOGL": "コミュニケーション",
    "TSLA": "自動車",
    "AVGO": "半導体",
    "JPM": "金融",
    "LLY": "ヘルスケア",
    "V": "金融",
    "WMT": "生活必需品",
    # JP
    "7203.T": "自動車",
    "6758.T": "電機",
    "7974.T": "ゲーム",
    "9984.T": "通信・投資",
    "6861.T": "電機",
    "8035.T": "半導体",
    "8306.T": "金融",
    "9432.T": "通信",
    "6501.T": "電機",
    "8058.T": "商社",
    "4063.T": "素材",
    "6098.T": "サービス",
    "7267.T": "自動車",
    "8053.T": "商社",
    "2914.T": "生活必需品",
    "8766.T": "金融",
    "6902.T": "自動車部品",
    "8001.T": "商社",
    "9433.T": "通信",
    "4502.T": "ヘルスケア",
    "7011.T": "重工",
    "4578.T": "ヘルスケア",
    "1605.T": "エネルギー",
    "6301.T": "機械",
    # Crypto
    "BTC-USD": "暗号資産",
    "ETH-USD": "暗号資産",
    "XRP-USD": "暗号資産",
    "SOL-USD": "暗号資産",
    "BNB-USD": "暗号資産",
    "DOGE-USD": "暗号資産",
    "ADA-USD": "暗号資産",
}

SECTOR_OPTIONS: list[str] = [
    "すべて",
    "テクノロジー",
    "半導体",
    "コミュニケーション",
    "一般消費財",
    "生活必需品",
    "金融",
    "ヘルスケア",
    "自動車",
    "自動車部品",
    "通信",
    "通信・投資",
    "電機",
    "ゲーム",
    "商社",
    "素材",
    "重工",
    "エネルギー",
    "機械",
    "サービス",
    "暗号資産",
]


def sector_of(symbol: str) -> str:
    return SYMBOL_SECTORS.get(symbol.upper(), "その他")

DEFAULT_RF_FEATURE_FLAGS = {
    "return": True,
    "ma_ratio": True,
    "momentum": True,
    "volatility": True,
    "rsi": True,
    "macd": True,
}


@dataclass
class ForecastResult:
    history: pd.Series
    forecast: pd.Series
    backtest_actual: pd.Series
    backtest_pred: pd.Series
    backtest_mae: float
    backtest_rmse: float
    resolved_symbol: str
    company_name: str
    model_name: str
    optimization_note: str = ""
    rolling_mae: float = float("nan")
    rolling_rmse: float = float("nan")


@dataclass
class CandidateScore:
    symbol: str
    name: str
    last_price: float
    predicted_price: float
    expected_return: float
    momentum_60: float
    volatility_60: float
    score: float
