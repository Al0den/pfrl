from python.market_cache import MarketCache

market = MarketCache()

tickers = [
    # Tech / Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA",
    "INTC", "CSCO", "ORCL", "IBM", "ADBE",
    "QCOM", "AVGO", "TXN",

    # Industrie / Défense
    "BA", "CAT", "GE", "LMT", "NOC", "HON", "DE",
    "MMM", "GD", "RTX", "ETN", "EMR", "ITW",

    # Finance
    "JPM", "GS", "MS", "BAC", "C", "WFC",
    "BLK", "AXP", "PNC", "SCHW",

    # Consommation / Staples
    "PG", "KO", "PEP", "WMT", "COST",
    "MCD", "NKE", "HD", "LOW", "SBUX",

    # Santé
    "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "ABT", "TMO", "BMY",

    # Énergie / Matières premières
    "XOM", "CVX", "COP", "SLB", "EOG",
    "PSX", "VLO",

    # Telecom / Media
    "T", "VZ", "DIS", "CMCSA",

    # Indice de référence
    "^GSPC"
]


matrice_poids_alpha = 1
matrice_poids_seuil = 0.1
