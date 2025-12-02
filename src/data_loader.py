# src/data_loader.py
import ccxt
import requests
import pandas as pd
from datetime import datetime

CRYPTO_PANIC_API_KEY = "28632b5ff5225bba0186643e2e7a70b60abb27e2"  # потрібно зареєструватися на CryptoPanic

def get_market_data(symbol="BTC/USDT", timeframe="1h", limit=500):
    """
    Завантажує історичні дані з Binance через ccxt.

    :param symbol: торговельна пара
    :param timeframe: таймфрейм свічок (1m, 5m, 1h, 1d)
    :param limit: кількість свічок для завантаження
    :return: DataFrame з ринковими даними
    """
    binance = ccxt.binance()
    print(f"Завантаження даних для {symbol} з Binance...")

    # Завантаження OHLCV
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # Конвертуємо у DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    print(f"Завантажено {len(df)} записів.")
    return df

# Додамо до існуючого файлу


def get_news_data(public=True, filter="crypto", limit=100):
    """
    Завантажує останні новини з CryptoPanic API.

    :param public: True для публічних новин
    :param filter: категорія новин
    :param limit: кількість новин
    :return: DataFrame з новинами
    """
    print(f"Завантаження новин з CryptoPanic...")

    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTO_PANIC_API_KEY,
        "public": int(public),
        "filter": filter,
        "kind": "news",
        "limit": limit
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Витягаємо потрібні поля
    news_list = []
    for item in data.get("results", []):
        title = item.get("title", "")
        if not title:
            continue  # Пропускаємо новини без заголовка

        news_list.append({
            "title": title,
            "published_at": item.get("published_at"),
            "domain": item.get("domain"),
            "source": item.get("source", {}).get("title") if item.get("source") else None,
            "url": item.get("url") or item.get("link") or "",
            "kind": item.get("kind"),
            "votes": item.get("votes", {})
        })

    df = pd.DataFrame(news_list)
    df.sort_values(by="published_at", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Завантажено {len(df)} новин.")
    return df

