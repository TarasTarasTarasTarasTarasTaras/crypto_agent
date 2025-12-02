import pandas as pd
import re
import string
from datetime import timedelta
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
from src.model import NewsAttention

# 1. Назва моделі BERT
bert_model_name = "bert-base-uncased"  # можна замінити на "ukrainian-bert" або інший, якщо потрібна українська

# 2. Ініціалізація токенізатора
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# 3. Ініціалізація самої моделі
bert_model = BertModel.from_pretrained(bert_model_name)

# 4. Переключення моделі у режим оцінки (без градієнтів)
bert_model.eval()

# 5. Якщо є GPU, можна перенести модель на CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)


news_attention = NewsAttention()

def prepare_model_input_with_attention(integrated_df):
    X = []
    y = []

    for i in range(len(integrated_df)-1):
        row = integrated_df.iloc[i]
        next_row = integrated_df.iloc[i+1]

        # Ринкові ознаки
        market_features = torch.tensor([
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ], dtype=torch.float)

        # Новини з Attention
        news_tokens_list = row['news_tokens']
        news_embeddings = [get_news_embeddings([tokens]) for tokens in [news_tokens_list] if tokens]
        news_emb_attended = news_attention(news_embeddings)

        # Об'єднуємо
        features = torch.cat([market_features, news_emb_attended])
        X.append(features)

        # Цільова ознака: напрямок ціни наступного кроку
        y.append(1 if next_row['close'] > row['close'] else 0)

    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


def preprocess_news_data(df):
    """
    Очищення тексту новин та підготовка до моделі.
    """
    cleaned_texts = []
    for text in df['title']:
        # Переводимо у нижній регістр
        text = text.lower()
        # Видаляємо URL
        text = re.sub(r"http\S+", "", text)
        # Видаляємо знаки пунктуації
        text = text.translate(str.maketrans('', '', string.punctuation))
        cleaned_texts.append(text)

    df['cleaned_title'] = cleaned_texts

    # Приклад токенізації BERT
    df['tokens'] = df['cleaned_title'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

    return df


def integrate_market_news(market_df, news_df, interval="1h"):
    """
    Прив'язує новини до відповідного таймфрейму ринкових даних.

    :param market_df: DataFrame ринкових даних (index = timestamp)
    :param news_df: DataFrame новин з колонками 'published_at' та 'tokens'
    :param interval: таймфрейм ринку ('1h', '30m' і т.д.)
    :return: DataFrame з інтегрованими ознаками
    """
    df = market_df.copy()

    # Переконуємося, що індекс market_df tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    # Переконуємося, що published_at tz-aware
    news_df = news_df.copy()
    news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')

    if news_df['published_at'].dt.tz is None:
        news_df['published_at'] = news_df['published_at'].dt.tz_localize('UTC')
    else:
        news_df['published_at'] = news_df['published_at'].dt.tz_convert('UTC')

    # Ініціалізація колонок
    df['news_tokens'] = [[] for _ in range(len(df))]
    df['news_count'] = 0

    # Перебір інтервалів ринку
    for idx, row in df.iterrows():
        start_time = idx
        if interval.endswith('h'):
            delta = timedelta(hours=int(interval[:-1]))
        elif interval.endswith('m'):
            delta = timedelta(minutes=int(interval[:-1]))
        else:
            delta = timedelta(hours=1)

        end_time = idx + delta

        # Локалізація start/end якщо tz-naive
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')
        if end_time.tzinfo is None:
            end_time = end_time.tz_localize('UTC')

        # Вибираємо новини у відповідному інтервалі
        mask = (news_df['published_at'] >= start_time) & (news_df['published_at'] < end_time)
        tokens_list = news_df.loc[mask, 'tokens'].tolist()

        # Об'єднуємо токени у один список
        if tokens_list:
            df.at[idx, 'news_tokens'] = [token for sublist in tokens_list for token in sublist]
            df.at[idx, 'news_count'] = len(tokens_list)
        else:
            df.at[idx, 'news_tokens'] = []
            df.at[idx, 'news_count'] = 0

    return df


bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name)



def get_news_embeddings(tokens_list):
    """
    Приймає список токенів новин для одного таймфрейму,
    повертає embedding через BERT ([CLS] токен).
    """
    if not tokens_list:
        return torch.zeros(768)  # або hidden_size твоєї BERT моделі

    # Переконаємося, що всі токени – рядки
    tokens_str_list = [str(token) for token in tokens_list]

    # Перетворюємо список токенів у рядок
    text = " ".join(tokens_str_list)

    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] токен

    return cls_embedding.squeeze(0)


def prepare_model_input(df):
    X_list = []
    y_list = []

    for i, row in df.iterrows():
        # Якщо колонки 'open', 'close' немає, пропускаємо або ставимо 0
        open_price = row['open'] if 'open' in df.columns else 0
        close_price = row['close'] if 'close' in df.columns else 0

        # Приклад формування фіч
        features = [
            open_price,
            close_price,
            # додай інші фічі, якщо потрібно
        ]
        X_list.append(features)

        # Цільова змінна: рух ціни
        y_val = int(close_price > open_price) if 'open' in df.columns and 'close' in df.columns else 0
        y_list.append(y_val)

    return np.array(X_list), np.array(y_list)



def prepare_news_only_input(df, max_len=None):
    X = []
    y = []

    for _, row in df.iterrows():
        tokens = row.get('news_tokens', None)
        target = row.get('target', None)

        if tokens is None or not isinstance(tokens, list):
            continue

        X.append(tokens)
        y.append(target)

    # Якщо max_len не заданий — беремо максимальну реальну довжину
    if max_len is None:
        max_len = max(len(seq) for seq in X)

    # Падінг усіх токенів до однієї довжини
    X_padded = np.zeros((len(X), max_len), dtype=np.float32)
    for i, seq in enumerate(X):
        length = min(len(seq), max_len)
        X_padded[i, :length] = seq[:length]

    return X_padded, np.array(y, dtype=np.int64)