import torch
from src.model import CryptoAgent, NewsAttention
from src.preprocessing import get_news_embeddings

news_attention = NewsAttention()

def get_model_predictions(agent, X):
    agent.eval()
    with torch.no_grad():
        outputs = agent(X)
        return outputs  # ймовірності [batch, 2]

def prepare_ensemble_inputs(integrated_df):
    X_market_only = []
    X_news_only = []
    X_combined = []
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

        # Новини
        news_tokens_list = row['news_tokens']
        news_embeddings = [get_news_embeddings([news_tokens_list])] if news_tokens_list else [torch.zeros(768)]
        news_emb_avg = torch.mean(torch.stack(news_embeddings), dim=0)
        news_emb_attended = news_attention(news_embeddings)

        # Підготовка входів
        X_market_only.append(market_features)
        X_news_only.append(news_emb_avg)
        X_combined.append(torch.cat([market_features, news_emb_attended]))

        # Цільова ознака
        y.append(1 if next_row['close'] > row['close'] else 0)

    return torch.stack(X_market_only), torch.stack(X_news_only), torch.stack(X_combined), torch.tensor(y, dtype=torch.long)

def ensemble_predict(agent_market, agent_news, agent_combined, X_market, X_news, X_combined):
    pred_market = get_model_predictions(agent_market, X_market)
    pred_news = get_model_predictions(agent_news, X_news)
    pred_combined = get_model_predictions(agent_combined, X_combined)

    # Усереднюємо ймовірності
    final_probs = (pred_market + pred_news + pred_combined) / 3
    _, final_pred = torch.max(final_probs, 1)
    return final_pred

def run_ensemble(agent_market, agent_news, agent_combined, X_market, X_news, X_combined):
    """
    Генерує фінальний прогноз через ensemble трьох агентів.
    Усереднює ймовірності (soft voting).
    """
    agent_market.eval()
    agent_news.eval()
    agent_combined.eval()

    with torch.no_grad():
        probs_market = torch.softmax(agent_market(X_market), dim=1)
        probs_news = torch.softmax(agent_news(X_news), dim=1)
        probs_combined = torch.softmax(agent_combined(X_combined), dim=1)

        # Усереднення ймовірностей
        ensemble_probs = (probs_market + probs_news + probs_combined) / 3.0
        ensemble_pred = torch.argmax(ensemble_probs, dim=1)

    return ensemble_pred, ensemble_probs


def simulate_trading_with_preds(predicted, prices):
    """
    Симуляція торгівлі за сигналами ensemble (без використання агента).
    """
    cash = 1000
    position = 0
    equity = []

    for i in range(len(predicted)):
        price = prices[i]
        signal = predicted[i].item() if isinstance(predicted[i], torch.Tensor) else predicted[i]

        if signal == 1 and position == 0:
            position = 1
            buy_price = price
        elif signal == 0 and position == 1:
            position = 0
            cash += price - buy_price

        equity.append(cash + (price - buy_price if position == 1 else 0))

    return equity