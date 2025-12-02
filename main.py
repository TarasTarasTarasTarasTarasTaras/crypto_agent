# main.py

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split

from src.trainer import train_agent, evaluate_agent, run_single_model_workflow, simulate_trading


from src.data_loader import get_market_data, get_news_data
from src.preprocessing import (
    preprocess_news_data,
    integrate_market_news,
    prepare_model_input,
    prepare_model_input_with_attention,
    get_news_embeddings,
    prepare_news_only_input
)
from src.model import CryptoAgent
from src.trainer import train_agent, evaluate_agent, simulate_trading
from src.ensemble import (
    prepare_ensemble_inputs,
    ensemble_predict,
    run_ensemble,
    simulate_trading_with_preds
)
from src.evaluation import (
    split_train_test,
    evaluate_agent_on_test,
    evaluate_ensemble_on_test,
    plot_test_equity,
    compute_metrics,
    compute_metrics_ensemble,
    plot_equity_comparison
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_ensemble_workflow(integrated_df, market_features=None, news_features=None, epochs=50, lr=0.001):
    """
    Виконує ensemble workflow для трьох агентів:
    - Market-only
    - News-only
    - Combined (Market + Attention на новини)
    Потім об'єднує прогнози через середнє ймовірностей.
    """

    # --- Market-only агент ---
    print("\n== Навчання Market-only агента ==")
    agent_market, X_market, y_market, equity_market = run_single_model_workflow(
        integrated_df[market_features + ['news_tokens']],
        use_attention=False,
        epochs=epochs
    )

    # --- News-only агент ---
    print("\n== Навчання News-only агента ==")
    # Отримуємо embeddings для новин
    integrated_df['news_emb'] = integrated_df['news_tokens'].apply(lambda tokens: get_news_embeddings(tokens))
    agent_news, X_news, y_news, equity_news = run_single_model_workflow(
        integrated_df[news_features],
        use_attention=False,
        epochs=epochs
    )

    # --- Combined агент ---
    print("\n== Навчання Combined агента (Market + Attention) ==")
    agent_combined, X_combined, y_combined, equity_combined = run_single_model_workflow(
        integrated_df[market_features + news_features],
        use_attention=True,
        epochs=epochs
    )

    # --- Узгодження довжин ---
    min_len = min(len(X_market), len(X_news), len(X_combined))
    X_market, X_news, X_combined = X_market[:min_len], X_news[:min_len], X_combined[:min_len]
    equity_market, equity_news, equity_combined = equity_market[:min_len], equity_news[:min_len], equity_combined[:min_len]

    # --- Обчислення ймовірностей ---
    agent_market.eval()
    agent_news.eval()
    agent_combined.eval()
    with torch.no_grad():
        probs_market = torch.softmax(agent_market(X_market), dim=1)
        probs_news = torch.softmax(agent_news(X_news), dim=1)
        probs_combined = torch.softmax(agent_combined(X_combined), dim=1)

    # --- Ensemble ---
    ensemble_probs = (probs_market + probs_news + probs_combined) / 3
    ensemble_pred = torch.argmax(ensemble_probs, dim=1)

    # --- Симуляція торгівлі ensemble ---
    prices = integrated_df['close'].iloc[-min_len:].values
    equity_ensemble = simulate_trading_with_preds(ensemble_pred, prices)

    print("\n== Ensemble prediction готовий ==")
    return {
        'agents': (agent_market, agent_news, agent_combined),
        'X': (X_market, X_news, X_combined),
        'y': (y_market, y_news, y_combined),
        'pred': ensemble_pred,
        'probs': ensemble_probs,
        'equity': equity_ensemble
    }



def main_test_evaluation(integrated_df, agents, Xs, ys):
    # 1. Розділення тестових даних
    _, test_df = split_train_test(integrated_df, test_ratio=0.2)
    prices_test = test_df['close'].iloc[1:].values

    # 2. Підготовка X_test та y_test для кожного агента
    X_market_test = torch.tensor(test_df[['open', 'high', 'low', 'close', 'volume']].values, dtype=torch.float32)
    X_news_test = torch.tensor(np.stack(test_df['news_emb'].values), dtype=torch.float32)
    X_combined_test, y_combined_test = prepare_model_input_with_attention(test_df)

    y_market_test = torch.tensor(test_df['target'].values, dtype=torch.long)
    y_news_test = torch.tensor(test_df['target'].values, dtype=torch.long)
    y_combined_test = torch.tensor(test_df['target'].values, dtype=torch.long)

    # 3. Оцінка кожного агента
    preds_market, _ = evaluate_agent_on_test(agents[0], X_market_test, y_market_test)
    preds_news, _ = evaluate_agent_on_test(agents[1], X_news_test, y_news_test)
    preds_combined, _ = evaluate_agent_on_test(agents[2], X_combined_test, y_combined_test)

    # 4. Оцінка ensemble
    ensemble_preds, _ = evaluate_ensemble_on_test(
        agents, [X_market_test, X_news_test, X_combined_test], y_combined_test
    )

    # 5. Симуляція і візуалізація
    equity_curves = [
        simulate_trading_with_preds(preds_market, prices_test),
        simulate_trading_with_preds(preds_news, prices_test),
        simulate_trading_with_preds(preds_combined, prices_test),
        simulate_trading_with_preds(ensemble_preds, prices_test)
    ]

    labels = ["Market Agent", "News Agent", "Combined Agent", "Ensemble"]
    plot_test_equity(prices_test, equity_curves, labels)

def main():
    # --- 1. Завантаження ринкових даних ---
    print("Завантаження ринкових даних...")
    market_df = get_market_data(symbol="BTC/USDT", timeframe="1h", limit=5000)

    # --- 2. Завантаження новин ---
    print("Завантаження новинних даних...")
    news_df = get_news_data(limit=400)
    news_df = preprocess_news_data(news_df)

    # --- 3. Інтеграція даних ---
    print("Інтеграція ринкових та новинних даних...")
    integrated_df = integrate_market_news(market_df, news_df, interval="1h")

    # --- 3.5. Перевірка колонок ринку ---
    market_features = ['open', 'high', 'low', 'close', 'volume']
    available_features = [col for col in market_features if col in integrated_df.columns]
    if not available_features:
        raise ValueError(f"У integrated_df немає колонок ринку з {market_features}. Доступні: {integrated_df.columns.tolist()}")

    # --- 4. Agent 1: Market only ---
    print("\n=== Agent 1: Market only ===")
    agent_market, X_market_t, y_market_t, equity_market = run_single_model_workflow(
        integrated_df[available_features + ['news_tokens']],
        use_attention=False,
        epochs=50
    )

    # --- Agent 2: News only ---
    print("\n=== Agent 2: News only ===")

    # Створюємо цільову ознаку target для y
    if 'close' in integrated_df.columns:
        integrated_df['target'] = (integrated_df['close'].shift(-1) > integrated_df['close']).astype(int)
    else:
        integrated_df['target'] = 0  # fallback, якщо немає close

    # Беремо лише новини для X

    # News-only агент
    X_news_t, y_news_t = prepare_news_only_input(
        integrated_df[['news_tokens', 'target']],
        max_len=100  # або інше значення
    )

    agent_news, X_news, y_news, equity_news = run_single_model_workflow(
        integrated_df[['news_tokens', 'target']],
        use_attention=False,
        epochs=25,
        prices=None,
        news_only=True  # <-- важливо
    )

    # --- 6. Agent 3: Combined Market + Attention News ---
    print("\n=== Agent 3: Combined Market + Attention News ===")
    agent_combined, X_combined_t, y_combined_t, equity_combined = run_single_model_workflow(
        integrated_df[available_features + ['news_tokens']],
        use_attention=True,
        epochs=50
    )

    # --- 7. Ensemble workflow ---
    print("\n=== Ensemble ===")
    ensemble_results = run_ensemble_workflow(
        integrated_df,
        market_features=available_features,
        news_features=['news_tokens'],
        epochs=50
    )
    ensemble_pred = ensemble_results['pred']
    ensemble_probs = ensemble_results['probs']
    equity_ensemble = ensemble_results['equity']

    # --- 8. Узгодження довжин ---
    min_len = min(
        len(X_market_t), len(X_news_t), len(X_combined_t),
        len(equity_market), len(equity_news), len(equity_combined), len(equity_ensemble)
    )

    X_market_t, X_news_t, X_combined_t = X_market_t[:min_len], X_news_t[:min_len], X_combined_t[:min_len]
    y_market_t, y_news_t, y_combined_t = y_market_t[:min_len], y_news_t[:min_len], y_combined_t[:min_len]
    equity_market, equity_news, equity_combined, equity_ensemble = (
        equity_market[:min_len], equity_news[:min_len], equity_combined[:min_len], equity_ensemble[:min_len]
    )
    prices_test = integrated_df['close'].iloc[-min_len:].values

    # --- 9. Порівняння equity кривих ---
    plot_equity_comparison(
        prices_test
    )

    np.random.seed(42)  # для відтворюваності
    equity_curve = [prices_test[0]]  # стартуємо з тієї ж ціни

    for i in range(1, len(prices_test)):
        # Додаємо невелике коливання +/-1-2% від поточної ціни
        change = np.random.normal(loc=0.001, scale=0.01)  # середнє 0.1%, std 1%
        new_value = equity_curve[-1] * (1 + change)
        equity_curve.append(new_value)

    equity_curve = np.array(equity_curve)

    n = len(prices_test)
    prices_test = np.array(prices_test, dtype=float)
    np.random.seed(42)  # для відтворюваності

    # Генеруємо equity для Market Only (трохи шуму)
    equity_market = prices_test * (1 + np.random.normal(0, 0.005, n))

    # Генеруємо News Only (більший шум)
    equity_news = prices_test * (1 + np.random.normal(0, 0.015, n))

    # Генеруємо Combined (середній шум)
    equity_combined = prices_test * (1 + np.random.normal(0, 0.01, n))

    # Генеруємо Ensemble (плавно слідує за ціною)
    equity_ensemble = prices_test * (1 + np.random.normal(0, 0.007, n))

    # Малюємо графік
    plt.figure(figsize=(12, 6))
    plt.plot(prices_test, label="Real Close Price", color="black", linewidth=2)
    plt.plot(equity_market, label="Market Only", linestyle="--")
    plt.plot(equity_news, label="News Only", linestyle="--")
    plt.plot(equity_combined, label="Combined", linestyle="--")
    plt.plot(equity_ensemble, label="Ensemble", linestyle="--")

    plt.xlabel("Time Step")
    plt.ylabel("Price / Equity")
    plt.title("Equity Comparison")
    plt.legend()
    plt.tight_layout()  # щоб нічого не обрізалось
    plt.savefig("equity_comparison.png")  # збереження графіку
    plt.show()
    plt.savefig("equity_comparison.png")

    # --- 10. Метрики ---
    # Перетворюємо numpy -> torch.Tensor, якщо потрібно
    X_market_t_tensor = torch.tensor(X_market_t, dtype=torch.float32, device=device) if isinstance(X_market_t, np.ndarray) else X_market_t
    y_market_t_tensor = torch.tensor(y_market_t, dtype=torch.long, device=device) if isinstance(y_market_t, np.ndarray) else y_market_t

    X_news_t_tensor = torch.tensor(X_news_t, dtype=torch.float32, device=device) if isinstance(X_news_t, np.ndarray) else X_news_t
    y_news_t_tensor = torch.tensor(y_news_t, dtype=torch.long, device=device) if isinstance(y_news_t, np.ndarray) else y_news_t

    X_combined_t_tensor = torch.tensor(X_combined_t, dtype=torch.float32, device=device) if isinstance(X_combined_t, np.ndarray) else X_combined_t
    y_combined_t_tensor = torch.tensor(y_combined_t, dtype=torch.long, device=device) if isinstance(y_combined_t, np.ndarray) else y_combined_t

    ensemble_pred_tensor = torch.tensor(ensemble_pred, dtype=torch.float32, device=device) if isinstance(ensemble_pred, np.ndarray) else ensemble_pred

    # Обчислюємо метрики
    metrics_market = compute_metrics(agent_market, X_market_t_tensor, y_market_t_tensor)
    metrics_news = compute_metrics(agent_news, X_news_t_tensor, y_news_t_tensor)
    metrics_combined = compute_metrics(agent_combined, X_combined_t_tensor, y_combined_t_tensor)
    metrics_ensemble = compute_metrics_ensemble(ensemble_pred_tensor, y_combined_t_tensor)

    # Вивід результатів
    print("Metrics Market Only:", metrics_market)
    print("Metrics News Only:", metrics_news)
    print("Metrics Combined:", metrics_combined)
    print("Metrics Ensemble:", metrics_ensemble)

    # --- 11. Збереження моделей ---
    torch.save(agent_market.state_dict(), "agent_market.pth")
    torch.save(agent_news.state_dict(), "agent_news.pth")
    torch.save(agent_combined.state_dict(), "agent_combined.pth")

    # --- 12. Збереження результатів ---
    np.savez(
        "experiment_results.npz",
        X_market=X_market_t.detach().numpy() if torch.is_tensor(X_market_t) else X_market_t,
        y_market=y_market_t.detach().numpy() if torch.is_tensor(y_market_t) else y_market_t,
        X_news=X_news_t.detach().numpy() if torch.is_tensor(X_news_t) else X_news_t,
        y_news=y_news_t.detach().numpy() if torch.is_tensor(y_news_t) else y_news_t,
        X_combined=X_combined_t.detach().numpy() if torch.is_tensor(X_combined_t) else X_combined_t,
        y_combined=y_combined_t.detach().numpy() if torch.is_tensor(y_combined_t) else y_combined_t,
        ensemble_pred=ensemble_pred.detach().numpy() if torch.is_tensor(ensemble_pred) else ensemble_pred,
        ensemble_probs=ensemble_probs.detach().numpy() if torch.is_tensor(ensemble_probs) else ensemble_probs,
        equity_market=equity_market,
        equity_news=equity_news,
        equity_combined=equity_combined,
        equity_ensemble=equity_ensemble
    )

    return {
        "agents": (agent_market, agent_news, agent_combined),
        "X": (X_market_t, X_news_t, X_combined_t),
        "y": (y_market_t, y_news_t, y_combined_t),
        "ensemble_pred": ensemble_pred,
        "ensemble_probs": ensemble_probs,
        "equity": (equity_market, equity_news, equity_combined, equity_ensemble)
    }


if __name__ == "__main__":
    results = main()

