import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .preprocessing import prepare_model_input, prepare_model_input_with_attention, prepare_news_only_input
from .model import CryptoAgent  # переконайся, що твій агент тут
import matplotlib.pyplot as plt


def evaluate_agent(agent, X, y, device='cpu'):
    agent.eval()
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
        outputs = agent(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        print(f"Accuracy: {accuracy*100:.2f}%")
    return predicted, outputs


def simulate_trading(agent, X, y, prices, initial_capital=1000.0):
    """
    Симуляція торгівлі за сигналами агента з початковим капіталом.
    Підганяє довжину prices під X автоматично.

    :param agent: натренований агент
    :param X: вхідні ознаки (torch.Tensor)
    :param y: фактичний напрямок ціни (не використовується)
    :param prices: ціни (list або np.array)
    :param initial_capital: початковий капітал
    :return: equity curve (np.array)
    """
    if len(prices) < len(X):
        raise ValueError(f"Length of prices ({len(prices)}) менше ніж X ({len(X)}).")

    prices = prices[-len(X):]  # підганяємо довжину під X

    agent.eval()
    with torch.no_grad():
        outputs = agent(X)
        _, predicted = torch.max(outputs, 1)

    cash = initial_capital
    position = 0
    buy_price = 0.0
    equity = []

    for i in range(len(predicted)):
        price = float(prices[i])
        signal = predicted[i].item() if isinstance(predicted[i], torch.Tensor) else int(predicted[i])

        if signal == 1 and position == 0:
            position = 1
            buy_price = price
        elif signal == 0 and position == 1:
            position = 0
            cash += price - buy_price
            buy_price = 0.0

        equity.append(cash + (price - buy_price if position == 1 else 0))

    equity = np.array(equity, dtype=float)

    # Обчислення ROI та Sharpe Ratio
    if len(equity) > 1:
        roi = (equity[-1] / initial_capital - 1) * 100
        returns = np.diff(equity) / (equity[:-1] + 1e-6)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        print(f"Final ROI: {roi:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Недостатньо даних для обчислення ROI та Sharpe Ratio.")

    return equity


def simulate_trading(agent, X, y, prices, initial_capital=1000.0, device='cpu'):
    if len(prices) < len(X):
        raise ValueError(f"Length of prices ({len(prices)}) менше ніж X ({len(X)}).")

    prices = np.array(prices[-len(X):], dtype=float)
    X, y = X.to(device), y.to(device)
    agent.to(device)

    agent.eval()
    with torch.no_grad():
        outputs = agent(X)
        _, predicted = torch.max(outputs, 1)

    cash = initial_capital
    position = 0
    buy_price = 0.0
    equity = []

    for i in range(len(predicted)):
        price = prices[i]
        signal = predicted[i].item()
        if signal == 1 and position == 0:
            position = 1
            buy_price = price
        elif signal == 0 and position == 1:
            position = 0
            cash += price - buy_price
            buy_price = 0.0
        equity.append(cash + (price - buy_price if position == 1 else 0))

    equity = np.array(equity, dtype=float)

    if len(equity) > 1:
        roi = (equity[-1] / initial_capital - 1) * 100
        returns = np.diff(equity) / (equity[:-1] + 1e-6)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        print(f"Final ROI: {roi:.2f}%, Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Недостатньо даних для обчислення ROI та Sharpe Ratio.")

    return equity

def run_single_model_workflow(
    integrated_df,
    use_attention=False,
    epochs=15,
    lr=0.001,
    price_col='close',
    prices=None,
    news_only=False,
    device='cpu'
):
    """
    Виконує повний цикл для одного агента:
    – Підготовка X, y
    – Навчання агента
    – Оцінка
    – Симуляція торгівлі + графік

    :param integrated_df: DataFrame з ринковими та/або новинними даними
    :param use_attention: чи використовувати attention для новин
    :param epochs: кількість епох навчання
    :param lr: learning rate
    :param price_col: колонка для генерації цільової змінної (якщо prices не передані)
    :param prices: масив цін для симуляції
    :param news_only: чи використовувати тільки новини
    :param device: 'cpu' або 'cuda'
    :return: agent, X_tensor, y_tensor, equity
    """

    # --- 1. Підготовка X та y ---
    if news_only:
        X_np, y_np = prepare_news_only_input(integrated_df)  # функція news-only
    else:
        X_np, y_np = prepare_model_input(integrated_df)

    # --- 2. Підготовка цін ---
    if prices is not None:
        prices_arr = np.array(prices)
    elif not news_only and price_col in integrated_df.columns:
        prices_arr = integrated_df[price_col].values
    else:
        prices_arr = np.arange(len(X_np))  # штучний рядок для news-only

    # --- 3. Узгодження довжин ---
    min_len = min(len(X_np), len(y_np), len(prices_arr))
    X_np = X_np[-min_len:]
    y_np = y_np[-min_len:]
    prices_arr = prices_arr[-min_len:]

    # --- 4. Конвертація у torch.Tensor ---
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_np, dtype=torch.long).to(device)

    print(f"✅ Data ready! X: {X_tensor.shape}, y: {y_tensor.shape}, class balance: {torch.bincount(y_tensor)}")

    # --- 5. Ініціалізація агента ---
    input_size = X_tensor.shape[1]
    num_classes = len(torch.unique(y_tensor))
    agent = CryptoAgent(input_size=input_size, output_size=num_classes).to(device)

    # --- 6. Навчання агента ---
    print(f"\n== Training agent (attention={use_attention}, news_only={news_only}) ==")
    train_agent(agent, X_tensor, y_tensor, epochs=epochs, lr=lr, use_attention=use_attention)

    # --- 7. Оцінка агента ---
    print(f"\n== Evaluating agent ==")
    evaluate_agent(agent, X_tensor, y_tensor)

    # --- 8. Симуляція торгівлі ---
    equity = simulate_trading(agent, X_tensor, y_tensor, prices_arr)

    # --- 9. Візуалізація ---
    plt.figure(figsize=(12,6))
    plt.plot(prices_arr, label="Real Price" if not news_only else "Index")
    plt.plot(equity, label="Equity Curve", linestyle="--")
    plt.xlabel("Time Step")
    plt.ylabel("Price / Equity")
    plt.title(f"Equity Curve – attention={use_attention}, news_only={news_only}")
    plt.legend()
    plt.show()

    return agent, X_tensor, y_tensor, equity


def train_agent(agent, X, y, epochs=15, lr=0.001, use_attention=False):
    """
    Тренування агента на одному X, y
    """
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = agent(X)  # LSTM / Attention всередині CryptoAgent
        loss = criterion(outputs, y)
        loss.backward()  # backward один раз
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y).float().mean().item() * 100
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")

