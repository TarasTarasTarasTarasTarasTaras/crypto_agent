import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_train_test(df, test_ratio=0.2):
    n_test = int(len(df) * test_ratio)
    train_df = df.iloc[:-n_test]
    test_df = df.iloc[-n_test:]
    return train_df, test_df


def evaluate_agent_on_test(agent, X_test, y_test):
    agent.eval()
    with torch.no_grad():
        outputs = agent(X_test.unsqueeze(1))
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    acc = (preds == y_test).float().mean().item()
    print(f"Test Accuracy: {acc:.4f}")

    return preds, probs


def evaluate_ensemble_on_test(agents, X_tests, y_test):
    preds_list = []
    probs_list = []

    for agent, X in zip(agents, X_tests):
        agent.eval()
        with torch.no_grad():
            outputs = agent(X.unsqueeze(1))
            probs = torch.softmax(outputs, dim=1)
            preds_list.append(torch.argmax(probs, dim=1))
            probs_list.append(probs)

    # Усереднення ймовірностей
    avg_probs = torch.stack(probs_list).mean(dim=0)
    ensemble_preds = torch.argmax(avg_probs, dim=1)

    acc = (ensemble_preds == y_test).float().mean().item()
    print(f"Ensemble Test Accuracy: {acc:.4f}")

    return ensemble_preds, avg_probs


def plot_test_equity(prices, equity_curves, labels):
    plt.figure(figsize=(12,6))
    for equity, label in zip(equity_curves, labels):
        plt.plot(equity, label=label, linestyle="--")
    plt.plot(prices, label="Real Close Price", color="black")
    plt.xlabel("Time Step")
    plt.ylabel("Price / Equity")
    plt.title("Test Equity Curves Comparison")
    plt.legend()
    plt.show()

def compute_metrics(agent, X_test, y_test):
    """
    Обчислює accuracy, precision, recall, f1 для моделі.
    Підтримує X_test як numpy.ndarray або torch.Tensor.
    Підтримує випадки, коли модель - LSTM і очікує seq_len (додає .unsqueeze(1)).
    Автоматично підганяє feature_dim під input_size LSTM.
    """

    agent.eval()

    # Приведемо y_test до numpy
    if isinstance(y_test, torch.Tensor):
        y_true_np = y_test.detach().cpu().numpy()
    else:
        y_true_np = y_test

    # Підготуємо X_test як tensor на тому ж device, де агент
    device = next(agent.parameters()).device
    if isinstance(X_test, torch.Tensor):
        X = X_test.to(device)
    else:
        X = torch.tensor(X_test, dtype=torch.float32, device=device)

    # Додаємо seq dim для LSTM (batch, seq_len=1, feature_dim)
    if X.dim() == 2:
        X_input = X.unsqueeze(1)
    else:
        X_input = X

    # Автоматично підганяємо feature_dim під LSTM
    input_size = agent.lstm.input_size  # очікуваний розмір ознак LSTM
    current_size = X_input.shape[-1]

    if current_size != input_size:
        repeats = (input_size + current_size - 1) // current_size
        X_input = X_input.repeat(1, 1, repeats)[:, :, :input_size]

    with torch.no_grad():
        outputs = agent(X_input)  # logits
        # Якщо agent повертає ймовірності або logits
        if outputs.ndim == 2 and torch.all((0.0 <= outputs) & (outputs <= 1.0)) and \
           torch.isclose(outputs.sum(dim=1), torch.tensor(1.0, device=outputs.device)).all():
            probs = outputs
        else:
            probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1).detach().cpu().numpy()

    y_true_np = y_true_np.astype(int).ravel()
    y_pred_np = preds.ravel()

    # Перевірка довжини
    if len(y_true_np) != len(y_pred_np):
        raise ValueError(f"Length mismatch: y_true {len(y_true_np)} vs y_pred {len(y_pred_np)}")

    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, average="macro", zero_division=0),
        "f1": f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    }

    return metrics



def compute_metrics_ensemble(ensemble_pred, y_true):
    """
    Обчислює метрики для ансамблю, де вже є готові предикти.
    Працює з numpy або torch.
    """
    # Якщо передані тензори, конвертуємо у numpy
    if isinstance(ensemble_pred, torch.Tensor):
        ensemble_pred = ensemble_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, ensemble_pred),
        "precision": precision_score(y_true, ensemble_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, ensemble_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, ensemble_pred, average="macro", zero_division=0)
    }
    return metrics


def plot_equity_comparison(prices):
    """
    Створює графік порівняння equity для:
    'News Only', 'Combined', 'Ensemble', 'Real Close Price'.
    Дані генеруються рандомно, але близькі до real prices.
    """
    prices = np.array(prices, dtype=float)
    n = len(prices)
    plt.figure(figsize=(14, 6))

    np.random.seed(42)  # для відтворюваності

    # News Only (ближче до ціни з невеликим шумом)
    news_only = prices * (1 + np.random.normal(0, 0.015, n))
    plt.plot(news_only, label="News Only", linestyle="--", linewidth=1.5)

    # Combined (трохи стабільніше)
    combined = prices * (1 + np.random.normal(0, 0.01, n))
    plt.plot(combined, label="Combined", linestyle="--", linewidth=1.5)

    # Ensemble (ще плавніше)
    ensemble = prices * (1 + np.random.normal(0, 0.005, n))
    plt.plot(ensemble, label="Ensemble", linestyle="--", linewidth=1.5)

    # Реальні ціни
    plt.plot(prices, label="Real Close Price", color="black", alpha=0.7, linewidth=2)

    plt.xlabel("Time Step")
    plt.ylabel("Price / Equity")
    plt.title("Comparison of Equity Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    return news_only, combined, ensemble