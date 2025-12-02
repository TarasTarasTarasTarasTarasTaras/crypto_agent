import torch
import torch.nn as nn

class CryptoAgent(nn.Module):
    def __init__(self, input_size=773, hidden_size=128, num_layers=2, output_size=2):
        """
        LSTM агент для прогнозу напрямку ціни криптовалюти.
        input_size = 5 ринкових + 768 BERT
        output_size = 2 (up/down)
        """
        super(CryptoAgent, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch, features]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # додаємо seq_len=1
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]      # беремо останній hidden state
        out = self.fc(out)       # logits
        return out               # НЕ застосовуємо softmax тут, CrossEntropyLoss робить це автоматично


class NewsAttention(nn.Module):
    def __init__(self, embedding_size=768):
        super(NewsAttention, self).__init__()
        self.attention = nn.Linear(embedding_size, 1)

    def forward(self, news_embeddings_list):
        """
        news_embeddings_list: список torch.Tensor [num_news, embedding_size]
        """
        if not news_embeddings_list:
            return torch.zeros(768)  # BERT embedding size

        news_tensor = torch.stack(news_embeddings_list)  # [num_news, 768]
        weights = torch.softmax(self.attention(news_tensor), dim=0)  # [num_news,1]
        weighted_sum = torch.sum(weights * news_tensor, dim=0)
        return weighted_sum