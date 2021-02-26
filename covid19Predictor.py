import torch
from torch import nn


class covid19Predictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, seq_length, num_layers=2):
        super(covid19Predictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.5
        )

        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim),
            torch.zeros(self.num_layers, self.seq_length, self.hidden_dim)
        )

    def forward(self, input):
        lstm_out, _ = self.lstm(
            input.view(len(input), self.seq_length, -1),
            self.hidden
        )

        y_pred = self.linear(
            lstm_out.view(self.seq_length, len(input), self.hidden_dim)[-1]
        )

        return y_pred