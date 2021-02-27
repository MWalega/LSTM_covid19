import torch
from torch import nn


class covid19Predictor(nn.Module):

    def __init__(self, n_features, n_hidden, seq_length, n_layers=2):
        super(covid19Predictor, self).__init__()

        self.n_hidden = n_hidden
        self.seq_length = seq_length
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_length, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_length, self.n_hidden)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_length, -1),
            self.hidden
        )



        y_pred = self.linear(
            lstm_out.view(self.seq_length, len(sequences), self.n_hidden)[-1]
        )

        return y_pred