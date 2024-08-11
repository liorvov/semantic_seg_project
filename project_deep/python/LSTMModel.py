import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_d, hidden_d, layer_d, output_d):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_d
        self.layer_dim = layer_d
        self.lstm1 = nn.LSTM(input_d, hidden_d, layer_d, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(hidden_d * 2, hidden_d, layer_d, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_d * 2, output_d)

    def forward(self, x):
        out, (hn, cn) = self.lstm1(x)
        #  out, (hn, cn) = self.lstm2(out)
        out = self.fc(out.squeeze(0))
        out = out.unsqueeze(0)
        return out
