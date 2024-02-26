import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, single_pred=True, d_in=3):
        super().__init__()
        # kernel size 7
        if single_pred:
            predict_channels = [0]
        else:
            predict_channels = list(range(d_in))

        self.predict_channels = predict_channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            # nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            # nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            # nn.BatchNorm1d(128),
            # nn.Flatten()
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2)

        self.fc_layers = nn.Sequential(
            nn.Linear(100, 64), nn.Tanh(), nn.Linear(64, 6), nn.Tanh(), nn.Linear(6, len(predict_channels))
        )

    def _forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc_layers(x[:, -1, :])
        return x

    def forward(self, whole_example, input_len):
        """
        Args:
            whole_example: (N, l, d_in)
            input_len: int
        Returns:
            (N, l, d_in) where self.predict_channels on position [input_len: ] has been changed by the prediction
        """
        whole_example_clone = whole_example.clone().detach()
        total_len = whole_example_clone.shape[1]
        assert input_len < total_len

        while True:
            if input_len == total_len:
                return whole_example_clone
            x = whole_example[:, :input_len, :]
            y_hat = self._forward(x)
            whole_example_clone[:, input_len, self.predict_channels] = y_hat[:, self.predict_channels]
            input_len += 1
