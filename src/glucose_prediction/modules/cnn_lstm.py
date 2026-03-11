from typing import List

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """CNN-LSTM model for time-series glucose prediction.

    Architecture: 4 Conv1d layers -> 2-layer LSTM -> 3 FC layers.
    At inference, predicts auto-regressively: each predicted timestep is fed back
    as input to predict the next one.
    """

    def __init__(self, single_pred: bool = True, d_in: int = 3, input_length: int = 12) -> None:
        super().__init__()
        # single_pred=True -> only predict glucose (channel 0)
        # single_pred=False -> predict all d_in channels
        self.predict_channels: List[int] = [0] if single_pred else list(range(d_in))

        # Short input sequences (e.g. PH=30 → input_length=6) need padding on
        # the k=5 conv layers to avoid shrinking below kernel size.
        # Two unpadded k=5 convs reduce length by 8; sequences <= 8 need padding.
        k5_padding = 2 if input_length <= 8 else 0

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=d_in, out_channels=32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=k5_padding),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=k5_padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2)

        self.fc_layers = nn.Sequential(
            nn.Linear(100, 64), nn.Tanh(), nn.Linear(64, 6), nn.Tanh(), nn.Linear(6, len(self.predict_channels))
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step prediction from a history window.

        Args:
            x: (N, L, d_in) input sequence.

        Returns:
            (N, n_predict_channels) predicted values for the next timestep.
        """
        x = x.permute(0, 2, 1)  # (N, d_in, L) for Conv1d
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (N, L', 128) for LSTM
        x, _ = self.lstm(x)
        x = self.fc_layers(x[:, -1, :])  # take last timestep hidden state
        return x

    def forward(self, whole_example: torch.Tensor, input_len: int) -> torch.Tensor:
        """Auto-regressive prediction over the horizon.

        Takes the full (history + future) tensor, uses the first ``input_len``
        timesteps as context, and predicts one step at a time by feeding each
        prediction back into the model. Avoids in-place ops so autograd can
        track gradients through the full auto-regressive chain.

        Args:
            whole_example: (N, L_total, d_in) full sequence including future slots.
            input_len: number of history timesteps to use as initial context.

        Returns:
            (N, L_total, d_in) tensor with predicted channels filled in from
            position ``input_len`` onward.
        """
        total_len = whole_example.shape[1]
        assert input_len < total_len

        # Start with the history portion (no grad issues — just a slice)
        history = whole_example[:, :input_len, :]
        predicted_steps: List[torch.Tensor] = []

        for step in range(total_len - input_len):
            y_hat = self._forward(history)  # (N, n_predict_channels)

            # Build the next timestep: start from ground-truth non-predicted
            # channels, overwrite predicted channels with model output.
            gt_row = whole_example[:, input_len + step, :]  # (N, d_in)
            # Scatter predicted values into the row without in-place ops
            pred_row = gt_row.clone()
            idx = torch.tensor(self.predict_channels, device=y_hat.device)
            pred_row = pred_row.scatter(1, idx.unsqueeze(0).expand(y_hat.shape[0], -1), y_hat)

            predicted_steps.append(pred_row.unsqueeze(1))  # (N, 1, d_in)
            # Grow history with the new predicted row for the next iteration
            history = torch.cat([history, predicted_steps[-1]], dim=1)

        return torch.cat([whole_example[:, :input_len, :], torch.cat(predicted_steps, dim=1)], dim=1)
