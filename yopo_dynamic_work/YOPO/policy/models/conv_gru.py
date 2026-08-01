import torch
from torch import nn


class ConvGRUCell(nn.Module):
    """A compact convolutional GRU cell for low-resolution feature maps."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvGRU kernel_size must be odd")
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        combined_channels = input_channels + hidden_channels
        self.gates = nn.Conv2d(
            combined_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            combined_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden is None:
            hidden = features.new_zeros(
                features.shape[0],
                self.hidden_channels,
                features.shape[2],
                features.shape[3],
            )
        gate_input = torch.cat((features, hidden), dim=1)
        update, reset = torch.sigmoid(self.gates(gate_input)).chunk(2, dim=1)
        candidate_input = torch.cat((features, reset * hidden), dim=1)
        candidate = torch.tanh(self.candidate(candidate_input))
        return (1.0 - update) * hidden + update * candidate


class ConvGRU(nn.Module):
    """Run one ConvGRU cell over a [B,T,C,H,W] feature sequence."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.cell = ConvGRUCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError("ConvGRU expects [batch,time,channels,height,width]")
        hidden = None
        for time_index in range(sequence.shape[1]):
            hidden = self.cell(sequence[:, time_index], hidden)
        return hidden
