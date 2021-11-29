import sample
import torch
from torch import nn
from dataclasses import dataclass, field


class Criterion(nn.Module):
    @dataclass
    class Losses:
        total_loss: torch.Tensor = field(
            default_factory=lambda: torch.tensor(0.0)  # type:ignore[no-any-return]
        )
        cross_entropy_loss: torch.Tensor = field(
            default_factory=lambda: torch.tensor(0.0)  # type: ignore[no-any-return]
        )

    def __init__(self, cross_entropy_loss_coeff: float = 1.0, **kwargs):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.cross_entropy_loss_coeff = cross_entropy_loss_coeff

    def forward(
        self, prediction: sample.AugmentedTarget, actual: sample.AugmentedTarget
    ) -> Losses:
        cross_entropy_loss = self.cross_entropy_loss(
            input=prediction.label, target=actual.label
        )
        return Criterion.Losses(
            total_loss=cross_entropy_loss * self.cross_entropy_loss_coeff,
            cross_entropy_loss=cross_entropy_loss,
        )
