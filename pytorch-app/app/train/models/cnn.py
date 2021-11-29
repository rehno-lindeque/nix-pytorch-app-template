from functools import reduce
from dataclasses import dataclass, field
from enum import Enum
import sample
import torch
from torch import nn
from typing import Optional, Mapping, AbstractSet, Union, Tuple, TypeVar, Sequence

T = TypeVar("T")


def product(xs: Sequence[T]) -> T:
    def mul(result, x):
        return result * x

    return reduce(mul, xs)


class Stage(Enum):
    input = "input"
    resampled = "resampled"
    output = "output"


@dataclass(frozen=True)
class LayerParams:
    channels: int
    scale_exponent: int = 0
    # norm_groups : int


@dataclass(frozen=True)
class Hyperparams:
    class_to_idx: Mapping[str, int] = field(
        default_factory=lambda: {
            "background": 0,
            "ignore": 1,  # / 'empty' / 'missing'
            "a": 2,
            "b": 3,
            "c": 4,
        }
    )
    in_channels: int = 3
    batch_size: int = 8
    tile_size: int = 128
    stage_depth: Mapping[Stage, int] = field(
        default_factory=lambda: {Stage.resampled: 1}
    )

    @property
    def class_keys(self) -> AbstractSet[str]:
        return self.class_to_idx.keys()

    @property
    def class_channels(self) -> int:
        return len(self.class_to_idx)

    def layer_at(self, stage: Stage, stage_index: Optional[int] = None) -> LayerParams:
        if stage_index is not None and stage_index < 0:
            stage_index = self.stage_depth[stage] + stage_index

        if stage == Stage.input:
            return LayerParams(channels=self.in_channels)
        elif stage == Stage.resampled:
            return LayerParams(channels=self.in_channels ** 2, scale_exponent=-1)
        elif stage == Stage.output:
            return LayerParams(channels=len(self.class_to_idx), scale_exponent=-2)
        else:
            assert False, f"Unknown stage {stage}"

    def shape_at(self, stage: Stage, stage_index: Optional[int] = None) -> torch.Size:
        layer = self.layer_at(stage)
        return torch.Size(
            (
                self.batch_size,
                layer.channels,
                int(self.tile_size * 2 ** layer.scale_exponent),
                int(self.tile_size * 2 ** layer.scale_exponent),
            )
        )

    def __getitem__(self, key: Union[Stage, Tuple[Stage, int]]) -> LayerParams:
        return self.layer_at(*key) if isinstance(key, tuple) else self.layer_at(key)


class Classify(nn.Module):
    def __init__(self, in_shape: torch.Size, out_channels):
        super().__init__()
        self.classify = nn.Linear(
            in_features=product(in_shape[1:]),
            out_features=out_channels,
        )

    def forward(self, input):
        output = self.classify(input.view(input.shape[0], -1))
        return output


class Resample(nn.Conv2d):
    def __init__(self, in_layer: LayerParams, out_layer: LayerParams, **kwargs):
        scale_exponent = out_layer.scale_exponent - in_layer.scale_exponent
        assert (
            scale_exponent == -1
        ), f"Only supporting 2 ** -1 subsampling for now (got 2 ** {scale_exponent})"
        super().__init__(
            in_channels=in_layer.channels,
            out_channels=out_layer.channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **kwargs,
        )


class Model(nn.Module):
    resample: nn.Module
    classify: nn.Module

    def __init__(self, **kwargs):
        self.params = Hyperparams(**kwargs)
        params = self.params
        super().__init__()
        self.resample = nn.Sequential(
            Resample(
                params[Stage.input],
                params[Stage.resampled, 0],
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(params[Stage.resampled, 0].channels),
        )

        self.classify = Classify(
            in_shape=params.shape_at(Stage.resampled, -1),
            out_channels=params[Stage.output].channels,
        )

    def forward(self, x: sample.AugmentedSource) -> sample.AugmentedTarget:
        resampled = self.resample(x.image)
        return sample.AugmentedTarget(label=self.classify(resampled))
