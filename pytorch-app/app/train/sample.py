from typing import NamedTuple, Union, List
import torch


class Meta(NamedTuple):
    name: Union[str, List[str]]


class Source(NamedTuple):
    image: torch.Tensor  # [B, C, H, W] | [C, H, W]


class AugmentedSource(NamedTuple):
    image: torch.Tensor  # [B, C, H, W]


class Target(NamedTuple):
    label: torch.Tensor  # [B] | 1


class AugmentedTarget(NamedTuple):
    label: torch.Tensor  # [B]


class Sample(NamedTuple):
    """
    A training sample. Note that a sample may hold a single datapoint, but
    will more commonly be transformed into a mini-batch of data when used with
    torch.utils.data.Dataloader.
    """

    meta: Meta
    source: Source
    target: Target


class AugmentedSample(NamedTuple):
    meta: Meta
    source: AugmentedSource
    target: AugmentedTarget
