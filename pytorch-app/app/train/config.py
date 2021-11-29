from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Mapping
from common import DirPath, Epoch


Kwargs = Dict[str, Any]


@dataclass(frozen=True)
class AugmentationConfig:
    module: str = "default"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class DatasetConfig:
    module: str = "default"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class DataloaderConfig:
    batch_size: int = 2
    workers: int = 4
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class ModelConfig:
    module: str = "cnn"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class CriterionConfig:
    module: str = "default"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class OptimizerConfig:
    # module: str = "adam"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class SchedulerConfig:
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class VisualizerConfig:
    module: str = "default"
    kwargs: Kwargs = field(default_factory=lambda: {})


@dataclass(frozen=True)
class LoggingInterval:
    epoch: Optional[Epoch] = None
    batch_step: Optional[int] = None

    def __post_init__(self):
        # Enforce pre-conditions
        assert (
            self.epoch is None or self.epoch > 0
        ), f"invalid logging interval for epochs (got {self.epoch} ≯ 0)"
        assert (
            self.batch_step is None or self.batch_step > 0
        ), f"invalid logging interval for batch steps (got {self.batch_step} ≯ 0)"
        assert (
            self.epoch is not None or self.batch_step is not None
        ), f"logging interval must be set (got {self})"


@dataclass(frozen=True)
class LoggerConfig:
    visualize: bool = True
    visualize_interval: LoggingInterval = LoggingInterval(epoch=Epoch(10))
    kwargs: Kwargs = field(
        default_factory=lambda: {
            # See https://docs.wandb.ai/ref/python/init
            "project": "pytorch-app",
            # "entity": "username",
            "mode": "disabled",
        }
    )

    def __post_init__(self):
        # Convert dictionaries to dataclasses if any were supplied (helpful for JSON decoding)
        if isinstance(self.visualize_interval, Mapping):
            object.__setattr__(
                self, "visualize_interval", LoggingInterval(**self.visualize_interval)
            )


@dataclass(frozen=True)
class Config:
    cuda: bool = False
    save: bool = True
    save_dir: DirPath = DirPath("./experiments/default")
    resume_dir: Optional[DirPath] = None
    # log_dir: DirPath = DirPath("./logs/default")
    learning_rate: float = 5e-4  # TODO: learning rate scheduler
    start_epoch: Epoch = Epoch(0)
    stop_epoch: Epoch = Epoch(20)
    train_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(kwargs={"root": "./data/train"})
    )
    validation_dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(kwargs={"root": "./data/validation"})
    )
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    criterion: CriterionConfig = field(default_factory=CriterionConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)

    # Add paths to any additional source models below
    # pretrained_model: Optional[FilePath] = None
    # teacher_model: Optional[FilePath] = None

    def __post_init__(self):
        # Enforce pre-conditions
        assert (
            self.stop_epoch > self.start_epoch
        ), f"stop epoch must exceed start epoch (got {self.stop_epoch} ≯ {self.start_epoch})"

        # Convert dictionaries to dataclasses if any were supplied (helpful for JSON decoding)
        if isinstance(self.train_dataset, Mapping):
            object.__setattr__(
                self, "train_dataset", DatasetConfig(**self.train_dataset)
            )
        if isinstance(self.validation_dataset, Mapping):
            object.__setattr__(
                self, "validation_dataset", DatasetConfig(**self.validation_dataset)
            )
        if isinstance(self.augmentation, Mapping):
            object.__setattr__(
                self, "augmentation", AugmentationConfig(**self.augmentation)
            )
        if isinstance(self.dataloader, Mapping):
            object.__setattr__(self, "dataloader", DataloaderConfig(**self.dataloader))
        if isinstance(self.model, Mapping):
            object.__setattr__(self, "model", ModelConfig(**self.model))
        if isinstance(self.optimizer, Mapping):
            object.__setattr__(self, "optimizer", OptimizerConfig(**self.optimizer))
        if isinstance(self.scheduler, Mapping):
            object.__setattr__(self, "scheduler", SchedulerConfig(**self.scheduler))
        if isinstance(self.criterion, Mapping):
            object.__setattr__(self, "criterion", CriterionConfig(**self.criterion))
        if isinstance(self.visualizer, Mapping):
            object.__setattr__(self, "visualizer", VisualizerConfig(**self.visualizer))
        if isinstance(self.logger, Mapping):
            object.__setattr__(self, "logger", LoggerConfig(**self.logger))
