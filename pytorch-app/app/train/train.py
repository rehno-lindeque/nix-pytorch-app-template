#!/usr/bin/env python
import argparse
from config import Config
import criterions
from dataclasses import dataclass
import dataclasses
import datasets
import json
import models
import sys
import torch
from state import State
import state
from torch import nn
import torchvision
import augmentations
from tqdm import tqdm
from common import Epoch
import visualizers
import wandb
from typing import Callable, Dict, Optional
import utils.dataclasses


@dataclass
class Measurements:
    losses_sum: Dict[str, torch.Tensor]
    losses_avg: Dict[str, torch.Tensor]


def train(
    config: Config,
    device: None,
    epoch: Epoch,
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    augmentation: nn.Module,
    visualizer: Callable,
    criterion: nn.Module,
    optimizer: nn.Module,
    scheduler: nn.Module,
    class_to_idx: Dict[str, int],
) -> Measurements:
    # Setup
    model.train()

    batch_steps = len(dataloader)
    measurements = Measurements(
        losses_sum=dataclasses.asdict(criterion.Losses()),
        losses_avg=dataclasses.asdict(criterion.Losses()),
    )

    # Mini-batch passes
    for batch_subindex, x in enumerate(augmentation(x) for x in tqdm(dataloader)):
        step = batch_subindex + epoch * batch_steps

        # Forward pass
        prediction = model(x.source)

        # Calculate losses
        losses = criterion(prediction, actual=x.target)
        loss = losses.total_loss

        # Record metrics
        losses_dict = {k: v.detach().sum() for k, v in utils.dataclasses.Items(losses)}
        del losses
        for k, v in losses_dict.items():
            measurements.losses_sum[k] += v
        wandb.log(
            {f"train/losses/{k}": v for k, v in losses_dict.items()},
            step=step,
        )

        # Log optimizer learning rate
        wandb.log({"train/learning_rate": scheduler.get_last_lr()[0]}, step=step)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adjust learning rate
        scheduler.step()

        # Visualize sample and prediction
        if (
            config.logger.visualize
            and (
                config.logger.visualize_interval.epoch is None
                or epoch % config.logger.visualize_interval.epoch == 0
            )
            and (
                config.logger.visualize_interval.batch_step is None
                or batch_subindex % config.logger.visualize_interval.batch_step == 0
            )
            and (batch_subindex != 0 or epoch != 0)  # Always skip the very first step
        ):
            with torch.no_grad():
                wandb.log(
                    {
                        f"train/visualizations/{k}": v
                        for k, v in visualizer(x, prediction=prediction).items()
                    },
                    step=step,
                )

    for k, v in measurements.losses_sum.items():
        measurements.losses_avg[k] = v / ((batch_subindex + 1) * dataloader.batch_size)

    return measurements


def validate(criterion: nn.Module) -> Measurements:
    measurements = Measurements(
        losses_sum=dataclasses.asdict(criterion.Losses()),
        losses_avg=dataclasses.asdict(criterion.Losses()),
    )
    # TODO
    return measurements


def main() -> int:
    # Command-line options
    parser = argparse.ArgumentParser(description="Minimal python app")
    parser.add_argument(
        "--config", type=str, default=None, help="path to a config.json file"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="path to save checkpoints"
    )
    commandline_args = parser.parse_args()

    if commandline_args.config is None:
        extra_config = {}
    else:
        with open(commandline_args.config) as file:
            extra_config = json.load(file)

    # Load base configuration from python
    config = Config(
        # Override configuration from a json file specified on the command-line
        **extra_config,
        # Override configuration from command-line arguments
        **{
            k: v
            for k, v in vars(commandline_args).items()
            if k != "config" and v is not None
        },
    )

    # Intialize logger
    config_dict = dataclasses.asdict(config)
    wandb.init(config=config_dict, **config.logger.kwargs)
    if dict(wandb.config) != config_dict:
        raise Exception(
            "Wandb produced an altered config. Sweeps should be generated via a nix derivation so that nix inputs remain determistic."
        )

    # Intialize torch
    device = torch.device("cuda:0" if config.cuda else "cpu")

    # Initialize datasets
    train_dataset = getattr(datasets, config.train_dataset.module).Dataset(
        transform=torchvision.transforms.ToTensor(),
        **config.train_dataset.kwargs,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.dataloader.workers,
        pin_memory=True if config.cuda else False,
        **config.dataloader.kwargs,
    )
    validation_dataset = getattr(datasets, config.validation_dataset.module).Dataset(
        transform=torchvision.transforms.ToTensor(),
        **config.validation_dataset.kwargs,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.dataloader.workers,
        pin_memory=True if config.cuda else False,
        **config.dataloader.kwargs,
    )

    # Define the model
    model = (
        getattr(models, config.model.module)
        .Model(class_to_idx=train_dataset.class_to_idx, **config.model.kwargs)
        .to(device)
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, **config.optimizer.kwargs
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=(lambda _: 1.0), **config.scheduler.kwargs
    )

    # TODO: Resume previous training from serialized state
    # if config.resume_dir is not None:
    #     resume_state = state.load(config.resume_dir)
    #     model.load_state_dict(state.checkpoint.model)
    #     optimizer.load_state_dict(state.checkpoint.optimizer)

    # Define the augmentation pipeline
    augmentation = getattr(augmentations, config.augmentation.module).Augmentation(
        **config.augmentation.kwargs
    )

    # Criterion
    criterion = getattr(criterions, config.criterion.module).Criterion(
        **config.criterion.kwargs
    )

    # Watch model, criterion for logging etc
    wandb.watch(model, criterion=criterion)

    # Visualizer
    visualizer = getattr(visualizers, config.visualizer.module).Visualizer(
        class_to_idx=train_dataset.class_to_idx, **config.visualizer.kwargs
    )

    best_train_losses: Dict[str, torch.Tensor] = dataclasses.asdict(criterion.Losses())
    best_train_epochs: Dict[str, Optional[Epoch]] = {
        k: None for k, v in best_train_losses.items()
    }
    best_val_losses: Dict[str, torch.Tensor] = dataclasses.asdict(criterion.Losses())
    best_val_epochs: Dict[str, Optional[Epoch]] = {
        k: None for k, v in best_val_losses.items()
    }

    for epoch in (
        Epoch(epoch) for epoch in range(config.start_epoch, config.stop_epoch)
    ):
        # Train
        training_measurements = train(
            config=config,
            device=device,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            augmentation=augmentation,
            visualizer=visualizer,
            class_to_idx=train_dataset.class_to_idx,
        )

        # Record best training metrics (see https://docs.wandb.ai/guides/track/log#common-workflows)
        # TODO: A good practice is to record best accuracy instead of (or along with) loss
        for k, v in training_measurements.losses_avg.items():
            if (
                best_train_epochs[k] is None
                or training_measurements.losses_avg[k] < best_train_losses[k]
            ):
                best_train_epochs[k] = epoch
                best_train_losses[k] = training_measurements.losses_avg[k]
                wandb.run.summary[f"train/best/{k}/epoch"] = best_train_epochs[k]
                wandb.run.summary[f"train/best/{k}/avg"] = best_train_losses[k]

        # Validate
        validation_measurements = validate(
            criterion=criterion,
        )

        # Record best validation metrics
        for k, v in validation_measurements.losses_avg.items():
            if (
                best_val_epochs[k] is None
                or validation_measurements.losses_avg[k] < best_val_losses[k]
            ):
                best_val_epochs[k] = epoch
                best_val_losses[k] = validation_measurements.losses_avg[k]
                wandb.run.summary[f"val/best/{k}/epoch"] = best_val_epochs[k]
                wandb.run.summary[f"val/best/{k}/avg"] = best_val_losses[k]

        # Save state
        state.save(
            State(
                checkpoint=state.Checkpoint(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                epoch=epoch,
                measurements=dict(
                    training=dataclasses.asdict(training_measurements),
                    validation=dataclasses.asdict(validation_measurements),
                ),
            ),
            root_path=config.save_dir,
        )

    wandb.log({}, commit=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
