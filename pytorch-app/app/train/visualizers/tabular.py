from app.train import sample
from app.train.sample import AugmentedSample
from app.train.visualizers import simple
from typing import Mapping
import torch
import wandb


class Visualizer:
    def __init__(self, class_to_idx: Mapping[str, int], **kwargs):
        self.simple_visualizer = simple.Visualizer(class_to_idx=class_to_idx, **kwargs)
        self.classes = [
            k for k, _ in sorted(class_to_idx.items(), key=lambda item: item[1])
        ]

    def __call__(
        self,
        x: AugmentedSample,
        prediction: sample.AugmentedTarget,
    ) -> Mapping[str, wandb.Table]:
        images = self.simple_visualizer(x, prediction)

        def plot_scores(scores: torch.Tensor):
            return {
                class_name: score.item()
                for class_name, score in zip(self.classes, scores)
            }

        return {
            "samples": wandb.Table(
                data=[
                    [name, image, actual_label, plot_scores(predicted_scores)]
                    for (name, image), actual_label, predicted_scores in zip(
                        images.items(), x.target.label, prediction.label.detach()
                    )
                ],
                columns=["name", "image", "actual", "predicted"],
            ),
        }
