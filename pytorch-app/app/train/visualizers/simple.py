import torchvision.transforms.functional
import wandb
from app.train.sample import AugmentedSample
from typing import Mapping, Any


class Visualizer:
    def __init__(self, image_size: int = 128, **kwargs):
        self.image_size = image_size

    def __call__(self, x: AugmentedSample, prediction: Any) -> Mapping[str, wandb.Image]:
        # Rescale images
        rescaled_image = torchvision.transforms.functional.resize(
            x.source.image,
            size=self.image_size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
        )

        if isinstance(x.meta.name, str):
            return {x.meta.name: wandb.Image(rescaled_image, caption=x.meta.name)}
        else:
            return {
                name: wandb.Image(image, caption=name)
                for name, image in zip(x.meta.name, rescaled_image)
            }
