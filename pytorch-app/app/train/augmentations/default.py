from app.train.sample import Sample, AugmentedSample
from torch import nn
import app.train.sample as sample
import kornia


class Augmentation(nn.Module):
    augmentation: kornia.augmentation.AugmentationSequential

    def __init__(self, image_size: int = 128, **kwargs):
        if kwargs != {}:
            raise Exception("Augmentation: kwargs is unsupported at this time")
        super().__init__()
        self.augmentation = kornia.augmentation.AugmentationSequential(
            kornia.geometry.transform.Resize(size=image_size),
            data_keys=["input"],
        )

    def forward(self, x: Sample) -> AugmentedSample:
        # Random augmentation
        augmented_image = self.augmentation(x.source.image)

        return AugmentedSample(
            meta=x.meta,
            source=sample.AugmentedSource(image=augmented_image),
            target=sample.AugmentedTarget(label=x.target.label),
        )
