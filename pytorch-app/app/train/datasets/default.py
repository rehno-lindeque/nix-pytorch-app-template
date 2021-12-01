from app.train.sample import Sample
from typing import Mapping
import app.train.sample as sample
import os
import torch.utils.data
import torchvision.datasets


class Dataset(torch.utils.data.Dataset[Sample]):
    class ImageFolder(torchvision.datasets.ImageFolder):
        # Introduce additional class labels if required
        # def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        #     classes, _ = super().find_classes(directory)
        #     classes = ["background", "ignore", *sorted(set(classes) - {"background", "ignore"})]
        #     class_to_idx = {k: i for i, k in enumerate(classes)}
        #     return classes, class_to_idx
        pass

    image_folder: ImageFolder

    def __init__(self, **kwargs):
        self.image_folder = Dataset.ImageFolder(**kwargs)

    @property
    def class_to_idx(self) -> Mapping[str, int]:
        return self.image_folder.class_to_idx  # type: ignore

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, index) -> Sample:
        image, label = self.image_folder[index]
        image_path, class_index = self.image_folder.imgs[index]
        return Sample(
            meta=sample.Meta(name=os.path.splitext(os.path.basename(image_path))[0]),
            source=sample.Source(image=image),
            target=sample.Target(label=label),
        )
