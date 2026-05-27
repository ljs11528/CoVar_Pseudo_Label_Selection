from abc import ABC
from pathlib import Path

from .datamodule import DataModule
from .utils import get_batch

# for type hint
from typing import Optional, Callable, Tuple, List, Union
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder

from .datasets import LabeledDataset
from .utils import BatchGeneratorType

DatasetType = Union[LabeledDataset, Dataset]
DataLoaderType = Union[DataLoader, List[DataLoader]]


class SSLDataModule(DataModule, ABC):
    num_classes: int = 1

    total_train_size: int = 1
    total_test_size: int = 1

    DATASET = type(Dataset)

    def __init__(self,
                 train_batch_size: int,
                 unlabeled_batch_size: int,
                 test_batch_size: int,
                 num_workers: int,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 train_min_size: int = 0,
                 unlabeled_train_min_size: int = 0,
                 test_min_size: int = 0,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 dims: Optional[Tuple[int, ...]] = None):
        super(SSLDataModule, self).__init__(train_transform=train_transform,
                                            val_transform=val_transform,
                                            test_transform=test_transform,
                                            dims=dims)
        self.train_batch_size = train_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.prefetch_factor = int(prefetch_factor)

        self.train_min_size = max(train_min_size, 0)
        self.unlabeled_train_min_size = max(unlabeled_train_min_size, 0)
        self.test_min_size = max(test_min_size, 0)

        self._labeled_train_set: Optional[DatasetType] = None
        self._unlabeled_train_set: Optional[DatasetType] = None
        self._validation_set: Optional[DatasetType] = None
        self._test_set: Optional[DatasetType] = None

        # dataset stats
        self.dataset_mean: Optional[List[float]] = None
        self.dataset_std: Optional[List[float]] = None

    def _loader_kwargs(self, **kwargs):
        loader_kwargs = {
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.persistent_workers
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        loader_kwargs.update(kwargs)

        if self.num_workers <= 0:
            # PyTorch only accepts these options when worker processes are enabled.
            loader_kwargs.pop("persistent_workers", None)
            loader_kwargs.pop("prefetch_factor", None)

        return loader_kwargs

    @property
    def labeled_train_set(self) -> Optional[DatasetType]:
        return self._labeled_train_set

    @labeled_train_set.setter
    def labeled_train_set(self, dataset: Optional[DatasetType]) -> None:
        self._labeled_train_set = dataset

    @property
    def unlabeled_train_set(self) -> Optional[DatasetType]:
        return self._unlabeled_train_set

    @unlabeled_train_set.setter
    def unlabeled_train_set(self, dataset: Optional[DatasetType]) -> None:
        self._unlabeled_train_set = dataset

    @property
    def validation_set(self) -> Optional[DatasetType]:
        return self._validation_set

    @validation_set.setter
    def validation_set(self, dataset: Optional[DatasetType]) -> None:
        self._validation_set = dataset

    @property
    def test_set(self) -> Optional[DatasetType]:
        return self._test_set

    @test_set.setter
    def test_set(self, dataset: Optional[DatasetType]) -> None:
        self._test_set = dataset

    def train_dataloader(self, **kwargs) -> DataLoaderType:
        return [DataLoader(self.labeled_train_set,
                           shuffle=True,
                           batch_size=self.train_batch_size,
                           num_workers=self.num_workers,
                           drop_last=True,
                           **self._loader_kwargs(**kwargs)),
                DataLoader(self.unlabeled_train_set,
                           shuffle=True,
                           batch_size=self.unlabeled_batch_size,
                           num_workers=self.num_workers,
                           drop_last=True,
                           **self._loader_kwargs(**kwargs))]

    def val_dataloader(self, **kwargs) -> Optional[DataLoaderType]:
        return DataLoader(self.validation_set,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **self._loader_kwargs(**kwargs)) if self.validation_set is not None else None

    def test_dataloader(self, **kwargs) -> Optional[DataLoaderType]:
        return DataLoader(self.test_set,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **self._loader_kwargs(**kwargs)) if self.test_set is not None else None

    def get_train_batch(self, train_loaders: List[DataLoader], **kwargs) -> BatchGeneratorType:
        return get_batch(train_loaders, **kwargs)

    def save_split_info(self, output_path: Union[str, Path]) -> None:
        for subset, filename in [(self.labeled_train_set, "labeled_train_set.txt"),
                                 (self.unlabeled_train_set, "unlabeled_train_set.txt"),
                                 (self.validation_set, "validation_set.txt")]:
            if hasattr(subset, "dataset") and isinstance(subset.dataset, Subset):
                # save split info if subset is manually split
                full_set = subset.dataset.dataset
                indices = subset.dataset.indices

                if isinstance(full_set, ImageFolder):
                    # save file paths
                    split_info = [str(Path(full_set.imgs[i][0]).relative_to(full_set.root)) + "\n" for i in indices]
                else:
                    # save index values
                    split_info = [f"{i}\n" for i in indices]

                with open(Path(output_path) / filename, "w") as f:
                    f.writelines(split_info)
