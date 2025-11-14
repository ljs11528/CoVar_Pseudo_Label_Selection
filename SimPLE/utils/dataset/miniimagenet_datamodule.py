from .miniimagenet import MiniImageNet
from .cifar10_datamodule import CIFAR10DataModule
from .utils import per_class_random_split_by_ratio, per_class_random_split
import numpy as np

# for type hint
from typing import Optional, Tuple


class MiniImageNetDataModule(CIFAR10DataModule):
    num_classes: int = 100

    total_train_size: int = 50_000
    total_test_size: int = 10_000

    DATASET = MiniImageNet

    def __init__(self,
                 data_dir: str,
                 labeled_train_size: int,
                 validation_size: int,
                 unlabeled_train_size: Optional[int] = None,
                 dims: Optional[Tuple[int, ...]] = None,
                 **kwargs):
        if dims is None:
            dims = (3, 84, 84)

        super(MiniImageNetDataModule, self).__init__(
            data_dir=data_dir,
            labeled_train_size=labeled_train_size,
            validation_size=validation_size,
            unlabeled_train_size=unlabeled_train_size,
            dims=dims,
            **kwargs)

        # dataset stats
        # Mini-ImageNet mean, std values in CHW
        self.dataset_mean = [0.40233998, 0.47269102, 0.44823737]
        self.dataset_std = [0.2884859, 0.28327602, 0.27511246]

    def setup(self, stage: Optional[str] = None):
        """
        Override setup to be robust when the actual number of training samples differs from
        the expected total (e.g., when using a local folder of images instead of an hdf5 file).
        This computes/adjusts `unlabeled_train_size` based on the actual dataset length and
        then delegates to the shared `setup_helper` implementation.
        """
        full_train_set = self.DATASET(root=self.data_dir, train=True)
        full_test_set = self.DATASET(root=self.data_dir, train=False)

        total_available = len(full_train_set)

        # Capture requested sizes
        req_v = int(self.validation_size)
        req_l = int(self.labeled_train_size)
        req_u = int(self.unlabeled_train_size) if (self.unlabeled_train_size is not None) else None

        # If unlabeled_train_size was not provided, compute it from available data
        if req_u is None:
            req_u = max(0, total_available - req_v - req_l)

        requested_total = req_v + req_l + req_u

        # If requested total differs from available, rescale requested splits proportionally so they sum to total_available
        if requested_total != total_available and requested_total > 0:
            scale = float(total_available) / float(requested_total)
            new_v = int(round(req_v * scale))
            new_l = int(round(req_l * scale))
            # ensure sum matches total_available
            new_u = max(0, total_available - new_v - new_l)
            print(f"Warning: miniimagenet dataset size ({total_available}) does not match requested total ({requested_total})."
                  f" Rescaling splits: validation {req_v}->{new_v}, labeled {req_l}->{new_l}, unlabeled {req_u}->{new_u}.")
            # assign back
            self.validation_size = new_v
            self.labeled_train_size = new_l
            self.unlabeled_train_size = new_u
        else:
            # requested_total matches available (or requested_total == 0)
            self.validation_size = req_v
            self.labeled_train_size = req_l
            self.unlabeled_train_size = req_u

        self.setup_helper(full_train_set=full_train_set, full_test_set=full_test_set, stage=stage)

    def split_dataset(self, dataset, **kwargs):
        """
        Override splitting to handle cases where requested subset lengths are not
        divisible evenly by the number of classes. Falls back to a ratio-based
        per-class split which preserves class balance approximately.
        """
        lengths = [self.validation_size, self.labeled_train_size, self.unlabeled_train_size]
        # ensure lengths are ints
        lengths = [int(l) for l in lengths]
        lengths_arr = np.array(lengths)
        # if all lengths divisible by num_classes, use the original exact split
        if np.all(lengths_arr % self.num_classes == 0):
            return per_class_random_split(dataset, lengths=lengths, num_classes=self.num_classes, **kwargs)

        # otherwise compute ratios and use ratio-based splitting (allows uneven class counts)
        total = lengths_arr.sum()
        if total <= 0:
            raise ValueError("Requested total split size must be > 0")
        ratios = (lengths_arr.astype(float) / float(total)).tolist()
        return per_class_random_split_by_ratio(dataset, ratios=ratios, num_classes=self.num_classes, uneven_split=False)
