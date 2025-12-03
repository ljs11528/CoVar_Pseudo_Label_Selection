import h5py
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_file_from_google_drive

from pathlib import Path

# for type hint
from typing import Optional, Callable


class MiniImageNet(VisionDataset):
    base_folder = 'mini-imagenet'
    gdrive_id = '1EKmnUcpipszzBHBRcXxmejuO4pceD4ht'
    file_md5 = '3bda5120eb7353dd88e06de46e680146'
    filename = 'mini-imagenet.hdf5'

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.root = root
        # Support two modes:
        # 1) hdf5 mode (original): dataset stored in mini-imagenet.hdf5 under root/base_folder
        # 2) local folder mode: `root` points to a directory containing 100 class subfolders each with images
        # Keep existing download behavior unchanged.
        self.local_mode = False

        # determine available sources
        root_path = Path(self.root)
        possible_hdf5 = root_path / self.base_folder / self.filename
        possible_hdf5_alt = root_path / self.filename

        # Allow an explicit environment override that points to a local folder containing
        # class subfolders. This adds a new reading path without removing existing behavior.
        # Example: export MINIIMAGENET_LOCAL_PATH=/path/to/mini-imagenet
        env_local = None
        try:
            from os import environ
            env_local = environ.get('MINIIMAGENET_LOCAL_PATH', None)
        except Exception:
            env_local = None

    # Decide mode: prefer hdf5 if present. If not, check environment-local path first,
    # then check `root`, `root/base_folder` and also `root/data` / `root/data/base_folder`
    # for class subfolders. This allows passing the project root as `root` while the
    # actual dataset lives under `root/data/mini-imagenet`.
    # Only if nothing is found and download=True do we attempt network download.
        folder_candidate = None

        if possible_hdf5.exists() or possible_hdf5_alt.exists():
            # hdf5 mode (unchanged)
            hdf5_path = possible_hdf5 if possible_hdf5.exists() else possible_hdf5_alt
            img_key = 'train_img' if self.train else 'test_img'
            target_key = 'train_target' if self.train else 'test_target'
            with h5py.File(hdf5_path, "r", swmr=True) as h5_f:
                self.data = h5_f[img_key][...]
                self.target = h5_f[target_key][...]
            return

        # Check env-specified path
        if env_local is not None:
            env_path = Path(env_local)
            if env_path.is_dir():
                subdirs = [p for p in sorted(env_path.iterdir()) if p.is_dir()]
                if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                    folder_candidate = env_path

        # Check root itself
        if folder_candidate is None and root_path.is_dir():
            subdirs = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
            if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                folder_candidate = root_path

        # Check root/base_folder
        if folder_candidate is None:
            base_candidate = root_path / self.base_folder
            if base_candidate.is_dir():
                subdirs = [p for p in sorted(base_candidate.iterdir()) if p.is_dir()]
                if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                    folder_candidate = base_candidate

        # Also check root/data and root/data/base_folder (common project layout)
        if folder_candidate is None:
            data_root = root_path / 'data'
            if data_root.is_dir():
                # check data_root itself
                subdirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
                if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                    folder_candidate = data_root
                else:
                    data_base = data_root / self.base_folder
                    if data_base.is_dir():
                        subdirs = [p for p in sorted(data_base.iterdir()) if p.is_dir()]
                        if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                            folder_candidate = data_base

        # If still not found, try to download if requested
        if folder_candidate is None:
            if download:
                # download() will check integrity again and will not re-download if a local
                # valid copy exists under the computed download_root
                self.download()

                # After download attempt, re-run initialization: try to load hdf5 or local folders
                possible_hdf5 = root_path / self.base_folder / self.filename
                possible_hdf5_alt = root_path / self.filename
                if possible_hdf5.exists() or possible_hdf5_alt.exists():
                    hdf5_path = possible_hdf5 if possible_hdf5.exists() else possible_hdf5_alt
                    img_key = 'train_img' if self.train else 'test_img'
                    target_key = 'train_target' if self.train else 'test_target'
                    with h5py.File(hdf5_path, "r", swmr=True) as h5_f:
                        self.data = h5_f[img_key][...]
                        self.target = h5_f[target_key][...]
                    return
                # Check env_local again in case download changed structure
                if env_local is not None:
                    env_path = Path(env_local)
                    if env_path.is_dir():
                        subdirs = [p for p in sorted(env_path.iterdir()) if p.is_dir()]
                        if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                            folder_candidate = env_path

        if folder_candidate is None:
            raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it or point `root` at a directory of class folders (or set MINIIMAGENET_LOCAL_PATH)')

        # Build samples list from class folders
        self.local_mode = True
        self.classes = [p.name for p in sorted(folder_candidate.iterdir()) if p.is_dir()]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        samples = []
        # accept common image extensions
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        for cls_name in self.classes:
            cls_folder = folder_candidate / cls_name
            for img_path in sorted(cls_folder.iterdir()):
                if img_path.suffix.lower() in img_exts:
                    samples.append((str(img_path), self.class_to_idx[cls_name]))
        if len(samples) == 0:
            raise RuntimeError(f'No images found under {folder_candidate}')
        self.samples = samples

    @property
    def data_root(self) -> Path:
        # For hdf5 mode the file resides under root/base_folder; for local_mode use root or base folder depending
        root_path = Path(self.root)
        if self.local_mode:
            # samples are taken from root or root/base_folder depending on detection; return parent folder
            return root_path
        else:
            return root_path / self.base_folder

    @property
    def download_root(self) -> Path:
        return self.data_root

    def __len__(self):
        if self.local_mode:
            return len(self.samples)
        return len(self.target)

    def __getitem__(self, idx):
        if self.local_mode:
            img_path, target = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
        else:
            img, target = Image.fromarray(self.data[idx]), self.target[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_file_from_google_drive(file_id=self.gdrive_id,
                                        root=str(self.download_root),
                                        filename=self.filename,
                                        md5=self.file_md5)

    def _check_integrity(self):
        # consider hdf5 integrity or local folder existence
        root_path = Path(self.root)
        hdf5_path = root_path / self.base_folder / self.filename
        hdf5_path_alt = root_path / self.filename
        if hdf5_path.exists() or hdf5_path_alt.exists():
            return check_integrity(fpath=str(hdf5_path if hdf5_path.exists() else hdf5_path_alt), md5=self.file_md5)

        # if a local folder of class subfolders exists, consider it valid
        folder_candidate = None
        if root_path.is_dir():
            subdirs = [p for p in sorted(root_path.iterdir()) if p.is_dir()]
            if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                return True
        base_candidate = root_path / self.base_folder
        if base_candidate.is_dir():
            subdirs = [p for p in sorted(base_candidate.iterdir()) if p.is_dir()]
            if len(subdirs) > 0 and any(len(list(p.glob("*.*"))) > 0 for p in subdirs):
                return True

        return False
