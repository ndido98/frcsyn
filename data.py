from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import random

import cv2
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import albumentations as A
import albumentations.pytorch as APT

from containers import SupervisedArray, SupervisedCoupleArray


def _load_image(file: Path) -> np.ndarray:
    img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not load image {file}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class FaceClassificationDataset(ABC, Dataset):
    @property
    @abstractmethod
    def n_classes(self) -> int:
        pass


class SingleFaceClassificationDataset(FaceClassificationDataset):
    def __init__(
        self,
        root_dir: Path,
        from_class: int | None = None,
        to_class: int | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        all_classes = sorted([int(c.name) for c in root_dir.iterdir() if c.is_dir()])
        if from_class is None and to_class is None:
            classes = all_classes
        elif from_class is None:
            classes = all_classes[:to_class]
        elif to_class is None:
            classes = all_classes[from_class:]
        else:
            classes = all_classes[from_class:to_class]
        self._n_classes = len(classes)
        files = []
        for i, klass in enumerate(classes):
            class_dir = root_dir / str(klass)
            files.extend(
                [
                    (str((class_dir / f).resolve()), i)
                    for f in class_dir.iterdir()
                    if f.suffix in (".png", ".jpg")
                ]
            )
        self.files = SupervisedArray(files)

    @property
    def n_classes(self) -> int:
        return self._n_classes
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        file, klass = self.files[idx]
        img = _load_image(file)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, klass


class JointFaceClassificationDataset(FaceClassificationDataset):
    def __init__(self, datasets: list[FaceClassificationDataset]) -> None:
        self.datasets = datasets
        self.datasets_lengths = [len(d) for d in self.datasets]

    @property
    def n_classes(self) -> int:
        return sum(d.n_classes for d in self.datasets)

    def __len__(self) -> int:
        return sum(self.datasets_lengths)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        dataset_idx = np.searchsorted(np.cumsum(self.datasets_lengths), idx, side="right")
        sample_idx = idx - sum(self.datasets_lengths[:dataset_idx])
        file, original_class = self.datasets[dataset_idx][sample_idx]
        shifted_class = sum(d.n_classes for d in self.datasets[:dataset_idx]) + original_class
        return file, shifted_class


class FaceCouplesDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        from_class: int | None = None,
        to_class: int | None = None,
        max_matches_per_image: int = 3,
        max_nonmatches_per_image: int = 3,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.transform = transform
        all_classes = sorted([int(c.name) for c in root_dir.iterdir() if c.is_dir()])
        if from_class is None and to_class is None:
            classes = all_classes
        elif from_class is None:
            classes = all_classes[:to_class]
        elif to_class is None:
            classes = all_classes[from_class:]
        else:
            classes = all_classes[from_class:to_class]
        files: dict[str, list[Path]] = {}
        for klass in classes:
            class_dir = root_dir / str(klass)
            files[klass] = [
                ((class_dir / f).resolve(), klass)
                for f in class_dir.iterdir()
                if f.suffix in (".png", ".jpg")
            ]
        matching_couples, non_matching_couples = set(), set()
        for klass in classes:
            # Add the matching samples
            class_files = files[klass]
            for file1, _ in class_files:
                matches = 0
                while matches < max_matches_per_image:
                    file2 = random.choice(class_files)[0]
                    couple = ((str(file1), str(file2)), 1)
                    if couple not in matching_couples:
                        matching_couples.add(couple)
                        matches += 1
            # Add the non-matching samples
            other_classes = [c for c in classes if c != klass]
            for file1, _ in class_files:
                non_matches = 0
                while non_matches < max_nonmatches_per_image:
                    chosen_class = random.choice(other_classes)
                    file2 = random.choice(files[chosen_class])[0]
                    couple = ((str(file1), str(file2)), 0)
                    if couple not in non_matching_couples:
                        non_matching_couples.add(couple)
                        non_matches += 1
        couples = list(matching_couples) + list(non_matching_couples)
        self.couples = SupervisedCoupleArray(couples)

    def __len__(self) -> int:
        return len(self.couples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        (file1, file2), is_same = self.couples[idx]
        img1 = _load_image(file1)
        img2 = _load_image(file2)
        if self.transform:
            result = self.transform(image=img1, image2=img2)
            img1, img2 = result["image"], result["image2"]
        return img1, img2, is_same


class CouplesFileDataset(Dataset):
    def __init__(self, root_dir: Path, couples_file: Path, transform: Callable | None = None) -> None:
        super().__init__()
        self.root_dir = root_dir
        file_lines = couples_file.read_text().strip().splitlines()
        file_couples = [line.split(";") for line in file_lines]
        self.couples = [(Path(c1), Path(c2)) for c1, c2 in file_couples]
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.couples)
    
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img1, img2 = self.couples[idx]
        img1 = _load_image(self.root_dir / img1)
        img2 = _load_image(self.root_dir / img2)
        if self.transform:
            result = self.transform(image=img1, image2=img2)
            img1, img2 = result["image"], result["image2"]
        return img1, img2


@dataclass
class PredictDataset:
    root: str
    couples_file: str


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        casia_root: Path,
        extra_training_sets: list[Path] | None = None,
        predict_sets: list[PredictDataset] | None = None,
        casia_train: bool = True,
        casia_val_n_classes: int = 200,
        max_matches_per_image: int = 3,
        max_nonmatches_per_image: int = 3,
        augment: bool = False,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.casia_root = casia_root
        self.extra_training_sets = extra_training_sets if extra_training_sets is not None else []
        self.predict_sets = predict_sets if predict_sets is not None else []
        self.casia_train = casia_train
        self.casia_val_n_classes = casia_val_n_classes
        self.max_matches_per_image = max_matches_per_image
        self.max_nonmatches_per_image = max_nonmatches_per_image
        self.augment = augment
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        base_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
            APT.ToTensorV2(),
        ], additional_targets={"image2": "image"})
        if self.augment:
            train_transform = A.Compose([
                A.HorizontalFlip(p=0.2),
                A.RandomResizedCrop(112, 112, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=0.2),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0, p=0.2),
                base_transform,
            ], additional_targets={"image2": "image"})
        else:
            train_transform = base_transform
        train_datasets = []
        if self.casia_train:
            train_datasets.append(SingleFaceClassificationDataset(self.casia_root, from_class=self.casia_val_n_classes, transform=train_transform))
        for ts in self.extra_training_sets:
            train_datasets.append(SingleFaceClassificationDataset(ts, transform=train_transform))
        self.train_dataset = JointFaceClassificationDataset(train_datasets)
        self.val_dataset = FaceCouplesDataset(
            self.casia_root,
            to_class=self.casia_val_n_classes,
            max_matches_per_image=self.max_matches_per_image,
            max_nonmatches_per_image=self.max_nonmatches_per_image,
            transform=base_transform,
        )
        self.predict_datasets = []
        for ps in self.predict_sets:
            self.predict_datasets.append(CouplesFileDataset(Path(ps.root), Path(ps.couples_file), transform=base_transform))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    def predict_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            for ds in self.predict_datasets
        ]