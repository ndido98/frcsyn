from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable
import random

import cv2
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


class FaceCouplesDataset(FaceClassificationDataset):
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
        self._n_classes = len(classes)
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
        random.shuffle(couples)
        self.couples = SupervisedCoupleArray(couples)
    
    @property
    def n_classes(self) -> int:
        return self._n_classes

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


class JointFaceCouplesDataset(FaceClassificationDataset):
    def __init__(self, datasets: list[FaceCouplesDataset]) -> None:
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
        file1, file2, is_same = self.datasets[dataset_idx][sample_idx]
        return file1, file2, is_same


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


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        datasets_root: str,
        include_real_training: bool,
        include_synth_training: bool,
        val_n_classes: int = 200,
        max_matches_per_image: int = 3,
        max_nonmatches_per_image: int = 3,
        augment: bool = False,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.datasets_root = Path(datasets_root)
        self.include_real_training = include_real_training
        self.include_synth_training = include_synth_training
        self.val_n_classes = val_n_classes
        self.max_matches_per_image = max_matches_per_image
        self.max_nonmatches_per_image = max_nonmatches_per_image
        self.augment = augment
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        ethnicities = ["Asian", "Black", "Indian", "Other", "White"]
        genders = ["Female", "Male"]
        dcface_root = self.datasets_root / "Synth" / "DCFace" / "dcface_wacv" / "organized"
        dcface_dirs = [
            dcface_root / ethnicity / gender
            for ethnicity in ethnicities
            for gender in genders
        ]
        gandiffface_root = self.datasets_root / "Synth" / "GANDiffFace-processed"
        gandiffface_dirs = [
            gandiffface_root / f"{ethnicity}_{gender}"
            for ethnicity in ethnicities
            for gender in genders
        ]
        base_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=1.0),
            APT.ToTensorV2(),
        ], additional_targets={"image2": "image"})
        if self.augment:
            train_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(112, 112, scale=(0.2, 1.0), ratio=(0.75, 1.333), p=0.2),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0, p=0.2),
                A.Compose([
                    A.RandomScale(scale_limit=(0, 0.8), interpolation=cv2.INTER_CUBIC, p=1),
                    A.Resize(112, 112, interpolation=cv2.INTER_AREA, p=1),
                ], p=0.2),
                base_transform,
            ], additional_targets={"image2": "image"})
        else:
            train_transform = base_transform
        train_datasets = []
        casia_root = self.datasets_root / "Real" / "CASIA-WebFace" / "imgs"
        if self.include_real_training:
            train_datasets.append(SingleFaceClassificationDataset(casia_root, from_class=self.val_n_classes, transform=train_transform))
        if self.include_synth_training:
            for ts in dcface_dirs + gandiffface_dirs:
                train_datasets.append(SingleFaceClassificationDataset(ts, transform=train_transform, from_class=self.val_n_classes // 24))
        self.train_dataset = JointFaceClassificationDataset(train_datasets)
        if self.include_real_training:
            self.val_dataset = FaceCouplesDataset(
                casia_root,
                to_class=self.val_n_classes,
                max_matches_per_image=self.max_matches_per_image,
                max_nonmatches_per_image=self.max_nonmatches_per_image,
                transform=base_transform,
            )
        elif self.include_synth_training:
            val_datasets = []
            for ts in dcface_dirs + gandiffface_dirs:
                val_datasets.append(
                    FaceCouplesDataset(
                        ts,
                        to_class=self.val_n_classes // 24,
                        max_matches_per_image=self.max_matches_per_image,
                        max_nonmatches_per_image=self.max_nonmatches_per_image,
                        transform=base_transform,
                    )
                )
            self.val_dataset = JointFaceCouplesDataset(val_datasets)
        predict_sets = [
            (
                self.datasets_root / "comparison_files" / "sub-tasks_2.1_2.2" / "agedb_comparison.txt",
                self.datasets_root / "Real" / "AgeDB-processed" / "03_Protocol_Images",
            ),
            (
                self.datasets_root / "comparison_files" / "sub-tasks_2.1_2.2" / "bupt_comparison.txt",
                self.datasets_root / "Real" / "BUPT-BalancedFace-processed" / "race_per_7000",
            ),
            (
                self.datasets_root / "comparison_files" / "sub-tasks_2.1_2.2" / "cfp-fp_comparison.txt",
                self.datasets_root / "Real" / "CFP-FP-processed" / "cfp-dataset" / "Data" / "Images",
            ),
            (
                self.datasets_root / "comparison_files" / "sub-tasks_2.1_2.2" / "rof_comparison.txt",
                self.datasets_root / "Real" / "ROF-processed",
            ),
        ]
        self.predict_datasets = []
        for couples_file, root in predict_sets:
            self.predict_datasets.append(CouplesFileDataset(Path(root), Path(couples_file), transform=base_transform))
    
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