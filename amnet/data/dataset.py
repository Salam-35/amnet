"""AMOS22 Dataset Implementation"""

import os
import json
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AMOSDataset(Dataset):
    """AMOS22 dataset for abdominal organ segmentation"""

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 transforms=None,
                 cache_data: bool = True):

        self.data_root = Path(data_root)
        self.split = split
        self.transforms = transforms
        self.cache_data = cache_data
        self.cache = {} if cache_data else None

        # AMOS22 organ labels
        self.organ_labels = {
            0: "background", 1: "spleen", 2: "right_kidney", 3: "left_kidney",
            4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
            8: "aorta", 9: "IVC", 10: "portal_vein", 11: "pancreas",
            12: "right_adrenal", 13: "left_adrenal", 14: "duodenum", 15: "bladder"
        }

        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {split}")

    def _load_samples(self) -> list:
        """Load dataset samples"""
        samples_file = self.data_root / f"{self.split}.json"

        if samples_file.exists():
            with open(samples_file, 'r') as f:
                return json.load(f)

        # Create splits if not exist
        return self._create_splits()

    def _create_splits(self) -> list:
        """Create train/val/test splits"""
        images_dir = self.data_root / "imagesTr"
        labels_dir = self.data_root / "labelsTr"

        all_samples = []
        for img_file in images_dir.glob("*.nii.gz"):
            label_file = labels_dir / img_file.name
            if label_file.exists():
                all_samples.append({
                    "image": str(img_file),
                    "label": str(label_file),
                    "case_id": img_file.stem.split('.')[0]
                })

        # Split data (70/15/15)
        np.random.seed(42)
        np.random.shuffle(all_samples)

        n = len(all_samples)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)

        splits = {
            "train": all_samples[:train_end],
            "val": all_samples[train_end:val_end],
            "test": all_samples[val_end:]
        }

        # Save splits
        for split_name, split_data in splits.items():
            with open(self.data_root / f"{split_name}.json", 'w') as f:
                json.dump(split_data, f, indent=2)

        return splits[self.split]

    def _load_nifti(self, filepath: str) -> np.ndarray:
        """Load NIfTI file with error handling"""
        try:
            img = nib.load(filepath)
            data = img.get_fdata().astype(np.float32)
            return data
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess CT image"""
        # CT windowing
        window_level, window_width = 40.0, 400.0
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2

        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)

        return image

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Preprocess segmentation mask"""
        # Ensure mask values are in valid range
        mask = np.clip(mask, 0, self.num_classes - 1)
        return mask.astype(np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.organ_labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with caching support"""

        if self.cache_data and idx in self.cache:
            return self.cache[idx]

        sample = self.samples[idx]

        # Load image and mask
        image = self._load_nifti(sample["image"])
        mask = self._load_nifti(sample["label"])

        # Preprocess
        image = self._preprocess_image(image)
        mask = self._preprocess_mask(mask)

        # Create sample dict
        sample_dict = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).long(),
            "case_id": sample["case_id"]
        }

        # Apply transforms
        if self.transforms:
            sample_dict = self.transforms(sample_dict)

        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample_dict

        return sample_dict