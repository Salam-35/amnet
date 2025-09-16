"""
Medical Image Transforms for AMNet
Advanced augmentation and preprocessing for CT volumes
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple, Optional, Callable
import random
import logging

logger = logging.getLogger(__name__)


class MedicalTransforms:
    """Medical image transformation utilities"""

    @staticmethod
    def normalize_ct_intensity(image: np.ndarray,
                               window_level: float = 40.0,
                               window_width: float = 400.0) -> np.ndarray:
        """Apply CT windowing and normalization"""
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2

        # Clip values to window
        windowed = np.clip(image, min_val, max_val)

        # Normalize to [0, 1]
        normalized = (windowed - min_val) / (max_val - min_val)

        return normalized.astype(np.float32)

    @staticmethod
    def resample_volume(volume: np.ndarray,
                        target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
                        current_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                        order: int = 1) -> np.ndarray:
        """Resample volume to target spacing"""
        zoom_factors = [c / t for c, t in zip(current_spacing, target_spacing)]

        resampled = ndimage.zoom(volume, zoom_factors, order=order, mode='constant', cval=0)

        return resampled.astype(volume.dtype)

    @staticmethod
    def get_training_transforms(config) -> Callable:
        """Get training transforms with augmentation"""

        def transform_fn(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            image = sample['image']  # [D, H, W]
            mask = sample['mask']  # [D, H, W]

            # Convert to numpy for processing
            image_np = image.numpy()
            mask_np = mask.numpy()

            # Random rotation
            if random.random() < 0.6:
                angle = random.uniform(*config.augmentation.rotation_range)
                image_np = ndimage.rotate(image_np, angle, axes=(1, 2), reshape=False, order=1)
                mask_np = ndimage.rotate(mask_np, angle, axes=(1, 2), reshape=False, order=0)

            # Random scaling
            if random.random() < 0.4:
                scale = random.uniform(*config.augmentation.scaling_range)
                zoom_factors = [1.0, scale, scale]  # Only scale H, W
                image_np = ndimage.zoom(image_np, zoom_factors, order=1)
                mask_np = ndimage.zoom(mask_np, zoom_factors, order=0)

            # Random flipping
            if random.random() < 0.5:
                # Flip along width (left-right)
                image_np = np.flip(image_np, axis=2).copy()
                mask_np = np.flip(mask_np, axis=2).copy()

            # Gaussian noise
            if random.random() < 0.3:
                noise = np.random.normal(0, config.augmentation.noise_std, image_np.shape)
                image_np = image_np + noise.astype(np.float32)
                image_np = np.clip(image_np, 0, 1)

            # Intensity shifting
            if random.random() < 0.4:
                shift = random.uniform(-0.1, 0.1)
                image_np = np.clip(image_np + shift, 0, 1)

            # Contrast adjustment
            if random.random() < 0.4:
                contrast = random.uniform(0.8, 1.2)
                image_np = np.clip(image_np * contrast, 0, 1)

            # Ensure target size
            target_size = config.input_size
            image_np = resize_volume(image_np, target_size)
            mask_np = resize_volume(mask_np, target_size, order=0)

            # Convert back to tensors
            sample['image'] = torch.from_numpy(image_np).float().unsqueeze(0)  # Add channel dim
            sample['mask'] = torch.from_numpy(mask_np).long()

            return sample

        return transform_fn

    @staticmethod
    def get_validation_transforms() -> Callable:
        """Get validation transforms (no augmentation)"""

        def transform_fn(sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            image = sample['image'].numpy()
            mask = sample['mask'].numpy()

            # Only resize to target dimensions
            from scripts.config import Config
            config = Config()

            image = resize_volume(image, config.input_size)
            mask = resize_volume(mask, config.input_size, order=0)

            # Convert to tensors
            sample['image'] = torch.from_numpy(image).float().unsqueeze(0)
            sample['mask'] = torch.from_numpy(mask).long()

            return sample

        return transform_fn


def resize_volume(volume: np.ndarray,
                  target_size: Tuple[int, int, int],
                  order: int = 1) -> np.ndarray:
    """Resize volume to target size"""
    current_size = volume.shape
    zoom_factors = [t / c for t, c in zip(target_size, current_size)]

    resized = ndimage.zoom(volume, zoom_factors, order=order, mode='constant', cval=0)

    return resized


class ElasticTransform:
    """3D Elastic deformation for data augmentation"""

    def __init__(self, alpha: float = 1.0, sigma: float = 50.0, p: float = 0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation"""
        if random.random() > self.p:
            return image, mask

        shape = image.shape

        # Generate random displacement fields
        dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha
        dz = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), self.sigma) * self.alpha

        # Create coordinate grids
        z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

        # Apply displacement
        indices = [
            np.clip(z + dz, 0, shape[0] - 1),
            np.clip(y + dy, 0, shape[1] - 1),
            np.clip(x + dx, 0, shape[2] - 1)
        ]

        # Interpolate
        deformed_image = ndimage.map_coordinates(image, indices, order=1, mode='reflect')
        deformed_mask = ndimage.map_coordinates(mask, indices, order=0, mode='reflect')

        return deformed_image.astype(image.dtype), deformed_mask.astype(mask.dtype)


class GaussianBlur3D:
    """3D Gaussian blur augmentation"""

    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 2.0), p: float = 0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply 3D Gaussian blur"""
        if random.random() > self.p:
            return image

        sigma = random.uniform(*self.sigma_range)
        blurred = ndimage.gaussian_filter(image, sigma=sigma)

        return blurred.astype(image.dtype)


class RandomCrop3D:
    """3D random cropping with overlap handling"""

    def __init__(self, crop_size: Tuple[int, int, int], p: float = 0.5):
        self.crop_size = crop_size
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random 3D cropping"""
        if random.random() > self.p:
            return image, mask

        current_size = image.shape
        crop_d, crop_h, crop_w = self.crop_size

        # Random crop coordinates
        start_d = random.randint(0, max(0, current_size[0] - crop_d))
        start_h = random.randint(0, max(0, current_size[1] - crop_h))
        start_w = random.randint(0, max(0, current_size[2] - crop_w))

        # Extract crops
        cropped_image = image[
            start_d:start_d + crop_d,
            start_h:start_h + crop_h,
            start_w:start_w + crop_w
        ]

        cropped_mask = mask[
            start_d:start_d + crop_d,
            start_h:start_h + crop_h,
            start_w:start_w + crop_w
        ]

        return cropped_image, cropped_mask