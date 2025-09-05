import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CTPreprocessor:
    """Comprehensive CT image preprocessing"""

    def __init__(self,
                 target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
                 target_size: Optional[Tuple[int, int, int]] = None,
                 window_level: float = 40.0,
                 window_width: float = 400.0):

        self.target_spacing = target_spacing
        self.target_size = target_size
        self.window_level = window_level
        self.window_width = window_width

    def resample_sitk(self,
                      image: sitk.Image,
                      target_spacing: Optional[Tuple[float, float, float]] = None) -> sitk.Image:
        """Resample image using SimpleITK with high-quality interpolation"""

        if target_spacing is None:
            target_spacing = self.target_spacing

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Calculate new size
        new_size = [
            int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
            for i in range(len(original_size))
        ]

        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkBSpline)  # High-quality interpolation

        return resampler.Execute(image)

    def normalize_intensity(self,
                            volume: np.ndarray,
                            method: str = 'window') -> np.ndarray:
        """Normalize CT intensity values"""

        if method == 'window':
            # CT windowing
            min_val = self.window_level - self.window_width / 2
            max_val = self.window_level + self.window_width / 2
            volume = np.clip(volume, min_val, max_val)
            volume = (volume - min_val) / (max_val - min_val)

        elif method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(volume)
            std_val = np.std(volume)
            volume = (volume - mean_val) / (std_val + 1e-8)

        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(volume)
            max_val = np.max(volume)
            volume = (volume - min_val) / (max_val - min_val + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return volume.astype(np.float32)

    def remove_bed(self, volume: np.ndarray, threshold: float = -500) -> np.ndarray:
        """Remove CT bed artifacts"""

        # Create mask for body region (above threshold)
        body_mask = volume > threshold

        # Morphological operations to clean up mask
        body_mask = ndimage.binary_opening(body_mask, iterations=2)
        body_mask = ndimage.binary_closing(body_mask, iterations=2)

        # Keep largest connected component (body)
        labeled, num_labels = ndimage.label(body_mask)
        if num_labels > 0:
            # Find largest component
            sizes = ndimage.sum(body_mask, labeled, range(1, num_labels + 1))
            largest_label = np.argmax(sizes) + 1
            body_mask = labeled == largest_label

        # Apply mask
        volume_cleaned = volume.copy()
        volume_cleaned[~body_mask] = threshold

        return volume_cleaned

    def crop_to_body(self,
                     volume: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Crop volume to body region with padding"""

        if mask is None:
            # Create body mask
            mask = volume > -500
            mask = ndimage.binary_opening(mask, iterations=2)

        # Find bounding box
        coords = np.where(mask)

        if len(coords[0]) == 0:
            # No body found, return original
            return volume, {'crop_coords': None}

        min_coords = [np.min(coord) for coord in coords]
        max_coords = [np.max(coord) for coord in coords]

        # Add padding
        padding = 10
        min_coords = [max(0, coord - padding) for coord in min_coords]
        max_coords = [min(volume.shape[i], coord + padding) for i, coord in enumerate(max_coords)]

        # Crop
        cropped = volume[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]

        crop_info = {
            'crop_coords': (min_coords, max_coords),
            'original_shape': volume.shape,
            'cropped_shape': cropped.shape
        }

        return cropped, crop_info

    def preprocess_volume(self,
                          volume: np.ndarray,
                          spacing: Optional[Tuple[float, float, float]] = None) -> Tuple[np.ndarray, Dict]:
        """Complete preprocessing pipeline"""

        preprocessing_info = {
            'original_shape': volume.shape,
            'original_range': [float(np.min(volume)), float(np.max(volume))],
            'steps_applied': []
        }

        # Step 1: Remove bed artifacts
        volume = self.remove_bed(volume)
        preprocessing_info['steps_applied'].append('remove_bed')

        # Step 2: Normalize intensity
        volume = self.normalize_intensity(volume, method='window')
        preprocessing_info['steps_applied'].append('normalize_intensity')
        preprocessing_info['normalized_range'] = [float(np.min(volume)), float(np.max(volume))]

        # Step 3: Crop to body region
        volume, crop_info = self.crop_to_body(volume)
        preprocessing_info.update(crop_info)
        preprocessing_info['steps_applied'].append('crop_to_body')

        # Step 4: Resize to target size if specified
        if self.target_size is not None:
            current_shape = volume.shape
            zoom_factors = [t / c for t, c in zip(self.target_size, current_shape)]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
            preprocessing_info['steps_applied'].append('resize')
            preprocessing_info['resize_factors'] = zoom_factors

        preprocessing_info['final_shape'] = volume.shape
        preprocessing_info['final_range'] = [float(np.min(volume)), float(np.max(volume))]

        return volume, preprocessing_info


def preprocess_ct_volume(volume: np.ndarray,
                         spacing: Optional[Tuple[float, float, float]] = None,
                         target_size: Optional[Tuple[int, int, int]] = (128, 192, 192)) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for CT preprocessing

    Args:
        volume: Input CT volume
        spacing: Voxel spacing (if available)
        target_size: Target output size

    Returns:
        Tuple of (preprocessed_volume, preprocessing_info)
    """

    preprocessor = CTPreprocessor(target_size=target_size)
    return preprocessor.preprocess_volume(volume, spacing)


def compute_volume_statistics(volume: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive volume statistics"""

    stats = {
        'shape': volume.shape,
        'min': float(np.min(volume)),
        'max': float(np.max(volume)),
        'mean': float(np.mean(volume)),
        'std': float(np.std(volume)),
        'median': float(np.median(volume)),
        'percentile_1': float(np.percentile(volume, 1)),
        'percentile_99': float(np.percentile(volume, 99)),
        'non_zero_voxels': int(np.count_nonzero(volume)),
        'total_voxels': int(volume.size),
        'non_zero_fraction': float(np.count_nonzero(volume) / volume.size)
    }

    return stats