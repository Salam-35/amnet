# test_data.py
"""
Test data loading and preprocessing functionality
"""

import pytest
import numpy as np
import torch
import tempfile
import nibabel as nib
from pathlib import Path
import json

import sys

sys.path.append(str(Path(__file__).parent.parent))

from amnet.data.dataset import AMOSDataset
from amnet.data.transforms import MedicalTransforms
from amnet.data.preprocessing import CTPreprocessor, preprocess_ct_volume
from config import Config


class TestAMOSDataset:
    """Test AMOS dataset functionality"""

    @pytest.fixture
    def mock_data_dir(self):
        """Create mock data directory structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directory structure
            images_dir = temp_path / "imagesTr"
            labels_dir = temp_path / "labelsTr"
            images_dir.mkdir()
            labels_dir.mkdir()

            # Create mock NIfTI files
            for i in range(5):
                # Create mock volume data
                volume_shape = (64, 96, 96)
                image_data = np.random.randn(*volume_shape).astype(np.float32)
                label_data = np.random.randint(0, 16, volume_shape).astype(np.int16)

                # Create NIfTI images
                img_nifti = nib.Nifti1Image(image_data, np.eye(4))
                label_nifti = nib.Nifti1Image(label_data, np.eye(4))

                # Save files
                nib.save(img_nifti, images_dir / f"case_{i:03d}.nii.gz")
                nib.save(label_nifti, labels_dir / f"case_{i:03d}.nii.gz")

            # Create train split file
            train_split = []
            for i in range(5):
                train_split.append({
                    "case_id": f"case_{i:03d}",
                    "image": str(images_dir / f"case_{i:03d}.nii.gz"),
                    "label": str(labels_dir / f"case_{i:03d}.nii.gz")
                })

            with open(temp_path / "train.json", 'w') as f:
                json.dump(train_split, f)

            yield temp_path

    def test_dataset_creation(self, mock_data_dir):
        """Test dataset creation"""
        config = Config()
        dataset = AMOSDataset(
            data_root=str(mock_data_dir),
            split="train",
            config=config,
            cache_data=False
        )

        assert len(dataset) == 5
        assert dataset.num_classes == 16

    def test_dataset_getitem(self, mock_data_dir):
        """Test dataset __getitem__ method"""
        config = Config()
        dataset = AMOSDataset(
            data_root=str(mock_data_dir),
            split="train",
            config=config,
            cache_data=False
        )

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "mask" in sample
        assert "case_id" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64


class TestMedicalTransforms:
    """Test medical image transforms"""

    def test_ct_intensity_normalization(self):
        """Test CT intensity normalization"""
        # Create mock CT volume
        volume = np.random.randn(64, 96, 96) * 1000  # Simulate HU values

        normalized = MedicalTransforms.normalize_ct_intensity(volume)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32

    def test_volume_resampling(self):
        """Test volume resampling"""
        volume = np.random.randn(32, 48, 48).astype(np.float32)
        target_spacing = (2.0, 1.0, 1.0)
        current_spacing = (1.0, 1.0, 1.0)

        resampled = MedicalTransforms.resample_volume(
            volume, target_spacing, current_spacing
        )

        # Check that resampling changed the size
        assert resampled.shape != volume.shape
        assert resampled.dtype == volume.dtype

    def test_training_transforms(self):
        """Test training transforms"""
        config = Config()
        config.model.input_size = (64, 96, 96)

        transform_fn = MedicalTransforms.get_training_transforms(config)

        # Create mock sample
        sample = {
            'image': torch.randn(64, 96, 96),
            'mask': torch.randint(0, 16, (64, 96, 96)),
            'case_id': 'test_case'
        }

        transformed = transform_fn(sample)

        assert 'image' in transformed
        assert 'mask' in transformed
        assert transformed['image'].shape[0] == 1  # Channel dimension added
        assert transformed['image'].shape[1:] == config.model.input_size


class TestCTPreprocessor:
    """Test CT preprocessing functionality"""

    def test_preprocessor_initialization(self):
        """Test CT preprocessor initialization"""
        preprocessor = CTPreprocessor()

        assert preprocessor.target_spacing == (1.5, 1.0, 1.0)
        assert preprocessor.window_level == 40.0
        assert preprocessor.window_width == 400.0

    def test_intensity_normalization(self):
        """Test intensity normalization methods"""
        preprocessor = CTPreprocessor()
        volume = np.random.randn(32, 48, 48) * 1000

        # Test windowing
        normalized = preprocessor.normalize_intensity(volume, method='window')
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Test z-score normalization
        normalized_zscore = preprocessor.normalize_intensity(volume, method='zscore')
        assert abs(normalized_zscore.mean()) < 1e-6  # Mean should be ~0
        assert abs(normalized_zscore.std() - 1.0) < 1e-6  # Std should be ~1

    def test_bed_removal(self):
        """Test CT bed artifact removal"""
        preprocessor = CTPreprocessor()

        # Create volume with "bed" (very low values)
        volume = np.random.randn(32, 48, 48) * 100
        volume[:, :10, :] = -1000  # Simulate bed

        cleaned = preprocessor.remove_bed(volume)

        # Check that bed regions are set to threshold value
        assert np.all(cleaned[:, :10, :] == -500)

    def test_complete_preprocessing(self):
        """Test complete preprocessing pipeline"""
        volume = np.random.randn(32, 48, 48) * 500 + 40  # Simulate CT data

        processed, info = preprocess_ct_volume(volume, target_size=(64, 96, 96))

        assert processed.shape == (64, 96, 96)
        assert 'original_shape' in info
        assert 'final_shape' in info
        assert 'steps_applied' in info


