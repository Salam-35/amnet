"""
I/O utilities for AMNet
Professional file handling for medical images and results
"""

import os
import json
import yaml
import pickle
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def load_nifti_volume(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load NIfTI volume with metadata

    Args:
        filepath: Path to NIfTI file

    Returns:
        Tuple of (volume_data, metadata)
    """
    try:
        filepath = Path(filepath)

        # Load with nibabel
        img = nib.load(filepath)
        data = img.get_fdata().astype(np.float32)

        # Extract metadata
        metadata = {
            'shape': data.shape,
            'affine': img.affine,
            'header': dict(img.header),
            'spacing': img.header.get_zooms(),
            'orientation': nib.orientations.io_orientation(img.affine),
            'filepath': str(filepath)
        }

        logger.debug(f"Loaded NIfTI: {filepath} - Shape: {data.shape}")
        return data, metadata

    except Exception as e:
        logger.error(f"Error loading NIfTI file {filepath}: {e}")
        raise

def save_nifti_volume(data: np.ndarray,
                      filepath: Union[str, Path],
                      affine: Optional[np.ndarray] = None,
                      header: Optional[nib.Nifti1Header] = None) -> None:
    """
    Save volume as NIfTI file

    Args:
        data: Volume data to save
        filepath: Output filepath
        affine: Affine transformation matrix
        header: NIfTI header
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Default affine if not provided
        if affine is None:
            affine = np.eye(4)

        # Create NIfTI image
        img = nib.Nifti1Image(data, affine, header)

        # Save
        nib.save(img, filepath)
        logger.debug(f"Saved NIfTI: {filepath} - Shape: {data.shape}")

    except Exception as e:
        logger.error(f"Error saving NIfTI file {filepath}: {e}")
        raise

def load_sitk_volume(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load volume using SimpleITK with comprehensive metadata

    Args:
        filepath: Path to medical image file

    Returns:
        Tuple of (volume_data, metadata)
    """
    try:
        filepath = str(filepath)

        # Load with SimpleITK
        img = sitk.ReadImage(filepath)
        data = sitk.GetArrayFromImage(img).astype(np.float32)

        # Extract comprehensive metadata
        metadata = {
            'size': img.GetSize(),
            'spacing': img.GetSpacing(),
            'origin': img.GetOrigin(),
            'direction': img.GetDirection(),
            'dimension': img.GetDimension(),
            'pixel_type': img.GetPixelIDTypeAsString(),
            'number_of_components': img.GetNumberOfComponentsPerPixel(),
            'filepath': filepath
        }

        # Add DICOM metadata if available
        for key in img.GetMetaDataKeys():
            metadata[f'dicom_{key}'] = img.GetMetaData(key)

        logger.debug(f"Loaded with SimpleITK: {filepath} - Shape: {data.shape}")
        return data, metadata

    except Exception as e:
        logger.error(f"Error loading with SimpleITK {filepath}: {e}")
        raise

def save_sitk_volume(data: np.ndarray,
                     filepath: Union[str, Path],
                     spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     direction: Optional[Tuple] = None) -> None:
    """
    Save volume using SimpleITK

    Args:
        data: Volume data to save
        filepath: Output filepath
        spacing: Voxel spacing
        origin: Image origin
        direction: Image direction cosines
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create SimpleITK image
        img = sitk.GetImageFromArray(data)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)

        if direction is not None:
            img.SetDirection(direction)

        # Save
        sitk.WriteImage(img, str(filepath))
        logger.debug(f"Saved with SimpleITK: {filepath} - Shape: {data.shape}")

    except Exception as e:
        logger.error(f"Error saving with SimpleITK {filepath}: {e}")
        raise

def save_config(config: Any, filepath: Union[str, Path]) -> None:
    """Save configuration to YAML file"""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict if needed
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    logger.info(f"Configuration saved to {filepath}")

def load_config(filepath: Union[str, Path]) -> Dict:
    """Load configuration from YAML file"""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {filepath}")
    return config

def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save dictionary as JSON file"""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug(f"JSON saved to {filepath}")

def load_json(filepath: Union[str, Path]) -> Dict:
    """Load JSON file"""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.debug(f"JSON loaded from {filepath}")
    return data

def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Save object using pickle"""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug(f"Pickle saved to {filepath}")

def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickled object"""

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    logger.debug(f"Pickle loaded from {filepath}")
    return data

def save_predictions(predictions: np.ndarray,
                    case_id: str,
                    output_dir: Union[str, Path],
                    metadata: Optional[Dict] = None,
                    save_format: str = 'nifti') -> List[Path]:
    """
    Save model predictions with metadata

    Args:
        predictions: Prediction volume
        case_id: Case identifier
        output_dir: Output directory
        metadata: Original image metadata
        save_format: 'nifti', 'numpy', or 'both'

    Returns:
        List of saved file paths
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    if save_format in ['nifti', 'both']:
        nifti_path = output_dir / f"{case_id}_prediction.nii.gz"

        affine = metadata.get('affine') if metadata else None
        header = metadata.get('header') if metadata else None

        save_nifti_volume(predictions, nifti_path, affine, header)
        saved_files.append(nifti_path)

    if save_format in ['numpy', 'both']:
        numpy_path = output_dir / f"{case_id}_prediction.npy"
        np.save(numpy_path, predictions)
        saved_files.append(numpy_path)

    # Save metadata if provided
    if metadata:
        metadata_path = output_dir / f"{case_id}_metadata.json"

        # Convert numpy arrays to lists for JSON serialization
        json_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                json_metadata[key] = value.tolist()
            else:
                json_metadata[key] = value

        save_json(json_metadata, metadata_path)
        saved_files.append(metadata_path)

    logger.info(f"Predictions saved for {case_id}: {len(saved_files)} files")
    return saved_files

def load_model_checkpoint(checkpoint_path: Union[str, Path],
                         device: str = 'cpu') -> Dict[str, Any]:
    """Load model checkpoint with comprehensive validation"""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")

        # Log checkpoint info
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_dice': checkpoint.get('best_dice', 'unknown'),
            'file_size': checkpoint_path.stat().st_size / 1024 / 1024,  # MB
        }

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Checkpoint info: {info}")

        return checkpoint

    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

def export_model_to_onnx(model,
                        input_shape: Tuple[int, ...],
                        output_path: Union[str, Path],
                        opset_version: int = 11) -> None:
    """Export PyTorch model to ONNX format"""

    try:
        import torch.onnx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        logger.info(f"Model exported to ONNX: {output_path}")

    except ImportError:
        logger.error("ONNX export requires torch.onnx")
        raise
    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        raise

def create_submission_format(predictions_dir: Union[str, Path],
                           output_file: Union[str, Path],
                           format_type: str = 'amos') -> None:
    """Create competition submission format"""

    predictions_dir = Path(predictions_dir)
    output_file = Path(output_file)

    if format_type == 'amos':
        # AMOS22 submission format
        import zipfile

        with zipfile.ZipFile(output_file, 'w') as zf:
            for pred_file in predictions_dir.glob('*_prediction.nii.gz'):
                # Rename to expected format
                case_id = pred_file.stem.replace('_prediction', '')
                submission_name = f"{case_id}.nii.gz"
                zf.write(pred_file, submission_name)

        logger.info(f"AMOS submission created: {output_file}")

    else:
        raise ValueError(f"Unknown submission format: {format_type}")

def batch_convert_format(input_dir: Union[str, Path],
                        output_dir: Union[str, Path],
                        input_format: str = 'nii.gz',
                        output_format: str = 'npy') -> None:
    """Batch convert between different medical image formats"""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    if input_format == 'nii.gz':
        input_files = list(input_dir.glob('*.nii.gz'))
    elif input_format == 'nii':
        input_files = list(input_dir.glob('*.nii'))
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    logger.info(f"Converting {len(input_files)} files from {input_format} to {output_format}")

    for input_file in input_files:
        try:
            # Load data
            if input_format in ['nii', 'nii.gz']:
                data, metadata = load_nifti_volume(input_file)
            else:
                continue

            # Save in output format
            output_file = output_dir / f"{input_file.stem.replace('.nii', '')}.{output_format}"

            if output_format == 'npy':
                np.save(output_file, data)
            elif output_format in ['nii', 'nii.gz']:
                affine = metadata.get('affine')
                header = metadata.get('header')
                save_nifti_volume(data, output_file, affine, header)

            logger.debug(f"Converted: {input_file.name} -> {output_file.name}")

        except Exception as e:
            logger.error(f"Error converting {input_file}: {e}")

    logger.info(f"Conversion completed: {output_dir}")

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """Compute file hash for integrity checking"""

    import hashlib

    filepath = Path(filepath)

    if algorithm == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()

def validate_medical_image(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Validate medical image file and return diagnostic info"""

    filepath = Path(filepath)
    validation_result = {
        'valid': False,
        'file_exists': filepath.exists(),
        'file_size_mb': 0,
        'issues': [],
        'properties': {}
    }

    if not filepath.exists():
        validation_result['issues'].append('File does not exist')
        return validation_result

    # File size
    validation_result['file_size_mb'] = filepath.stat().st_size / 1024 / 1024

    try:
        # Load and validate
        data, metadata = load_nifti_volume(filepath)

        # Check basic properties
        validation_result['properties'] = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'min_value': float(np.min(data)),
            'max_value': float(np.max(data)),
            'mean_value': float(np.mean(data)),
            'has_nan': bool(np.any(np.isnan(data))),
            'has_inf': bool(np.any(np.isinf(data))),
            'spacing': metadata.get('spacing'),
        }

        # Check for issues
        if np.any(np.isnan(data)):
            validation_result['issues'].append('Contains NaN values')

        if np.any(np.isinf(data)):
            validation_result['issues'].append('Contains infinite values')

        if data.size == 0:
            validation_result['issues'].append('Empty volume')

        if len(data.shape) != 3:
            validation_result['issues'].append(f'Expected 3D volume, got {len(data.shape)}D')

        # Mark as valid if no critical issues
        critical_issues = ['Contains NaN values', 'Contains infinite values', 'Empty volume']
        has_critical_issues = any(issue in validation_result['issues'] for issue in critical_issues)
        validation_result['valid'] = not has_critical_issues

    except Exception as e:
        validation_result['issues'].append(f'Loading error: {str(e)}')

    return validation_result

def verify_data_integrity(data_root: Union[str, Path]) -> Dict[str, Any]:
    """
    Verify dataset integrity and report statistics

    Args:
        data_root: Root directory of dataset

    Returns:
        Dictionary with verification results
    """

    data_root = Path(data_root)
    results = {
        'valid_cases': [],
        'invalid_cases': [],
        'statistics': {},
        'errors': []
    }

    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"

    if not images_dir.exists():
        results['errors'].append(f"Images directory not found: {images_dir}")
        return results

    if not labels_dir.exists():
        results['errors'].append(f"Labels directory not found: {labels_dir}")
        return results

    # Check each image file
    for img_file in images_dir.glob("*.nii.gz"):
        label_file = labels_dir / img_file.name
        case_id = img_file.stem.replace('.nii', '')

        case_info = {'case_id': case_id, 'issues': []}

        try:
            # Check if label file exists
            if not label_file.exists():
                case_info['issues'].append('Missing label file')
                results['invalid_cases'].append(case_info)
                continue

            # Load and verify image
            img_data, img_meta = load_nifti_volume(img_file)
            label_data, label_meta = load_nifti_volume(label_file)

            # Check shapes match
            if img_data.shape != label_data.shape:
                case_info['issues'].append(f'Shape mismatch: img {img_data.shape} vs label {label_data.shape}')

            # Check label values
            unique_labels = np.unique(label_data)
            if np.max(unique_labels) > 15 or np.min(unique_labels) < 0:
                case_info['issues'].append(f'Invalid label values: {unique_labels}')

            # Check for NaN or inf values
            if np.any(~np.isfinite(img_data)):
                case_info['issues'].append('Non-finite values in image')

            if len(case_info['issues']) == 0:
                case_info.update({
                    'image_shape': img_data.shape,
                    'image_spacing': img_meta.get('spacing'),
                    'label_classes': unique_labels.tolist(),
                    'image_range': [float(np.min(img_data)), float(np.max(img_data))]
                })
                results['valid_cases'].append(case_info)
            else:
                results['invalid_cases'].append(case_info)

        except Exception as e:
            case_info['issues'].append(f'Loading error: {str(e)}')
            results['invalid_cases'].append(case_info)

    # Compute statistics
    if results['valid_cases']:
        shapes = [case['image_shape'] for case in results['valid_cases']]
        spacings = [case['image_spacing'] for case in results['valid_cases'] if case['image_spacing']]

        results['statistics'] = {
            'total_cases': len(results['valid_cases']) + len(results['invalid_cases']),
            'valid_cases': len(results['valid_cases']),
            'invalid_cases': len(results['invalid_cases']),
            'common_shape': max(set(map(tuple, shapes)), key=shapes.count) if shapes else None,
            'shape_variations': len(set(map(tuple, shapes))) if shapes else 0,
            'average_spacing': np.mean(spacings, axis=0).tolist() if spacings else None
        }

    logger.info(f"Data verification complete: {results['statistics']}")
    return results