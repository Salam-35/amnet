#!/usr/bin/env python3
"""
Data preparation script for AMOS22 dataset
Professional data preprocessing and validation
"""

import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from amnet.utils.logging import setup_logging
from amnet.utils.io import (
    load_nifti_volume, save_nifti_volume,
    verify_data_integrity, save_json
)
from amnet.data.preprocessing import preprocess_ct_volume, compute_volume_statistics

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Prepare AMOS22 dataset for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of AMOS22 dataset"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply preprocessing to images"
    )

    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify data integrity without preprocessing"
    )

    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create train/val/test splits"
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )

    parser.add_argument(
        "--target_size",
        nargs=3,
        type=int,
        default=[128, 192, 192],
        help="Target volume size (D H W)"
    )

    return parser.parse_args()


def create_dataset_splits(data_root: Path,
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Dict[str, List]:
    """Create train/validation/test splits"""

    logger.info("Creating dataset splits...")

    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Find all valid image-label pairs
    valid_cases = []

    for img_file in images_dir.glob("*.nii.gz"):
        label_file = labels_dir / img_file.name

        if label_file.exists():
            case_info = {
                "case_id": img_file.stem.replace('.nii', ''),
                "image": str(img_file),
                "label": str(label_file)
            }
            valid_cases.append(case_info)

    logger.info(f"Found {len(valid_cases)} valid cases")

    # Shuffle cases
    np.random.seed(42)
    np.random.shuffle(valid_cases)

    # Create splits
    n_total = len(valid_cases)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    splits = {
        'train': valid_cases[:n_train],
        'val': valid_cases[n_train:n_train + n_val],
        'test': valid_cases[n_train + n_val:]
    }

    # Save splits
    for split_name, cases in splits.items():
        split_file = data_root / f"{split_name}.json"
        save_json(cases, split_file)
        logger.info(f"{split_name.title()} split: {len(cases)} cases -> {split_file}")

    return splits


def analyze_dataset_statistics(data_root: Path) -> Dict:
    """Analyze dataset statistics"""

    logger.info("Analyzing dataset statistics...")

    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"

    statistics = {
        'total_cases': 0,
        'image_statistics': {
            'shapes': [],
            'spacings': [],
            'intensity_ranges': [],
            'volume_sizes_ml': []
        },
        'label_statistics': {
            'organ_counts': {},
            'volume_distributions': {}
        }
    }

    # Analyze each case
    for img_file in tqdm(list(images_dir.glob("*.nii.gz")), desc="Analyzing cases"):
        label_file = labels_dir / img_file.name

        if not label_file.exists():
            continue

        try:
            # Load image
            image, img_metadata = load_nifti_volume(img_file)
            image_stats = compute_volume_statistics(image)

            statistics['image_statistics']['shapes'].append(image.shape)
            statistics['image_statistics']['intensity_ranges'].append([image_stats['min'], image_stats['max']])

            if 'spacing' in img_metadata:
                statistics['image_statistics']['spacings'].append(img_metadata['spacing'])

            # Calculate volume in ml
            if 'spacing' in img_metadata:
                voxel_volume = np.prod(img_metadata['spacing'])  # mm³
                total_volume = voxel_volume * image.size / 1000  # ml
                statistics['image_statistics']['volume_sizes_ml'].append(total_volume)

            # Load label
            label, _ = load_nifti_volume(label_file)

            # Count organs
            unique_labels = np.unique(label)
            for organ_id in unique_labels:
                if organ_id not in statistics['label_statistics']['organ_counts']:
                    statistics['label_statistics']['organ_counts'][organ_id] = 0
                statistics['label_statistics']['organ_counts'][organ_id] += 1

            statistics['total_cases'] += 1

        except Exception as e:
            logger.warning(f"Error analyzing {img_file}: {e}")

    # Compute summary statistics
    if statistics['image_statistics']['shapes']:
        shapes = statistics['image_statistics']['shapes']
        statistics['summary'] = {
            'most_common_shape': max(set(map(tuple, shapes)), key=shapes.count),
            'shape_variations': len(set(map(tuple, shapes))),
            'mean_intensity_range': np.mean(statistics['image_statistics']['intensity_ranges'], axis=0).tolist()
        }

        if statistics['image_statistics']['spacings']:
            spacings = statistics['image_statistics']['spacings']
            statistics['summary']['mean_spacing'] = np.mean(spacings, axis=0).tolist()
            statistics['summary']['std_spacing'] = np.std(spacings, axis=0).tolist()

        if statistics['image_statistics']['volume_sizes_ml']:
            volumes = statistics['image_statistics']['volume_sizes_ml']
            statistics['summary']['mean_volume_ml'] = float(np.mean(volumes))
            statistics['summary']['std_volume_ml'] = float(np.std(volumes))

    logger.info(f"Dataset analysis completed: {statistics['total_cases']} cases")

    return statistics


def preprocess_dataset(data_root: Path,
                       output_dir: Path,
                       target_size: tuple = (128, 192, 192)) -> None:
    """Preprocess entire dataset"""

    logger.info("Starting dataset preprocessing...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories
    processed_images = output_dir / "imagesTr"
    processed_labels = output_dir / "labelsTr"
    processed_images.mkdir(exist_ok=True)
    processed_labels.mkdir(exist_ok=True)

    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"

    preprocessing_log = []

    for img_file in tqdm(list(images_dir.glob("*.nii.gz")), desc="Preprocessing"):
        label_file = labels_dir / img_file.name

        if not label_file.exists():
            continue

        case_id = img_file.stem.replace('.nii', '')

        try:
            # Load volumes
            image, img_metadata = load_nifti_volume(img_file)
            label, label_metadata = load_nifti_volume(label_file)

            logger.debug(f"Processing {case_id}: {image.shape}")

            # Preprocess image
            processed_image, preprocess_info = preprocess_ct_volume(
                image,
                spacing=img_metadata.get('spacing'),
                target_size=target_size
            )

            # Resize label to match
            if processed_image.shape != label.shape:
                from scipy import ndimage
                zoom_factors = [p / l for p, l in zip(processed_image.shape, label.shape)]
                processed_label = ndimage.zoom(label, zoom_factors, order=0)  # Nearest neighbor
            else:
                processed_label = label

            # Save processed volumes
            out_img_file = processed_images / img_file.name
            out_label_file = processed_labels / label_file.name

            save_nifti_volume(
                processed_image,
                out_img_file,
                affine=img_metadata.get('affine'),
                header=img_metadata.get('header')
            )

            save_nifti_volume(
                processed_label,
                out_label_file,
                affine=label_metadata.get('affine'),
                header=label_metadata.get('header')
            )

            # Log preprocessing info
            preprocessing_log.append({
                'case_id': case_id,
                'original_shape': image.shape,
                'processed_shape': processed_image.shape,
                'preprocessing_info': preprocess_info
            })

            logger.debug(f"Completed {case_id}: {image.shape} -> {processed_image.shape}")

        except Exception as e:
            logger.error(f"Error preprocessing {case_id}: {e}")

    # Save preprocessing log
    log_file = output_dir / "preprocessing_log.json"
    save_json(preprocessing_log, log_file)

    logger.info(f"Preprocessing completed: {len(preprocessing_log)} cases processed")
    logger.info(f"Output saved to: {output_dir}")


def main():
    """Main data preparation function"""

    args = parse_arguments()

    # Setup logging
    setup_logging(level=logging.INFO)

    logger.info("=" * 80)
    logger.info("AMOS22 Data Preparation Started")
    logger.info("=" * 80)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # Verify data root exists
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        sys.exit(1)

    # Verify data integrity
    logger.info("Step 1: Verifying data integrity...")
    integrity_results = verify_data_integrity(data_root)

    # Save integrity report
    output_dir.mkdir(parents=True, exist_ok=True)
    integrity_file = output_dir / "data_integrity_report.json"
    save_json(integrity_results, integrity_file)

    logger.info(f"Data verification completed:")
    logger.info(f"  Valid cases: {integrity_results['statistics']['valid_cases']}")
    logger.info(f"  Invalid cases: {integrity_results['statistics']['invalid_cases']}")

    if args.verify_only:
        logger.info("Verification complete (verify_only mode)")
        return

    # Analyze dataset statistics
    logger.info("Step 2: Analyzing dataset statistics...")
    statistics = analyze_dataset_statistics(data_root)

    # Save statistics
    stats_file = output_dir / "dataset_statistics.json"
    save_json(statistics, stats_file)
    logger.info(f"Dataset statistics saved to: {stats_file}")

    # Create splits
    if args.create_splits:
        logger.info("Step 3: Creating data splits...")
        splits = create_dataset_splits(
            data_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio
        )

        # Save split info
        split_info = {
            'train_cases': len(splits['train']),
            'val_cases': len(splits['val']),
            'test_cases': len(splits['test']),
            'ratios': {
                'train': args.train_ratio,
                'val': args.val_ratio,
                'test': 1.0 - args.train_ratio - args.val_ratio
            }
        }

        split_info_file = output_dir / "split_info.json"
        save_json(split_info, split_info_file)

    # Preprocess data
    if args.preprocess:
        logger.info("Step 4: Preprocessing dataset...")
        preprocess_dataset(
            data_root,
            output_dir,
            target_size=tuple(args.target_size)
        )

    logger.info("=" * 80)
    logger.info("Data Preparation Completed Successfully")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()