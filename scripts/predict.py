#!/usr/bin/env python3
"""
AMNet Inference Script
Single case or batch prediction with AMNet
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from amnet.config import Config
from amnet.models.amnet import AMNet
from amnet.data.transforms import MedicalTransforms
from amnet.utils.logging import setup_logging
from amnet.utils.io import (
    load_nifti_volume, save_nifti_volume,
    save_predictions, load_config
)
from amnet.utils.visualization import plot_segmentation_results

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="AMNet inference for single case or batch prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CT volume path or directory"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./predictions",
        help="Output directory for predictions"
    )

    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Ground truth segmentation for comparison"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--save_format",
        type=str,
        default="nifti",
        choices=["nifti", "numpy", "both"],
        help="Output format for predictions"
    )

    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Process all NIfTI files in input directory"
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use test time augmentation"
    )

    return parser.parse_args()


class AMNetPredictor:
    """Professional inference pipeline for AMNet"""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):

        self.device = device

        # Load checkpoint
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load configuration
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = Config()
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            logger.warning("No config in checkpoint, using default")
            self.config = Config()

        # Create and load model
        self.model = AMNet(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        # Get transforms
        self.transforms = MedicalTransforms.get_validation_transforms()

        logger.info(f"Model loaded with {self.model.count_parameters():,} parameters")

    def preprocess_volume(self, volume: np.ndarray) -> torch.Tensor:
        """Preprocess volume for inference"""

        # Apply transforms
        sample = {
            'image': torch.from_numpy(volume).float(),
            'case_id': 'inference'
        }

        sample = self.transforms(sample)

        # Add batch dimension and move to device
        image = sample['image'].unsqueeze(0).to(self.device)

        return image

    def predict_single(self,
                       volume: np.ndarray,
                       use_tta: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Predict segmentation for single volume

        Args:
            volume: Input CT volume
            use_tta: Use test time augmentation

        Returns:
            Tuple of (prediction, inference_info)
        """

        import time
        start_time = time.time()

        with torch.no_grad():

            if use_tta:
                # Test time augmentation
                predictions = self._predict_with_tta(volume)
            else:
                # Standard prediction
                image = self.preprocess_volume(volume)
                outputs = self.model(image)
                predictions = outputs['predictions']

                # Convert to class predictions
                predictions = torch.argmax(predictions, dim=1)
                predictions = predictions.cpu().numpy()[0]  # Remove batch dim

        inference_time = time.time() - start_time

        inference_info = {
            'inference_time': inference_time,
            'input_shape': volume.shape,
            'output_shape': predictions.shape,
            'use_tta': use_tta,
            'device': self.device
        }

        logger.info(f"Prediction completed in {inference_time:.3f}s")

        return predictions, inference_info

    def _predict_with_tta(self, volume: np.ndarray) -> np.ndarray:
        """Predict with test time augmentation"""

        predictions_list = []

        # Original
        image = self.preprocess_volume(volume)
        outputs = self.model(image)
        pred = torch.softmax(outputs['predictions'], dim=1)
        predictions_list.append(pred)

        # Flipped versions
        flips = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4)]

        for flip_dims in flips:
            # Flip input
            image_flipped = torch.flip(image, dims=flip_dims)

            # Predict
            outputs = self.model(image_flipped)
            pred = torch.softmax(outputs['predictions'], dim=1)

            # Flip back
            pred_flipped = torch.flip(pred, dims=flip_dims)
            predictions_list.append(pred_flipped)

        # Average all predictions
        avg_pred = torch.stack(predictions_list).mean(dim=0)
        final_pred = torch.argmax(avg_pred, dim=1)

        return final_pred.cpu().numpy()[0]

    def predict_file(self,
                     input_path: str,
                     output_dir: str,
                     ground_truth_path: str = None,
                     visualize: bool = False,
                     save_format: str = "nifti",
                     use_tta: bool = False) -> dict:
        """Predict segmentation for a single file"""

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        case_id = input_path.stem.replace('.nii', '')

        logger.info(f"Processing case: {case_id}")

        # Load input volume
        try:
            volume, metadata = load_nifti_volume(input_path)
            logger.info(f"Loaded volume: {volume.shape}")
        except Exception as e:
            logger.error(f"Error loading {input_path}: {e}")
            return {'success': False, 'error': str(e)}

        # Predict
        try:
            prediction, inference_info = self.predict_single(volume, use_tta)
            logger.info(f"Prediction shape: {prediction.shape}")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {'success': False, 'error': str(e)}

        # Save predictions
        try:
            saved_files = save_predictions(
                predictions=prediction,
                case_id=case_id,
                output_dir=output_dir,
                metadata=metadata,
                save_format=save_format
            )
            logger.info(f"Predictions saved: {saved_files}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return {'success': False, 'error': str(e)}

        results = {
            'success': True,
            'case_id': case_id,
            'input_path': str(input_path),
            'prediction_shape': prediction.shape,
            'inference_info': inference_info,
            'saved_files': [str(f) for f in saved_files]
        }

        # Load ground truth and compute metrics if available
        if ground_truth_path:
            try:
                gt_volume, _ = load_nifti_volume(ground_truth_path)

                # Quick metrics calculation
                from amnet.metrics.segmentation import SegmentationMetrics
                metrics_calc = SegmentationMetrics(16)

                # Compute Dice for each organ
                dice_scores = {}
                for class_id in range(1, 16):
                    dice = metrics_calc.dice_coefficient(prediction, gt_volume, class_id)
                    organ_name = metrics_calc.class_names[class_id]
                    dice_scores[organ_name] = dice

                results['metrics'] = {
                    'dice_scores': dice_scores,
                    'mean_dice': np.mean(list(dice_scores.values()))
                }

                logger.info(f"Mean Dice Score: {results['metrics']['mean_dice']:.4f}")

                # Visualization
                if visualize:
                    vis_path = output_dir / f"{case_id}_comparison.png"
                    plot_segmentation_results(
                        volume=volume,
                        ground_truth=gt_volume,
                        prediction=prediction,
                        case_id=case_id,
                        save_path=vis_path
                    )
                    results['visualization'] = str(vis_path)

            except Exception as e:
                logger.warning(f"Error processing ground truth: {e}")

        return results


def process_batch(predictor: AMNetPredictor, args) -> List[dict]:
    """Process batch of files"""

    input_dir = Path(args.input)
    results = []

    # Find all NIfTI files
    nifti_files = list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))

    if not nifti_files:
        logger.error(f"No NIfTI files found in {input_dir}")
        return results

    logger.info(f"Found {len(nifti_files)} files to process")

    for i, input_file in enumerate(nifti_files, 1):
        logger.info(f"Processing {i}/{len(nifti_files)}: {input_file.name}")

        # Find corresponding ground truth if available
        gt_path = None
        if args.ground_truth:
            gt_dir = Path(args.ground_truth)
            gt_path = gt_dir / input_file.name
            if not gt_path.exists():
                gt_path = None

        result = predictor.predict_file(
            input_path=str(input_file),
            output_dir=args.output_dir,
            ground_truth_path=str(gt_path) if gt_path else None,
            visualize=args.visualize,
            save_format=args.save_format,
            use_tta=args.tta
        )

        results.append(result)

    return results


def main():
    """Main prediction function"""

    args = parse_arguments()

    # Setup logging
    setup_logging(
        log_dir=Path(args.output_dir) / "logs",
        level=logging.INFO
    )

    logger.info("=" * 80)
    logger.info("AMNet Inference Started")
    logger.info("=" * 80)

    # Check inputs
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.input).exists():
        logger.error(f"Input not found: {args.input}")
        sys.exit(1)

    # Create predictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        predictor = AMNetPredictor(args.checkpoint, device)
    except Exception as e:
        logger.error(f"Error creating predictor: {e}")
        sys.exit(1)

    # Run prediction
    try:
        if args.batch_mode:
            results = process_batch(predictor, args)

            # Save batch results
            import json
            results_file = Path(args.output_dir) / "batch_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            # Summary
            successful = sum(1 for r in results if r.get('success', False))
            logger.info(f"Batch processing completed: {successful}/{len(results)} successful")

        else:
            result = predictor.predict_file(
                input_path=args.input,
                output_dir=args.output_dir,
                ground_truth_path=args.ground_truth,
                visualize=args.visualize,
                save_format=args.save_format,
                use_tta=args.tta
            )

            if result['success']:
                logger.info("Prediction completed successfully")
            else:
                logger.error(f"Prediction failed: {result.get('error')}")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Prediction interrupted by user")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

    finally:
        logger.info("Prediction script finished")


if __name__ == "__main__":
    main()