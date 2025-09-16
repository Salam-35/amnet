#!/usr/bin/env python3
"""
AMNet Evaluation Script
Comprehensive model evaluation on test dataset
"""

import sys
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.config import Config
from amnet.data.dataset import AMOSDataset
from amnet.data.transforms import MedicalTransforms
from amnet.models.amnet import AMNet
from amnet.evaluation.evaluator import AMNetEvaluator
from amnet.utils.logging import setup_logging, log_system_info
from amnet.utils.io import load_json

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Evaluate AMNet on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
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
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction volumes"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )

    return parser.parse_args()


def load_model_and_config(checkpoint_path: str) -> tuple:
    """Load model and configuration from checkpoint"""

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = Config()

        # Update config with saved values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        logger.warning("No config found in checkpoint, using default")
        config = Config()

    # Create and load model
    model = AMNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Log model info
    epoch = checkpoint.get('epoch', 'unknown')
    best_dice = checkpoint.get('best_dice', 'unknown')

    logger.info(f"Model loaded from epoch {epoch}")
    logger.info(f"Best Dice score: {best_dice}")
    logger.info(f"Model parameters: {model.count_parameters():,}")

    return model, config


def create_test_loader(config, data_root: str, split: str, batch_size: int, num_workers: int):
    """Create test data loader"""

    # Get validation transforms (no augmentation)
    transforms = MedicalTransforms.get_validation_transforms()

    # Create dataset
    test_dataset = AMOSDataset(
        data_root=data_root,
        split=split,
        config=config,
        transforms=transforms,
        cache_data=False  # Don't cache for evaluation
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"Test loader created: {len(test_dataset)} samples, {len(test_loader)} batches")

    return test_loader


def main():
    """Main evaluation function"""

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(
        log_dir=Path(args.output_dir) / "logs",
        level=logging.INFO
    )

    logger.info("=" * 80)
    logger.info("AMNet Evaluation Started")
    logger.info("=" * 80)

    # Log system info
    log_system_info()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model and config
    try:
        model, config = load_model_and_config(checkpoint_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create test loader
    try:
        test_loader = create_test_loader(
            config=config,
            data_root=args.data_root,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    except Exception as e:
        logger.error(f"Error creating data loader: {e}")
        sys.exit(1)

    # Create evaluator
    evaluator = AMNetEvaluator(
        model=model,
        config=config,
        device=device
    )

    try:
        # Run evaluation
        logger.info("Starting comprehensive evaluation...")

        overall_metrics = evaluator.evaluate_dataset(
            test_loader=test_loader,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )

        # Print final results
        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 80)

        logger.info(f"Overall Results:")
        logger.info(f"  Mean Dice Score: {overall_metrics['mean_dice']:.4f}")
        logger.info(f"  Mean Inference Time: {overall_metrics['mean_inference_time']:.3f}s")

        # Save final summary
        summary_file = output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            import json
            json.dump(overall_metrics, f, indent=2)

        logger.info(f"Detailed results saved to: {output_dir}")
        logger.info(f"Summary saved to: {summary_file}")

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    finally:
        logger.info("Evaluation script finished")


if __name__ == "__main__":
    main()