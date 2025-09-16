#!/usr/bin/env python3
"""
AMNet Training Script
Professional training script with comprehensive configuration and monitoring
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from amnet.data.dataset import AMOSDataset
from amnet.data.transforms import MedicalTransforms
from amnet.models.amnet import AMNet
from amnet.training.trainer import AMNetTrainer
from amnet.utils.logging import setup_logging
from amnet.utils.io import save_config, load_config

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train AMNet for abdominal organ segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
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
        default="./outputs",
        help="Output directory for results"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="AMNet-AbdominalSegmentation",
        help="Wandb project name"
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller dataset"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config batch size"
    )

    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use AMNet-Lite configuration (smaller model)"
    )

    return parser.parse_args()


def create_data_loaders(config, debug=False):
    """Create train and validation data loaders"""

    # Get transforms
    train_transforms = MedicalTransforms.get_training_transforms(config)
    val_transforms = MedicalTransforms.get_validation_transforms(config)

    # Create datasets
    train_dataset = AMOSDataset(
        data_root=config.data.root_dir,
        split="train",
        transforms=train_transforms,
        cache_data=not debug  # Disable caching in debug mode
    )

    val_dataset = AMOSDataset(
        data_root=config.data.root_dir,
        split="val",
        transforms=val_transforms,
        cache_data=not debug
    )

    # Debug mode: use smaller subset
    if debug:
        train_dataset.samples = train_dataset.samples[:10]
        val_dataset.samples = val_dataset.samples[:5]
        logger.info("Debug mode: Using reduced dataset")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
        persistent_workers=True if config.data.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=max(0, config.data.num_workers // 2),
        pin_memory=config.data.pin_memory,
        persistent_workers=True if config.data.num_workers > 0 else False
    )

    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer and scheduler if available
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_dice = checkpoint.get('best_dice', 0.0)

    logger.info(f"Loaded checkpoint from epoch {start_epoch}, best dice: {best_dice:.4f}")

    return start_epoch, best_dice


def main():
    """Main training function"""

    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(
        log_dir=Path(args.output_dir) / "logs",
        level=logging.DEBUG if args.debug else logging.INFO
    )

    logger.info("=" * 80)
    logger.info("AMNet Training Started")
    logger.info("=" * 80)

    # Load configuration
    config = Config(config_path=args.config if args.config else None)

    # Apply lite model configuration
    if args.lite:
        logger.info("Using AMNet-Micro configuration")
        # Ultra-minimal model for 24GB GPU with <20M parameters
        config.model.input_size = (64, 64, 64)  # Minimum viable input for ConvNeXt
        config.model.encoder_2d_name = "convnext_v2_tiny"
        config.model.feature_dim_2d = 64  # Small but usable
        config.model.encoder_2d_depths = [1, 1, 1, 1]  # Minimal depths
        config.model.encoder_3d_name = "resnet3d_18"
        config.model.feature_dim_3d = 128  # Small but usable
        config.model.encoder_3d_layers = [1, 1, 1, 1]  # Minimal layers
        config.model.fusion_dim = 32  # Small but usable
        config.model.scales = [1]  # Single scale only
        config.model.attention_heads = 1  # Single attention head
        config.training.batch_size = 4  # Reduce batch size for stability
        config.training.learning_rate = 0.0001  # Lower learning rate
        logger.info(f"AMNet-Ultra-Lite: input_size={config.model.input_size}, batch_size={config.training.batch_size}")

        # Update derived properties after lite configuration
        config._set_derived_properties()

    # Override config with arguments
    if args.data_root:
        config.data.root_dir = args.data_root
    if args.output_dir:
        config.paths.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_workers is not None:  # Check for None to allow 0
        config.data.num_workers = args.num_workers

    # Debug mode adjustments
    if args.debug:
        config.training.batch_size = min(config.training.batch_size, 2)
        config.training.max_epochs = 10
        config.logging.log_interval = 1
        logger.info("Debug mode enabled - reduced batch size and epochs")

    # Create output directories
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Save configuration
    save_config(config, output_dir / "config.yaml")

    # Log configuration
    logger.info("Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"Using device: {config.device}")
        if config.device.startswith('cuda'):
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available, using CPU")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, debug=args.debug)

    # Create model
    logger.info("Initializing model...")
    model = AMNet(config)
    logger.info(f"Model created with {model.count_parameters():,} parameters")

    # Create trainer
    trainer = AMNetTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        use_wandb=not args.no_wandb
    )

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch, best_dice = load_checkpoint(
                model, trainer.optimizer, trainer.scheduler, args.resume
            )
            trainer.current_epoch = start_epoch
            trainer.best_dice = best_dice
        else:
            logger.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)

    try:
        # Start training
        logger.info("Starting training...")
        training_history = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Best Dice Score: {trainer.best_dice:.4f}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(is_best=False)
        logger.info("Checkpoint saved")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        logger.info("Training script finished")


if __name__ == "__main__":
    main()