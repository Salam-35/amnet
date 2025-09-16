"""
AMNet Training Orchestrator
Professional training loop with comprehensive logging and monitoring
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
import wandb

from ..models.amnet import AMNet
from ..losses.compound import CompoundLoss
from ..metrics.segmentation import SegmentationMetrics
from ..utils.logging import setup_logging, log_metrics_table
from ..utils.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)
console = Console()

class AMNetTrainer:
    """Professional trainer for AMNet with comprehensive monitoring"""

    def __init__(self,
                 config,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 model: Optional[AMNet] = None,
                 use_wandb: bool = True):

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_wandb = use_wandb

        # Initialize model
        self.model = model or AMNet(config)
        self.model = self.model.to(config.device)

        # Loss function and metrics
        self.criterion = CompoundLoss(
            alpha=config.alpha_dice,
            beta=config.beta_focal,
            gamma=config.gamma_boundary,
            delta=config.delta_constraint
        )
        self.metrics_calculator = SegmentationMetrics(config.num_classes)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.training.betas
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_epochs,
            eta_min=1e-7
        )

        # Training state
        self.current_epoch = 0
        self.best_dice = 0.0
        self.best_epoch = 0
        self.early_stopping_counter = 0

        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=5
        )

        # Wandb initialization
        if self.use_wandb:
            wandb.init(
                project="AMNet-AbdominalSegmentation",
                config=config.__dict__,
                name=f"AMNet-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            wandb.watch(self.model, log="all", log_freq=100)

        logger.info(f"Trainer initialized - Model: {self.model.count_parameters():,} parameters")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with detailed logging"""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'dice_loss': 0.0,
            'focal_loss': 0.0,
            'boundary_loss': 0.0,
            'constraint_loss': 0.0
        }

        num_batches = len(self.train_loader)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            train_task = progress.add_task(
                f"[green]Epoch {self.current_epoch + 1}/{self.config.training.max_epochs}",
                total=num_batches
            )

            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                images = batch['image'].to(self.config.device)  # [B, 1, D, H, W]
                masks = batch['mask'].to(self.config.device)    # [B, D, H, W]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                predictions = outputs['predictions']
                constraint_loss = outputs['constraint_loss']

                # Compute loss
                loss_dict = self.criterion(predictions, masks, constraint_loss)
                total_loss = loss_dict['total_loss']

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()

                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()

                # Update progress
                progress.update(
                    train_task,
                    advance=1,
                    description=f"[green]Epoch {self.current_epoch + 1} | Loss: {total_loss.item():.4f}"
                )

                # Log batch metrics
                if batch_idx % self.config.log_interval == 0:
                    self._log_batch_metrics(batch_idx, loss_dict, num_batches)

        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate_epoch(self) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Validate model and compute comprehensive metrics"""
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'dice_loss': 0.0,
            'focal_loss': 0.0,
            'boundary_loss': 0.0,
            'constraint_loss': 0.0
        }

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                images = batch['image'].to(self.config.device)
                masks = batch['mask'].to(self.config.device)

                # Forward pass
                outputs = self.model(images)
                predictions = outputs['predictions']
                constraint_loss = outputs['constraint_loss']

                # Compute loss
                loss_dict = self.criterion(predictions, masks, constraint_loss)

                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()

                # Store predictions and targets for metrics
                pred_classes = torch.argmax(predictions, dim=1)
                all_predictions.append(pred_classes.cpu())
                all_targets.append(masks.cpu())

        # Average losses
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Compute segmentation metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Debug: Check prediction distribution
        unique_preds, pred_counts = torch.unique(all_predictions, return_counts=True)
        unique_targets, target_counts = torch.unique(all_targets, return_counts=True)
        self.logger.info(f"Predictions - Classes: {unique_preds.tolist()}, Counts: {pred_counts.tolist()}")
        self.logger.info(f"Targets - Classes: {unique_targets.tolist()}, Counts: {target_counts.tolist()}")

        # Convert to proper format for metrics
        predictions_np = all_predictions.numpy()
        targets_np = all_targets.numpy()

        # Calculate comprehensive metrics
        detailed_metrics = {}
        for b in range(min(10, predictions_np.shape[0])):  # Sample subset for efficiency
            batch_metrics = self.metrics_calculator.compute_all_metrics(
                torch.from_numpy(predictions_np[b:b+1]).unsqueeze(1),  # Add channel dim
                torch.from_numpy(targets_np[b:b+1])
            )

            # Accumulate metrics
            for metric_type, class_scores in batch_metrics.items():
                if metric_type not in detailed_metrics:
                    detailed_metrics[metric_type] = {}

                for class_name, score in class_scores.items():
                    if class_name not in detailed_metrics[metric_type]:
                        detailed_metrics[metric_type][class_name] = []
                    detailed_metrics[metric_type][class_name].append(score)

        # Average detailed metrics
        final_metrics = {}
        for metric_type, class_scores in detailed_metrics.items():
            final_metrics[metric_type] = {}
            for class_name, scores in class_scores.items():
                valid_scores = [s for s in scores if s != float('inf')]
                final_metrics[metric_type][class_name] = np.mean(valid_scores) if valid_scores else 0.0

        return epoch_losses, final_metrics

    def _log_batch_metrics(self, batch_idx: int, loss_dict: Dict[str, torch.Tensor], num_batches: int):
        """Log batch-level metrics"""
        if self.use_wandb:
            wandb.log({
                f"batch/total_loss": loss_dict['total_loss'].item(),
                f"batch/dice_loss": loss_dict['dice_loss'].item(),
                f"batch/focal_loss": loss_dict['focal_loss'].item(),
                f"batch/boundary_loss": loss_dict['boundary_loss'].item(),
                f"batch/constraint_loss": loss_dict['constraint_loss'].item(),
                f"batch/learning_rate": self.optimizer.param_groups[0]['lr'],
                f"batch/step": self.current_epoch * num_batches + batch_idx
            })

    def _log_epoch_metrics(self,
                          train_losses: Dict[str, float],
                          val_losses: Dict[str, float],
                          val_metrics: Dict[str, Dict[str, float]]):
        """Log comprehensive epoch metrics with beautiful formatting"""

        # Create metrics table
        table = Table(title=f"Epoch {self.current_epoch + 1} Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Train", style="magenta")
        table.add_column("Validation", style="green")
        table.add_column("Best", style="yellow")

        # Loss metrics
        for loss_name in train_losses:
            table.add_row(
                loss_name.replace('_', ' ').title(),
                f"{train_losses[loss_name]:.4f}",
                f"{val_losses[loss_name]:.4f}",
                "-"
            )

        # Segmentation metrics
        for metric_type in ['dice', 'iou', 'hd95', 'asd']:
            if metric_type in val_metrics:
                mean_score = val_metrics[metric_type].get('mean', 0.0)
                table.add_row(
                    metric_type.upper(),
                    "-",
                    f"{mean_score:.4f}",
                    f"{self.best_dice:.4f}" if metric_type == 'dice' else "-"
                )

        console.print(table)

        # Wandb logging
        if self.use_wandb:
            log_dict = {
                "epoch": self.current_epoch + 1,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }

            # Add train losses
            for key, value in train_losses.items():
                log_dict[f"train/{key}"] = value

            # Add validation losses
            for key, value in val_losses.items():
                log_dict[f"val/{key}"] = value

            # Add detailed metrics
            for metric_type, class_scores in val_metrics.items():
                for class_name, score in class_scores.items():
                    log_dict[f"val/{metric_type}/{class_name}"] = score

            wandb.log(log_dict)

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch,
            'config': self.config.__dict__
        }

        self.checkpoint_manager.save_checkpoint(checkpoint, is_best)

    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        console.print(f"[bold green]Starting AMNet Training[/bold green]")
        console.print(f"Device: {self.config.device}")
        console.print(f"Model Parameters: {self.model.count_parameters():,}")

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_hd95': [],
            'val_asd': []
        }

        for epoch in range(self.config.training.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Training phase
            train_losses = self.train_epoch()

            # Validation phase
            val_losses, val_metrics = self.validate_epoch()

            # Scheduler step
            self.scheduler.step()

            # Check for improvement
            current_dice = val_metrics.get('dice', {}).get('mean', 0.0)
            is_best = current_dice > self.best_dice

            if is_best:
                self.best_dice = current_dice
                self.best_epoch = epoch
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            # Save checkpoint
            if (epoch + 1) % 50 == 0 or is_best:
                self.save_checkpoint(is_best)

            # Log metrics
            self._log_epoch_metrics(train_losses, val_losses, val_metrics)

            # Update training history
            training_history['train_loss'].append(train_losses['total_loss'])
            training_history['val_loss'].append(val_losses['total_loss'])
            training_history['val_dice'].append(current_dice)
            training_history['val_hd95'].append(val_metrics.get('hd95', {}).get('mean', 0.0))
            training_history['val_asd'].append(val_metrics.get('asd', {}).get('mean', 0.0))

            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                break

            # Epoch timing
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

        # Final logging
        console.print(f"[bold green]Training completed![/bold green]")
        console.print(f"Best Dice Score: {self.best_dice:.4f} at epoch {self.best_epoch + 1}")

        if self.use_wandb:
            wandb.finish()

        return training_history