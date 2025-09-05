"""
Training callbacks for AMNet
Professional callbacks for monitoring and control during training
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Callback(ABC):
    """Base callback class"""

    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training"""
        pass

    def on_train_end(self, trainer) -> None:
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, epoch: int, trainer) -> None:
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        """Called at the end of each epoch"""
        pass

    def on_batch_begin(self, batch: int, trainer) -> None:
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, float], trainer) -> None:
        """Called at the end of each batch"""
        pass

class EarlyStopping(Callback):
    """Early stopping callback"""

    def __init__(self,
                 monitor: str = 'val_dice',
                 patience: int = 50,
                 min_delta: float = 1e-4,
                 mode: str = 'max',
                 restore_best_weights: bool = True):

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0

        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        current_score = logs.get(self.monitor)

        if current_score is None:
            logger.warning(f"Early stopping metric '{self.monitor}' not found")
            return

        if self.best_score is None:
            self.best_score = current_score
            self.best_weights = trainer.model.state_dict().copy()
            return

        if self.monitor_op(current_score, self.best_score + self.min_delta):
            # Improvement found
            self.best_score = current_score
            self.wait = 0

            if self.restore_best_weights:
                self.best_weights = trainer.model.state_dict().copy()

        else:
            # No improvement
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True

                if self.restore_best_weights and self.best_weights:
                    trainer.model.load_state_dict(self.best_weights)
                    logger.info(f"Restored best weights from epoch {epoch - self.wait}")

                logger.info(f"Early stopping triggered at epoch {epoch + 1}")

class ModelCheckpoint(Callback):
    """Model checkpointing callback"""

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_dice',
                 mode: str = 'max',
                 save_best_only: bool = True,
                 save_weights_only: bool = False,
                 period: int = 1):

        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period

        self.epochs_since_last_save = 0
        self.best_score = None

        if mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save < self.period:
            return

        self.epochs_since_last_save = 0
        filepath = self.filepath.parent / f"{self.filepath.stem}_epoch_{epoch:04d}{self.filepath.suffix}"

        current_score = logs.get(self.monitor)

        if self.save_best_only:
            if current_score is None:
                logger.warning(f"Checkpoint metric '{self.monitor}' not found")
                return

            if self.best_score is None or self.monitor_op(current_score, self.best_score):
                self.best_score = current_score
                self._save_checkpoint(filepath, epoch, logs, trainer)

                # Also save as best model
                best_path = self.filepath.parent / "best_model.pth"
                self._save_checkpoint(best_path, epoch, logs, trainer)

        else:
            self._save_checkpoint(filepath, epoch, logs, trainer)

    def _save_checkpoint(self, filepath: Path, epoch: int, logs: Dict[str, float], trainer):
        """Save checkpoint to file"""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if self.save_weights_only:
            torch.save(trainer.model.state_dict(), filepath)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'logs': logs,
                'best_score': self.best_score,
                'config': trainer.config.__dict__
            }
            torch.save(checkpoint, filepath)

        logger.info(f"Checkpoint saved: {filepath}")

class ReduceLROnPlateau(Callback):
    """Reduce learning rate when metric has stopped improving"""

    def __init__(self,
                 monitor: str = 'val_loss',
                 factor: float = 0.5,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 cooldown: int = 0,
                 min_lr: float = 1e-7,
                 mode: str = 'min'):

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.mode = mode

        self.wait = 0
        self.cooldown_counter = 0
        self.best_score = None

        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        current_score = logs.get(self.monitor)

        if current_score is None:
            logger.warning(f"ReduceLROnPlateau metric '{self.monitor}' not found")
            return

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0
            return

        if self.best_score is None:
            self.best_score = current_score
            return

        if self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                old_lr = trainer.optimizer.param_groups[0]['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)

                if old_lr > new_lr:
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    logger.info(f"Reduced learning rate from {old_lr:.2e} to {new_lr:.2e}")

                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

class MetricsLogger(Callback):
    """Log training metrics to file"""

    def __init__(self, log_dir: str, log_freq: int = 1):
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.metrics_history = []

        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        if (epoch + 1) % self.log_freq == 0:
            # Add epoch info
            log_entry = {
                'epoch': epoch + 1,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                **logs
            }

            self.metrics_history.append(log_entry)

            # Save to CSV
            import pandas as pd
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(self.log_dir / 'training_metrics.csv', index=False)

class LossPlotter(Callback):
    """Plot training curves"""

    def __init__(self, save_dir: str, plot_freq: int = 50):
        self.save_dir = Path(save_dir)
        self.plot_freq = plot_freq
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        # Collect metrics
        self.train_losses.append(logs.get('train_total_loss', 0))
        self.val_losses.append(logs.get('val_total_loss', 0))
        self.val_metrics.append(logs.get('val_dice', 0))

        if (epoch + 1) % self.plot_freq == 0:
            self._plot_curves(epoch + 1)

    def _plot_curves(self, epoch: int):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, label='Training Loss', color='blue')
        axes[0].plot(epochs, self.val_losses, label='Validation Loss', color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Metric curves
        axes[1].plot(epochs, self.val_metrics, label='Validation Dice', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice Score')
        axes[1].set_title('Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_curves_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

class GradientClipping(Callback):
    """Gradient clipping callback"""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []

    def on_batch_end(self, batch: int, logs: Dict[str, float], trainer) -> None:
        # Clip gradients
        total_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )

        self.grad_norms.append(float(total_norm))
        logs['grad_norm'] = float(total_norm)

class WandBLogger(Callback):
    """Weights & Biases logging callback"""

    def __init__(self, project: str, experiment_name: Optional[str] = None):
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            logger.warning("wandb not available, logging disabled")
            self.enabled = False
            return

        self.project = project
        self.experiment_name = experiment_name
        self.initialized = False

    def on_train_begin(self, trainer) -> None:
        if not self.enabled:
            return

        if not self.initialized:
            self.wandb.init(
                project=self.project,
                name=self.experiment_name,
                config=trainer.config.__dict__
            )
            self.wandb.watch(trainer.model, log="all", log_freq=100)
            self.initialized = True

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        if not self.enabled:
            return

        # Log metrics
        wandb_logs = {
            'epoch': epoch + 1,
            'learning_rate': trainer.optimizer.param_groups[0]['lr']
        }
        wandb_logs.update(logs)

        self.wandb.log(wandb_logs)

    def on_train_end(self, trainer) -> None:
        if self.enabled and self.initialized:
            self.wandb.finish()

class CallbackList:
    """Container for managing multiple callbacks"""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: Callback):
        """Add a callback to the list"""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer) -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float], trainer) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs, trainer)

    def on_batch_begin(self, batch: int, trainer) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch, trainer)

    def on_batch_end(self, batch: int, logs: Dict[str, float], trainer) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs, trainer)