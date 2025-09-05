"""
Checkpoint management utilities for AMNet
Professional checkpoint handling with versioning and cleanup
"""

import torch
import shutil
import json
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Professional checkpoint management with automatic cleanup"""

    def __init__(self,
                 checkpoint_dir: str,
                 max_checkpoints: int = 5,
                 save_best: bool = True):

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.save_best = save_best

        # Track checkpoints
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_metric = float('-inf')

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")

    def save_checkpoint(self,
                        checkpoint: Dict,
                        is_best: bool = False,
                        epoch: Optional[int] = None) -> str:
        """Save checkpoint with automatic cleanup"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if epoch is not None:
            filename = f"checkpoint_epoch_{epoch:04d}_{timestamp}.pth"
        else:
            filename = f"checkpoint_{timestamp}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update tracking
        self.checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'timestamp': timestamp,
            'is_best': is_best
        })

        # Save best checkpoint separately
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
            self.best_checkpoint = checkpoint_path
            logger.info(f"New best checkpoint saved: {best_path}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_model.pth"
        shutil.copy2(checkpoint_path, latest_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint with validation"""

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise

    def load_best_checkpoint(self) -> Optional[Dict]:
        """Load the best saved checkpoint"""

        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            return self.load_checkpoint(best_path)

        return None

    def load_latest_checkpoint(self) -> Optional[Dict]:
        """Load the latest checkpoint"""

        latest_path = self.checkpoint_dir / "latest_model.pth"
        if latest_path.exists():
            return self.load_checkpoint(latest_path)

        return None

    def _cleanup_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""

        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by epoch (most recent first)
        self.checkpoints.sort(key=lambda x: x['epoch'] or 0, reverse=True)

        # Keep best checkpoints and recent ones
        to_keep = []
        to_remove = []

        for i, ckpt in enumerate(self.checkpoints):
            if i < self.max_checkpoints or ckpt['is_best']:
                to_keep.append(ckpt)
            else:
                to_remove.append(ckpt)

        # Remove old checkpoint files
        for ckpt in to_remove:
            try:
                ckpt['path'].unlink()
                logger.debug(f"Removed old checkpoint: {ckpt['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {ckpt['path']}: {e}")

        self.checkpoints = to_keep

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        return self.checkpoints.copy()

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """Get information about a checkpoint without loading it"""

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'best_dice': checkpoint.get('best_dice', 'unknown'),
                'model_params': sum(p.numel() for p in checkpoint.get('model_state_dict', {}).values()),
                'optimizer': 'optimizer_state_dict' in checkpoint,
                'scheduler': 'scheduler_state_dict' in checkpoint,
                'config': checkpoint.get('config', {}),
                'file_size': Path(checkpoint_path).stat().st_size / 1024 / 1024  # MB
            }

            return info

        except Exception as e:
            logger.error(f"Error reading checkpoint info: {e}")
            return {'error': str(e)}