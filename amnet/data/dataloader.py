"""
Professional DataLoader implementation for AMNet
Advanced data loading with memory management and caching
"""

import torch
from torch.utils.data import DataLoader, Sampler, Dataset
import numpy as np
import logging
from typing import Iterator, Optional, List, Dict, Any
import random
from pathlib import Path
import threading
import queue
import time

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler):
    """Balanced batch sampler for medical image segmentation"""

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = True,
                 organ_presence_balance: bool = True):

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.organ_presence_balance = organ_presence_balance

        # Analyze dataset for organ presence
        if organ_presence_balance:
            self._analyze_organ_presence()

    def _analyze_organ_presence(self):
        """Analyze which organs are present in each sample"""
        logger.info("Analyzing organ presence for balanced sampling...")

        self.organ_indices = {i: [] for i in range(1, 16)}  # Skip background

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            mask = sample['mask'].numpy()

            unique_organs = np.unique(mask)
            for organ_id in unique_organs:
                if 1 <= organ_id <= 15:
                    self.organ_indices[organ_id].append(idx)

        logger.info("Organ presence analysis completed")
        for organ_id, indices in self.organ_indices.items():
            logger.info(f"  Organ {organ_id}: {len(indices)} samples")

    def __iter__(self) -> Iterator[List[int]]:
        if self.organ_presence_balance:
            return self._balanced_iter()
        else:
            return self._random_iter()

    def _random_iter(self) -> Iterator[List[int]]:
        """Standard random sampling"""
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def _balanced_iter(self) -> Iterator[List[int]]:
        """Balanced sampling considering organ presence"""
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)

        for i in range(0, len(all_indices), self.batch_size):
            if i + self.batch_size > len(all_indices) and self.drop_last:
                break

            batch_indices = all_indices[i:i + self.batch_size]

            # Try to balance organ representation in batch
            if len(batch_indices) == self.batch_size:
                batch_indices = self._balance_batch(batch_indices)

            yield batch_indices

    def _balance_batch(self, batch_indices: List[int]) -> List[int]:
        """Attempt to balance organ representation in batch"""
        # This is a simplified balancing - in practice you might want more sophisticated logic
        return batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class CachedDataLoader:
    """Data loader with intelligent caching"""

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 cache_size: int = 100,
                 prefetch_factor: int = 2):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.prefetch_factor = prefetch_factor

        # Cache management
        self.cache = {}
        self.cache_order = []
        self.cache_lock = threading.RLock()

        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor
        )

    def _manage_cache(self, key: str, data: Any):
        """Manage LRU cache"""
        with self.cache_lock:
            if key in self.cache:
                # Move to end (most recent)
                self.cache_order.remove(key)
                self.cache_order.append(key)
            else:
                # Add new item
                if len(self.cache) >= self.cache_size:
                    # Remove oldest item
                    oldest = self.cache_order.pop(0)
                    del self.cache[oldest]

                self.cache[key] = data
                self.cache_order.append(key)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class AdaptiveBatchSizeLoader:
    """Data loader that adapts batch size based on GPU memory"""

    def __init__(self,
                 dataset: Dataset,
                 initial_batch_size: int = 4,
                 max_batch_size: int = 16,
                 memory_threshold: float = 0.9):

        self.dataset = dataset
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold

        self.dataloader = None
        self._create_dataloader()

    def _create_dataloader(self):
        """Create dataloader with current batch size"""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.current_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def _check_memory_usage(self) -> float:
        """Check GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0

    def adapt_batch_size(self, oom_occurred: bool = False):
        """Adapt batch size based on memory usage"""
        if oom_occurred:
            # Decrease batch size
            new_batch_size = max(1, self.current_batch_size // 2)
            logger.warning(f"OOM detected, reducing batch size: {self.current_batch_size} -> {new_batch_size}")
            self.current_batch_size = new_batch_size
            self._create_dataloader()

        else:
            memory_usage = self._check_memory_usage()

            if memory_usage < 0.7 and self.current_batch_size < self.max_batch_size:
                # Increase batch size
                new_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                logger.info(f"Low memory usage, increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
                self._create_dataloader()

            elif memory_usage > self.memory_threshold:
                # Decrease batch size
                new_batch_size = max(1, self.current_batch_size - 1)
                logger.warning(
                    f"High memory usage, decreasing batch size: {self.current_batch_size} -> {new_batch_size}")
                self.current_batch_size = new_batch_size
                self._create_dataloader()

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def create_amnet_dataloader(dataset: Dataset,
                            batch_size: int = 4,
                            shuffle: bool = True,
                            num_workers: int = 8,
                            pin_memory: bool = True,
                            drop_last: bool = True,
                            use_balanced_sampling: bool = False,
                            **kwargs) -> DataLoader:
    """
    Create optimized DataLoader for AMNet training

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        use_balanced_sampling: Use balanced batch sampling
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """

    # Create sampler if balanced sampling is requested
    sampler = None
    if use_balanced_sampling and shuffle:
        sampler = BalancedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last
        )
        shuffle = False  # Disable shuffle when using custom sampler
        batch_size = 1  # Sampler handles batching

    # Configure DataLoader
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'drop_last': drop_last,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2,
        **kwargs
    }

    if sampler is not None:
        dataloader_kwargs['batch_sampler'] = sampler
        # Remove conflicting arguments
        for key in ['batch_size', 'shuffle', 'drop_last']:
            dataloader_kwargs.pop(key, None)

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    logger.info(f"DataLoader created:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Num workers: {num_workers}")
    logger.info(f"  Balanced sampling: {use_balanced_sampling}")
    logger.info(f"  Total batches: {len(dataloader)}")

    return dataloader


class MultiScaleDataLoader:
    """Multi-scale training data loader"""

    def __init__(self,
                 dataset: Dataset,
                 scales: List[Tuple[int, int, int]] = [(96, 144, 144), (128, 192, 192), (160, 240, 240)],
                 batch_sizes: List[int] = [6, 4, 2],
                 scale_weights: List[float] = [0.3, 0.5, 0.2]):

        self.dataset = dataset
        self.scales = scales
        self.batch_sizes = batch_sizes
        self.scale_weights = scale_weights

        assert len(scales) == len(batch_sizes) == len(scale_weights)

        # Create data loaders for each scale
        self.dataloaders = []
        for scale, batch_size in zip(scales, batch_sizes):
            # Create dataset copy with different target size
            scale_dataset = self._create_scale_dataset(scale)
            dataloader = DataLoader(
                scale_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            self.dataloaders.append(dataloader)

    def _create_scale_dataset(self, target_size: Tuple[int, int, int]):
        """Create dataset with specific target size"""
        # This would need to be implemented based on your dataset structure
        # For now, return the original dataset
        return self.dataset

    def __iter__(self):
        """Iterate through multi-scale batches"""
        iterators = [iter(dl) for dl in self.dataloaders]

        while True:
            # Choose scale based on weights
            scale_idx = np.random.choice(len(self.scales), p=self.scale_weights)

            try:
                batch = next(iterators[scale_idx])
                yield batch, self.scales[scale_idx]
            except StopIteration:
                # Restart iterator
                iterators[scale_idx] = iter(self.dataloaders[scale_idx])
                batch = next(iterators[scale_idx])
                yield batch, self.scales[scale_idx]