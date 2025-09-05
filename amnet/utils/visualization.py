"""
Visualization utilities for AMNet
Advanced medical image and results visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Define organ colors for consistent visualization
ORGAN_COLORS = {
    0: [0, 0, 0],  # background - black
    1: [255, 0, 0],  # spleen - red
    2: [0, 255, 0],  # right kidney - green
    3: [0, 0, 255],  # left kidney - blue
    4: [255, 255, 0],  # gallbladder - yellow
    5: [255, 0, 255],  # esophagus - magenta
    6: [0, 255, 255],  # liver - cyan
    7: [128, 0, 0],  # stomach - maroon
    8: [0, 128, 0],  # aorta - dark green
    9: [0, 0, 128],  # IVC - navy
    10: [128, 128, 0],  # portal vein - olive
    11: [128, 0, 128],  # pancreas - purple
    12: [0, 128, 128],  # right adrenal - teal
    13: [192, 192, 192],  # left adrenal - silver
    14: [255, 165, 0],  # duodenum - orange
    15: [255, 192, 203]  # bladder - pink
}


class MedicalVisualizer:
    """Professional medical image visualization"""

    def __init__(self, organ_names: Optional[List[str]] = None):

        self.organ_names = organ_names or [
            "Background", "Spleen", "Right Kidney", "Left Kidney", "Gallbladder",
            "Esophagus", "Liver", "Stomach", "Aorta", "IVC", "Portal Vein",
            "Pancreas", "Right Adrenal", "Left Adrenal", "Duodenum", "Bladder"
        ]

        # Create colormap
        colors = [[c / 255.0 for c in ORGAN_COLORS[i]] for i in range(len(self.organ_names))]
        self.cmap = ListedColormap(colors)

    def plot_ct_slices(self,
                       volume: np.ndarray,
                       mask: Optional[np.ndarray] = None,
                       prediction: Optional[np.ndarray] = None,
                       slice_indices: Optional[List[int]] = None,
                       save_path: Optional[str] = None,
                       title: str = "CT Volume"):
        """Plot CT slices with optional masks and predictions"""

        if slice_indices is None:
            # Select evenly spaced slices
            depth = volume.shape[0]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

        n_slices = len(slice_indices)
        n_cols = 3 if mask is not None and prediction is not None else (
            2 if mask is not None or prediction is not None else 1)

        fig, axes = plt.subplots(n_slices, n_cols, figsize=(5 * n_cols, 4 * n_slices))

        if n_slices == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

        for i, slice_idx in enumerate(slice_indices):
            col = 0

            # Plot CT slice
            ax = axes[i][col] if n_cols > 1 else axes[i]
            ax.imshow(volume[slice_idx], cmap='gray', aspect='equal')
            ax.set_title(f'CT Slice {slice_idx}')
            ax.axis('off')
            col += 1

            # Plot ground truth mask
            if mask is not None:
                ax = axes[i][col]
                ax.imshow(volume[slice_idx], cmap='gray', aspect='equal', alpha=0.7)
                ax.imshow(mask[slice_idx], cmap=self.cmap, alpha=0.5, vmin=0, vmax=len(self.organ_names) - 1)
                ax.set_title(f'Ground Truth {slice_idx}')
                ax.axis('off')
                col += 1

            # Plot prediction
            if prediction is not None:
                ax = axes[i][col]
                ax.imshow(volume[slice_idx], cmap='gray', aspect='equal', alpha=0.7)
                ax.imshow(prediction[slice_idx], cmap=self.cmap, alpha=0.5, vmin=0, vmax=len(self.organ_names) - 1)
                ax.set_title(f'Prediction {slice_idx}')
                ax.axis('off')

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")

        return fig

    def plot_segmentation_comparison(self,
                                     volume: np.ndarray,
                                     ground_truth: np.ndarray,
                                     prediction: np.ndarray,
                                     case_id: str = "Case",
                                     save_path: Optional[str] = None):
        """Plot side-by-side comparison of segmentation results"""

        depth = volume.shape[0]
        slice_indices = [depth // 4, depth // 2, 3 * depth // 4]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))

        for i, slice_idx in enumerate(slice_indices):
            # Original CT
            axes[i, 0].imshow(volume[slice_idx], cmap='gray')
            axes[i, 0].set_title(f'CT Slice {slice_idx}')
            axes[i, 0].axis('off')

            # Ground truth overlay
            axes[i, 1].imshow(volume[slice_idx], cmap='gray', alpha=0.7)
            axes[i, 1].imshow(ground_truth[slice_idx], cmap=self.cmap, alpha=0.5, vmin=0, vmax=15)
            axes[i, 1].set_title(f'Ground Truth')
            axes[i, 1].axis('off')

            # Prediction overlay
            axes[i, 2].imshow(volume[slice_idx], cmap='gray', alpha=0.7)
            axes[i, 2].imshow(prediction[slice_idx], cmap=self.cmap, alpha=0.5, vmin=0, vmax=15)
            axes[i, 2].set_title(f'Prediction')
            axes[i, 2].axis('off')

        plt.suptitle(f'Segmentation Results - {case_id}', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")

        return fig

    def plot_attention_maps(self,
                            volume: np.ndarray,
                            attention_weights: Dict[str, torch.Tensor],
                            save_path: Optional[str] = None):
        """Visualize attention maps"""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        slice_idx = volume.shape[0] // 2
        ct_slice = volume[slice_idx]

        # Plot original CT slice
        axes[0, 0].imshow(ct_slice, cmap='gray')
        axes[0, 0].set_title('Original CT')
        axes[0, 0].axis('off')

        # Plot different attention maps
        attention_types = ['axial_weights', 'coronal_weights', 'sagittal_weights', '2d_to_3d', '3d_to_2d']

        for i, att_type in enumerate(attention_types[:5]):
            if att_type in attention_weights:
                row = i // 3
                col = (i % 3) + (1 if row == 0 else 0)

                att_map = attention_weights[att_type]
                if isinstance(att_map, torch.Tensor):
                    att_map = att_map.cpu().numpy()

                # Handle different attention map shapes
                if att_map.ndim > 2:
                    att_map = np.mean(att_map, axis=tuple(range(att_map.ndim - 2)))

                # Resize to match CT slice if needed
                if att_map.shape != ct_slice.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = [ct_slice.shape[i] / att_map.shape[i] for i in range(2)]
                    att_map = zoom(att_map, zoom_factors)

                # Plot attention overlay
                axes[row, col].imshow(ct_slice, cmap='gray', alpha=0.7)
                im = axes[row, col].imshow(att_map, cmap='jet', alpha=0.5)
                axes[row, col].set_title(f'{att_type.replace("_", " ").title()}')
                axes[row, col].axis('off')

                # Add colorbar
                plt.colorbar(im, ax=axes[row, col], fraction=0.046)

        plt.suptitle('Attention Visualization', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")

        return fig

    def plot_metrics_comparison(self,
                                metrics_dict: Dict[str, Dict[str, float]],
                                save_path: Optional[str] = None):
        """Plot comparison of different metrics"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Dice scores
        if 'dice' in metrics_dict:
            organs = list(metrics_dict['dice'].keys())
            dice_scores = list(metrics_dict['dice'].values())

            axes[0, 0].barh(organs, dice_scores)
            axes[0, 0].set_xlabel('Dice Score')
            axes[0, 0].set_title('Dice Scores by Organ')
            axes[0, 0].set_xlim(0, 1)

        # HD95 scores
        if 'hd95' in metrics_dict:
            organs = list(metrics_dict['hd95'].keys())
            hd95_scores = list(metrics_dict['hd95'].values())

            axes[0, 1].barh(organs, hd95_scores)
            axes[0, 1].set_xlabel('HD95 (mm)')
            axes[0, 1].set_title('Hausdorff Distance 95% by Organ')

        # Training curves (if available)
        if 'train_loss' in metrics_dict:
            epochs = range(len(metrics_dict['train_loss']))
            axes[1, 0].plot(epochs, metrics_dict['train_loss'], label='Training Loss')
            if 'val_loss' in metrics_dict:
                axes[1, 0].plot(epochs, metrics_dict['val_loss'], label='Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Curves')
            axes[1, 0].legend()

        # Volume comparison
        if 'volume_error' in metrics_dict:
            organs = list(metrics_dict['volume_error'].keys())
            vol_errors = list(metrics_dict['volume_error'].values())

            axes[1, 1].bar(organs, vol_errors)
            axes[1, 1].set_ylabel('Volume Error (%)')
            axes[1, 1].set_title('Volume Estimation Error')
            axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")

        return fig

    def create_colorbar_legend(self, save_path: Optional[str] = None):
        """Create a colorbar legend for organ classes"""

        fig, ax = plt.subplots(figsize=(2, 8))

        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=plt.Normalize(vmin=0, vmax=len(self.organ_names) - 1))
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, ticks=range(len(self.organ_names)))
        cbar.set_ticklabels(self.organ_names)
        cbar.ax.tick_params(labelsize=10)

        ax.remove()
        plt.title('Organ Labels', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Colorbar legend saved to {save_path}")

        return fig


def plot_segmentation_results(volume: np.ndarray,
                              ground_truth: np.ndarray,
                              prediction: np.ndarray,
                              case_id: str = "Case",
                              save_path: Optional[str] = None):
    """Convenience function for plotting segmentation results"""

    visualizer = MedicalVisualizer()
    return visualizer.plot_segmentation_comparison(
        volume, ground_truth, prediction, case_id, save_path
    )


def plot_attention_maps(volume: np.ndarray,
                        attention_weights: Dict[str, torch.Tensor],
                        save_path: Optional[str] = None):
    """Convenience function for plotting attention maps"""

    visualizer = MedicalVisualizer()
    return visualizer.plot_attention_maps(volume, attention_weights, save_path)