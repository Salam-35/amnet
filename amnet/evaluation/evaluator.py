"""
AMNet Model Evaluator
Comprehensive evaluation with clinical metrics and visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.amnet import AMNet
from ..metrics.segmentation import SegmentationMetrics
from ..utils.visualization import plot_segmentation_results, plot_attention_maps

logger = logging.getLogger(__name__)
console = Console()

class AMNetEvaluator:
    """Comprehensive model evaluator with clinical analysis"""

    def __init__(self,
                 model: AMNet,
                 config,
                 device: str = "cuda"):

        self.model = model
        self.config = config
        self.device = device

        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()

        # Initialize metrics calculator
        self.metrics_calculator = SegmentationMetrics(
            num_classes=config.num_classes,
            ignore_background=True
        )

        # Results storage
        self.results = {
            'case_metrics': [],
            'class_metrics': {},
            'overall_metrics': {},
            'failed_cases': []
        }

        logger.info("AMNet Evaluator initialized")

    def evaluate_dataset(self,
                        test_loader,
                        save_predictions: bool = False,
                        output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Evaluate model on complete dataset

        Args:
            test_loader: DataLoader for test data
            save_predictions: Whether to save prediction volumes
            output_dir: Directory to save results

        Returns:
            Summary metrics dictionary
        """

        console.print("[bold blue]Starting Comprehensive Evaluation[/bold blue]")

        all_case_metrics = []
        prediction_times = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):

                # Move data to device
                images = batch['image'].to(self.device)  # [B, 1, D, H, W]
                masks = batch['mask'].to(self.device)    # [B, D, H, W]
                case_ids = batch['case_id']

                # Measure inference time
                start_time = time.time()

                # Forward pass
                outputs = self.model(images)
                predictions = outputs['predictions']  # [B, C, D, H, W]

                inference_time = time.time() - start_time
                prediction_times.append(inference_time)

                # Convert predictions to class predictions
                pred_classes = torch.argmax(predictions, dim=1)  # [B, D, H, W]

                # Compute metrics for each case in batch
                for b in range(images.size(0)):
                    case_id = case_ids[b]
                    pred_case = pred_classes[b].cpu().numpy()
                    target_case = masks[b].cpu().numpy()

                    # Compute comprehensive metrics
                    case_metrics = self._evaluate_single_case(
                        pred_case, target_case, case_id
                    )
                    case_metrics['inference_time'] = inference_time / images.size(0)
                    all_case_metrics.append(case_metrics)

                    # Save predictions if requested
                    if save_predictions and output_dir:
                        self._save_prediction(
                            pred_case, case_id, output_dir / "predictions"
                        )

        # Compile results
        self._compile_evaluation_results(all_case_metrics, prediction_times)

        # Generate evaluation report
        if output_dir:
            self._generate_evaluation_report(output_dir)

        # Print summary
        self._print_evaluation_summary()

        return self.results['overall_metrics']

    def _evaluate_single_case(self,
                             prediction: np.ndarray,
                             target: np.ndarray,
                             case_id: str) -> Dict[str, float]:
        """Evaluate single case with all metrics"""

        case_metrics = {
            'case_id': case_id,
            'dice_scores': {},
            'iou_scores': {},
            'hd95_scores': {},
            'asd_scores': {},
            'volume_errors': {}
        }

        # Compute metrics for each organ class
        for class_id in range(1, self.config.num_classes):  # Skip background
            class_name = self.metrics_calculator.class_names[class_id]

            try:
                # Dice coefficient
                dice = self.metrics_calculator.dice_coefficient(
                    prediction, target, class_id
                )
                case_metrics['dice_scores'][class_name] = dice

                # IoU score
                iou = self.metrics_calculator.iou_score(
                    prediction, target, class_id
                )
                case_metrics['iou_scores'][class_name] = iou

                # Surface metrics (if class present)
                if np.sum(target == class_id) > 0:
                    hd95 = self.metrics_calculator.hausdorff_distance_95(
                        prediction, target, class_id
                    )
                    asd = self.metrics_calculator.average_surface_distance(
                        prediction, target, class_id
                    )

                    case_metrics['hd95_scores'][class_name] = hd95
                    case_metrics['asd_scores'][class_name] = asd

                # Volume error
                pred_volume = np.sum(prediction == class_id)
                target_volume = np.sum(target == class_id)

                if target_volume > 0:
                    volume_error = abs(pred_volume - target_volume) / target_volume * 100
                    case_metrics['volume_errors'][class_name] = volume_error

            except Exception as e:
                logger.warning(f"Error computing metrics for {class_name} in {case_id}: {e}")
                case_metrics['dice_scores'][class_name] = 0.0
                case_metrics['iou_scores'][class_name] = 0.0

        return case_metrics

    def _compile_evaluation_results(self,
                                   all_case_metrics: List[Dict],
                                   prediction_times: List[float]):
        """Compile comprehensive evaluation results"""

        # Per-class statistics
        class_metrics = {}

        for metric_type in ['dice_scores', 'iou_scores', 'hd95_scores', 'asd_scores', 'volume_errors']:
            class_metrics[metric_type] = {}

            # Collect scores for each class
            for class_name in self.metrics_calculator.class_names[1:]:  # Skip background
                scores = []
                for case_metrics in all_case_metrics:
                    if class_name in case_metrics[metric_type]:
                        score = case_metrics[metric_type][class_name]
                        if score != float('inf') and not np.isnan(score):
                            scores.append(score)

                if scores:
                    class_metrics[metric_type][class_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'median': np.median(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'count': len(scores)
                    }

        # Overall statistics
        overall_metrics = {}

        # Mean Dice across all classes
        all_dice = []
        for case_metrics in all_case_metrics:
            case_dice = list(case_metrics['dice_scores'].values())
            if case_dice:
                all_dice.extend(case_dice)

        overall_metrics['mean_dice'] = np.mean(all_dice) if all_dice else 0.0
        overall_metrics['std_dice'] = np.std(all_dice) if all_dice else 0.0

        # Mean inference time
        overall_metrics['mean_inference_time'] = np.mean(prediction_times)
        overall_metrics['std_inference_time'] = np.std(prediction_times)

        # Store results
        self.results['case_metrics'] = all_case_metrics
        self.results['class_metrics'] = class_metrics
        self.results['overall_metrics'] = overall_metrics

    def _print_evaluation_summary(self):
        """Print beautiful evaluation summary"""

        # Overall metrics table
        table = Table(title="🏥 AMNet Evaluation Summary", show_header=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)
        table.add_column("Unit", style="white", width=10)

        overall = self.results['overall_metrics']

        table.add_row("Mean Dice Score", f"{overall['mean_dice']:.4f}", "")
        table.add_row("Dice Std", f"{overall['std_dice