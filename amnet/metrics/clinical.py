"""
Clinical Analysis Module for AMNet
Professional clinical validation and analysis tools
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ClinicalAnalyzer:
    """Professional clinical analysis for medical segmentation"""

    def __init__(self, organ_names: Optional[List[str]] = None):
        self.organ_names = organ_names or [
            "background", "spleen", "right_kidney", "left_kidney", "gallbladder",
            "esophagus", "liver", "stomach", "aorta", "IVC", "portal_vein",
            "pancreas", "right_adrenal", "left_adrenal", "duodenum", "bladder"
        ]

        # Clinical thresholds for organ segmentation
        self.clinical_thresholds = {
            'dice_excellent': 0.9,
            'dice_good': 0.8,
            'dice_acceptable': 0.7,
            'hd95_excellent': 2.0,  # mm
            'hd95_good': 5.0,
            'hd95_acceptable': 10.0,
            'volume_error_excellent': 5.0,  # %
            'volume_error_good': 10.0,
            'volume_error_acceptable': 20.0
        }

    def analyze_segmentation_quality(self,
                                     predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     case_ids: List[str],
                                     spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
        """
        Comprehensive clinical analysis of segmentation quality

        Args:
            predictions: Predicted segmentation masks [N, D, H, W]
            ground_truth: Ground truth masks [N, D, H, W]
            case_ids: List of case identifiers
            spacing: Voxel spacing

        Returns:
            Clinical analysis report
        """

        logger.info("Starting comprehensive clinical analysis...")

        analysis_report = {
            'overall_statistics': {},
            'per_organ_analysis': {},
            'per_case_analysis': [],
            'clinical_grades': {},
            'failure_analysis': {},
            'recommendations': []
        }

        # Per-case analysis
        case_analyses = []
        for i, case_id in enumerate(case_ids):
            case_analysis = self._analyze_single_case(
                predictions[i], ground_truth[i], case_id, spacing
            )
            case_analyses.append(case_analysis)

        analysis_report['per_case_analysis'] = case_analyses

        # Aggregate statistics
        analysis_report['overall_statistics'] = self._compute_overall_statistics(case_analyses)

        # Per-organ analysis
        analysis_report['per_organ_analysis'] = self._compute_per_organ_analysis(case_analyses)

        # Clinical grading
        analysis_report['clinical_grades'] = self._compute_clinical_grades(case_analyses)

        # Failure analysis
        analysis_report['failure_analysis'] = self._analyze_failures(case_analyses)

        # Clinical recommendations
        analysis_report['recommendations'] = self._generate_recommendations(analysis_report)

        logger.info("Clinical analysis completed")
        return analysis_report

    def _analyze_single_case(self,
                             prediction: np.ndarray,
                             ground_truth: np.ndarray,
                             case_id: str,
                             spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """Analyze single case with clinical metrics"""

        from ..metrics.segmentation import SegmentationMetrics
        metrics_calc = SegmentationMetrics(len(self.organ_names))

        case_analysis = {
            'case_id': case_id,
            'organ_metrics': {},
            'overall_quality': '',
            'clinical_concerns': [],
            'volume_analysis': {}
        }

        # Compute metrics for each organ
        for organ_id in range(1, len(self.organ_names)):
            organ_name = self.organ_names[organ_id]

            # Basic metrics
            dice = metrics_calc.dice_coefficient(prediction, ground_truth, organ_id)
            iou = metrics_calc.iou_score(prediction, ground_truth, organ_id)

            # Surface metrics
            hd95 = asd = float('inf')
            if np.sum(ground_truth == organ_id) > 0:
                hd95 = metrics_calc.hausdorff_distance_95(prediction, ground_truth, organ_id, spacing)
                asd = metrics_calc.average_surface_distance(prediction, ground_truth, organ_id, spacing)

            # Volume analysis
            pred_volume = np.sum(prediction == organ_id)
            gt_volume = np.sum(ground_truth == organ_id)

            volume_error = 0.0
            if gt_volume > 0:
                volume_error = abs(pred_volume - gt_volume) / gt_volume * 100

            case_analysis['organ_metrics'][organ_name] = {
                'dice': dice,
                'iou': iou,
                'hd95': hd95 if hd95 != float('inf') else None,
                'asd': asd if asd != float('inf') else None,
                'volume_error_percent': volume_error,
                'present_in_gt': gt_volume > 0,
                'detected': pred_volume > 0
            }

        # Overall case quality assessment
        case_analysis['overall_quality'] = self._assess_case_quality(case_analysis['organ_metrics'])

        # Clinical concerns
        case_analysis['clinical_concerns'] = self._identify_clinical_concerns(case_analysis['organ_metrics'])

        return case_analysis

    def _assess_case_quality(self, organ_metrics: Dict[str, Dict[str, float]]) -> str:
        """Assess overall case quality based on organ metrics"""

        dice_scores = []
        hd95_scores = []
        volume_errors = []

        for organ_name, metrics in organ_metrics.items():
            if metrics['present_in_gt']:
                dice_scores.append(metrics['dice'])
                if metrics['hd95'] is not None:
                    hd95_scores.append(metrics['hd95'])
                volume_errors.append(metrics['volume_error_percent'])

        if not dice_scores:
            return 'no_organs'

        mean_dice = np.mean(dice_scores)
        mean_hd95 = np.mean(hd95_scores) if hd95_scores else float('inf')
        mean_volume_error = np.mean(volume_errors)

        # Clinical grading system
        if (mean_dice >= self.clinical_thresholds['dice_excellent'] and
                mean_hd95 <= self.clinical_thresholds['hd95_excellent'] and
                mean_volume_error <= self.clinical_thresholds['volume_error_excellent']):
            return 'excellent'
        elif (mean_dice >= self.clinical_thresholds['dice_good'] and
              mean_hd95 <= self.clinical_thresholds['hd95_good'] and
              mean_volume_error <= self.clinical_thresholds['volume_error_good']):
            return 'good'
        elif (mean_dice >= self.clinical_thresholds['dice_acceptable'] and
              mean_hd95 <= self.clinical_thresholds['hd95_acceptable'] and
              mean_volume_error <= self.clinical_thresholds['volume_error_acceptable']):
            return 'acceptable'
        else:
            return 'poor'

    def _identify_clinical_concerns(self, organ_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Identify clinical concerns based on metrics"""

        concerns = []

        for organ_name, metrics in organ_metrics.items():
            if not metrics['present_in_gt']:
                continue

            # Missing organ detection
            if not metrics['detected']:
                concerns.append(f"Failed to detect {organ_name}")
                continue

            # Poor segmentation quality
            if metrics['dice'] < self.clinical_thresholds['dice_acceptable']:
                concerns.append(f"Poor {organ_name} segmentation (Dice: {metrics['dice']:.3f})")

            # Large boundary errors
            if metrics['hd95'] and metrics['hd95'] > self.clinical_thresholds['hd95_acceptable']:
                concerns.append(f"Large boundary error in {organ_name} (HD95: {metrics['hd95']:.1f}mm)")

            # Significant volume errors
            if metrics['volume_error_percent'] > self.clinical_thresholds['volume_error_acceptable']:
                concerns.append(f"Significant volume error in {organ_name} ({metrics['volume_error_percent']:.1f}%)")

        return concerns

    def _compute_overall_statistics(self, case_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute overall dataset statistics"""

        # Collect all metrics
        all_dice = []
        all_hd95 = []
        all_volume_errors = []
        quality_counts = {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0}

        for case_analysis in case_analyses:
            quality_counts[case_analysis['overall_quality']] += 1

            for organ_name, metrics in case_analysis['organ_metrics'].items():
                if metrics['present_in_gt']:
                    all_dice.append(metrics['dice'])
                    if metrics['hd95'] is not None:
                        all_hd95.append(metrics['hd95'])
                    all_volume_errors.append(metrics['volume_error_percent'])

        return {
            'total_cases': len(case_analyses),
            'mean_dice': np.mean(all_dice) if all_dice else 0,
            'std_dice': np.std(all_dice) if all_dice else 0,
            'median_dice': np.median(all_dice) if all_dice else 0,
            'mean_hd95': np.mean(all_hd95) if all_hd95 else 0,
            'std_hd95': np.std(all_hd95) if all_hd95 else 0,
            'median_hd95': np.median(all_hd95) if all_hd95 else 0,
            'mean_volume_error': np.mean(all_volume_errors) if all_volume_errors else 0,
            'std_volume_error': np.std(all_volume_errors) if all_volume_errors else 0,
            'quality_distribution': quality_counts
        }

    def _compute_per_organ_analysis(self, case_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute per-organ analysis"""

        organ_analysis = {}

        for organ_name in self.organ_names[1:]:  # Skip background
            organ_metrics = []
            detection_rate = 0
            total_present = 0

            for case_analysis in case_analyses:
                if organ_name in case_analysis['organ_metrics']:
                    metrics = case_analysis['organ_metrics'][organ_name]

                    if metrics['present_in_gt']:
                        total_present += 1
                        organ_metrics.append(metrics)
                        if metrics['detected']:
                            detection_rate += 1

            if organ_metrics:
                dice_scores = [m['dice'] for m in organ_metrics]
                hd95_scores = [m['hd95'] for m in organ_metrics if m['hd95'] is not None]
                volume_errors = [m['volume_error_percent'] for m in organ_metrics]

                organ_analysis[organ_name] = {
                    'cases_present': total_present,
                    'detection_rate': detection_rate / total_present if total_present > 0 else 0,
                    'mean_dice': np.mean(dice_scores),
                    'std_dice': np.std(dice_scores),
                    'median_dice': np.median(dice_scores),
                    'mean_hd95': np.mean(hd95_scores) if hd95_scores else None,
                    'std_hd95': np.std(hd95_scores) if hd95_scores else None,
                    'mean_volume_error': np.mean(volume_errors),
                    'std_volume_error': np.std(volume_errors)
                }

        return organ_analysis

    def _compute_clinical_grades(self, case_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute clinical performance grades"""

        grades = {
            'overall_grade': '',
            'organ_grades': {},
            'grade_distribution': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0},
            'pass_rate': 0.0
        }

        # Grade each case
        case_grades = []
        for case_analysis in case_analyses:
            quality = case_analysis['overall_quality']

            if quality == 'excellent':
                grade = 'A'
            elif quality == 'good':
                grade = 'B'
            elif quality == 'acceptable':
                grade = 'C'
            elif quality == 'poor':
                grade = 'D'
            else:
                grade = 'F'

            case_grades.append(grade)
            grades['grade_distribution'][grade] += 1

        # Overall system grade (based on percentage of acceptable or better)
        acceptable_or_better = sum(grades['grade_distribution'][g] for g in ['A', 'B', 'C'])
        total_cases = len(case_analyses)
        pass_rate = acceptable_or_better / total_cases if total_cases > 0 else 0

        grades['pass_rate'] = pass_rate

        if pass_rate >= 0.9:
            grades['overall_grade'] = 'A'
        elif pass_rate >= 0.8:
            grades['overall_grade'] = 'B'
        elif pass_rate >= 0.7:
            grades['overall_grade'] = 'C'
        elif pass_rate >= 0.6:
            grades['overall_grade'] = 'D'
        else:
            grades['overall_grade'] = 'F'

        return grades

    def _analyze_failures(self, case_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze common failure patterns"""

        failure_analysis = {
            'common_failure_organs': {},
            'failure_patterns': [],
            'challenging_cases': []
        }

        # Count failures per organ
        organ_failure_counts = {}
        for case_analysis in case_analyses:
            for concern in case_analysis['clinical_concerns']:
                for organ_name in self.organ_names[1:]:
                    if organ_name in concern:
                        if organ_name not in organ_failure_counts:
                            organ_failure_counts[organ_name] = 0
                        organ_failure_counts[organ_name] += 1

        failure_analysis['common_failure_organs'] = dict(
            sorted(organ_failure_counts.items(), key=lambda x: x[1], reverse=True)
        )

        # Identify challenging cases (poor quality with multiple concerns)
        challenging_cases = []
        for case_analysis in case_analyses:
            if (case_analysis['overall_quality'] == 'poor' and
                    len(case_analysis['clinical_concerns']) >= 3):
                challenging_cases.append({
                    'case_id': case_analysis['case_id'],
                    'concerns': case_analysis['clinical_concerns'],
                    'quality': case_analysis['overall_quality']
                })

        failure_analysis['challenging_cases'] = challenging_cases

        return failure_analysis

    def _generate_recommendations(self, analysis_report: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on analysis"""

        recommendations = []

        # Overall performance recommendations
        pass_rate = analysis_report['clinical_grades']['pass_rate']
        if pass_rate < 0.7:
            recommendations.append(
                "CRITICAL: Overall pass rate below 70%. Model needs significant improvement before clinical deployment.")
        elif pass_rate < 0.8:
            recommendations.append("WARNING: Pass rate below 80%. Additional validation and improvement recommended.")

        # Organ-specific recommendations
        failure_organs = analysis_report['failure_analysis']['common_failure_organs']
        if failure_organs:
            top_failure = list(failure_organs.keys())[0]
            recommendations.append(f"Focus improvement efforts on {top_failure} segmentation (highest failure rate).")

        # Dataset recommendations
        total_cases = analysis_report['overall_statistics']['total_cases']
        if total_cases < 100:
            recommendations.append("Expand validation dataset for more robust clinical assessment.")

        # Performance threshold recommendations
        mean_dice = analysis_report['overall_statistics']['mean_dice']
        if mean_dice < 0.8:
            recommendations.append(
                "Overall Dice score below clinical threshold. Consider model architecture improvements.")

        return recommendations

    def generate_clinical_report(self, analysis_report: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate comprehensive clinical report"""

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLINICAL ANALYSIS REPORT - AMNet Abdominal Organ Segmentation")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY:")
        report_lines.append("-" * 40)
        overall_grade = analysis_report['clinical_grades']['overall_grade']
        pass_rate = analysis_report['clinical_grades']['pass_rate']
        report_lines.append(f"Overall Clinical Grade: {overall_grade}")
        report_lines.append(f"Pass Rate: {pass_rate:.1%}")
        report_lines.append("")

        # Performance Metrics
        stats = analysis_report['overall_statistics']
        report_lines.append("PERFORMANCE METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Mean Dice Score: {stats['mean_dice']:.3f} ± {stats['std_dice']:.3f}")
        report_lines.append(f"Mean HD95: {stats['mean_hd95']:.2f} ± {stats['std_hd95']:.2f} mm")
        report_lines.append(f"Mean Volume Error: {stats['mean_volume_error']:.1f}% ± {stats['std_volume_error']:.1f}%")
        report_lines.append("")

        # Clinical Recommendations
        report_lines.append("CLINICAL RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        for i, rec in enumerate(analysis_report['recommendations'], 1):
            report_lines.append(f"{i}. {rec}")
        report_lines.append("")

        report_text = "\n".join(report_lines)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Clinical report saved to {save_path}")

        return report_text

    def plot_clinical_analysis(self, analysis_report: Dict[str, Any], save_dir: Optional[str] = None):
        """Generate clinical analysis plots"""

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Quality distribution pie chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        quality_dist = analysis_report['clinical_grades']['grade_distribution']
        labels = list(quality_dist.keys())
        sizes = list(quality_dist.values())
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']

        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Clinical Grade Distribution')

        # 2. Per-organ performance
        organ_analysis = analysis_report['per_organ_analysis']
        organ_names = list(organ_analysis.keys())
        dice_means = [organ_analysis[organ]['mean_dice'] for organ in organ_names]

        axes[0, 1].barh(organ_names, dice_means)
        axes[0, 1].set_xlabel('Mean Dice Score')
        axes[0, 1].set_title('Per-Organ Performance')
        axes[0, 1].axvline(x=0.8, color='r', linestyle='--', label='Clinical Threshold')
        axes[0, 1].legend()

        # 3. Failure analysis
        failure_organs = analysis_report['failure_analysis']['common_failure_organs']
        if failure_organs:
            organs = list(failure_organs.keys())[:10]  # Top 10
            counts = [failure_organs[organ] for organ in organs]

            axes[1, 0].bar(organs, counts)
            axes[1, 0].set_ylabel('Failure Count')
            axes[1, 0].set_title('Most Challenging Organs')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Performance distribution
        all_dice = []
        for case in analysis_report['per_case_analysis']:
            for organ_name, metrics in case['organ_metrics'].items():
                if metrics['present_in_gt']:
                    all_dice.append(metrics['dice'])

        if all_dice:
            axes[1, 1].hist(all_dice, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=np.mean(all_dice), color='r', linestyle='-', label=f'Mean: {np.mean(all_dice):.3f}')
            axes[1, 1].axvline(x=0.8, color='orange', linestyle='--', label='Clinical Threshold')
            axes[1, 1].set_xlabel('Dice Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Dice Score Distribution')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir / 'clinical_analysis.png', dpi=300, bbox_inches='tight')
            logger.info(f"Clinical analysis plots saved to {save_dir}")

        return fig