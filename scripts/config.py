
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = "AMNet"
    num_classes: int = 16
    input_size: Tuple[int, int, int] = (128, 128, 128)  # D, H, W

    # 2D Branch
    encoder_2d_name: str = "convnext_v2_base"
    feature_dim_2d: int = 1024
    encoder_2d_depths: List[int] = field(default_factory=lambda: [3, 3, 27, 3])

    # 3D Branch
    encoder_3d_name: str = "resnet3d_50"
    feature_dim_3d: int = 2048
    encoder_3d_layers: List[int] = field(default_factory=lambda: [3, 4, 6, 3])

    # Fusion
    fusion_dim: int = 512
    scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Attention
    attention_heads: int = 8
    attention_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.00001
    max_epochs: int = 100
    early_stopping: int = 50

    # Optimizer
    optimizer: str = "AdamW"
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: str = "CosineAnnealingLR"
    eta_min: float = 1e-7

    # Gradient
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class LossConfig:
    """Loss function configuration"""
    alpha_dice: float = 1.0
    beta_focal: float = 0.5
    gamma_boundary: float = 0.3
    delta_constraint: float = 0.2

    # Focal loss parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


@dataclass
class DataConfig:
    """Data configuration"""
    root_dir: str = "/media/salam/projects/amos22/data/amos/"
    num_workers: int = 8
    pin_memory: bool = False
    cache_data: bool = False

    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Preprocessing
    window_level: float = 40.0
    window_width: float = 400.0
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0)


@dataclass
class AugmentationConfig:
    """Data augmentation configuration"""
    probability: float = 0.8

    # Geometric
    rotation_range: Tuple[float, float] = (-10.0, 10.0)
    scaling_range: Tuple[float, float] = (0.9, 1.1)
    flip_probability: float = 0.5

    # Intensity
    noise_std: float = 0.1
    brightness_range: Tuple[float, float] = (-0.1, 0.1)
    contrast_range: Tuple[float, float] = (0.8, 1.2)

    # Advanced
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    elastic_probability: float = 0.3

    blur_sigma_range: Tuple[float, float] = (0.5, 2.0)
    blur_probability: float = 0.3


@dataclass
class PathsConfig:
    """Paths configuration"""
    output_dir: str = "../outputs"
    checkpoint_dir: str = "../outputs/checkpoints"
    log_dir: str = "../outputs/logs"
    predictions_dir: str = "../outputs/predictions"
    visualizations_dir: str = "../outputs/visualizations"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_interval: int = 10
    val_interval: int = 50
    save_checkpoint_interval: int = 100
    visualize_predictions: bool = True

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "AMNet-AbdominalSegmentation"
    wandb_entity: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str] = field(default_factory=lambda: ["dice", "iou", "hd95", "asd", "volume_error"])
    save_predictions: bool = True
    generate_plots: bool = True
    slice_indices: Optional[List[int]] = None
    colormap: str = "custom"


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 1
    use_tta: bool = False
    save_format: str = "nifti"

    # Post-processing
    remove_small_objects: bool = True
    min_object_size: int = 100


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_memory_usage: float = 0.9


class Config:
    """Complete AMNet configuration"""

    def __init__(self, config_path: Optional[str] = None):
        # Initialize all sub-configs
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.loss = LossConfig()
        self.data = DataConfig()
        self.augmentation = AugmentationConfig()
        self.paths = PathsConfig()
        self.logging = LoggingConfig()
        self.evaluation = EvaluationConfig()
        self.inference = InferenceConfig()
        self.hardware = HardwareConfig()

        # Organ labels
        self.organ_labels = {
            0: "background", 1: "spleen", 2: "right_kidney", 3: "left_kidney",
            4: "gallbladder", 5: "esophagus", 6: "liver", 7: "stomach",
            8: "aorta", 9: "IVC", 10: "portal_vein", 11: "pancreas",
            12: "right_adrenal", 13: "left_adrenal", 14: "duodenum", 15: "bladder"
        }

        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)

        # Apply environment overrides
        self._apply_environment_overrides()

        # Validate configuration
        self._validate_config()

        # Set derived properties
        self._set_derived_properties()

    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        self._update_from_dict(config_dict)
        logger.info(f"Configuration loaded from {config_path}")

    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_config = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
            elif section_name == "organ_labels":
                self.organ_labels = section_data
            else:
                logger.warning(f"Unknown config section: {section_name}")

    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'AMNET_DATA_ROOT': ('data', 'root_dir'),
            'AMNET_OUTPUT_DIR': ('paths', 'output_dir'),
            'AMNET_BATCH_SIZE': ('training', 'batch_size'),
            'AMNET_LEARNING_RATE': ('training', 'learning_rate'),
            'AMNET_MAX_EPOCHS': ('training', 'max_epochs'),
            'AMNET_NUM_WORKERS': ('data', 'num_workers'),
            'AMNET_DEVICE': ('hardware', 'device'),
            'WANDB_PROJECT': ('logging', 'wandb_project'),
            'WANDB_ENTITY': ('logging', 'wandb_entity'),
        }

        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                # Type conversion
                if key in ['batch_size', 'max_epochs', 'num_workers']:
                    value = int(value)
                elif key in ['learning_rate']:
                    value = float(value)
                elif key in ['use_wandb']:
                    value = value.lower() in ('true', '1', 'yes')

                section_config = getattr(self, section)
                setattr(section_config, key, value)
                logger.info(f"Environment override: {section}.{key} = {value}")

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Validate model config
        if self.model.num_classes < 2:
            errors.append("num_classes must be >= 2")

        if any(s <= 0 for s in self.model.input_size):
            errors.append("input_size dimensions must be positive")

        # Validate training config
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")

        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        if self.training.max_epochs <= 0:
            errors.append("max_epochs must be positive")

        # Validate data config
        if not (0 < self.data.train_ratio < 1):
            errors.append("train_ratio must be between 0 and 1")

        if not (0 < self.data.val_ratio < 1):
            errors.append("val_ratio must be between 0 and 1")

        if (self.data.train_ratio + self.data.val_ratio + self.data.test_ratio) != 1.0:
            errors.append("train/val/test ratios must sum to 1.0")

        # Validate hardware config
        if self.hardware.device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.hardware.device = 'cpu'

        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")

        logger.info("Configuration validation passed")

    def _set_derived_properties(self):
        """Set derived properties"""
        # Set convenience properties for backward compatibility
        self.num_classes = self.model.num_classes
        self.input_size = self.model.input_size
        self.feature_dim_2d = self.model.feature_dim_2d
        self.feature_dim_3d = self.model.feature_dim_3d
        self.fusion_dim = self.model.fusion_dim
        self.scales = self.model.scales

        self.batch_size = self.training.batch_size
        self.learning_rate = self.training.learning_rate
        self.max_epochs = self.training.max_epochs
        self.early_stopping = self.training.early_stopping

        self.alpha_dice = self.loss.alpha_dice
        self.beta_focal = self.loss.beta_focal
        self.gamma_boundary = self.loss.gamma_boundary
        self.delta_constraint = self.loss.delta_constraint

        self.data_root = self.data.root_dir
        self.num_workers = self.data.num_workers
        self.pin_memory = self.data.pin_memory
        self.cache_data = self.data.cache_data

        self.output_dir = self.paths.output_dir
        self.checkpoint_dir = self.paths.checkpoint_dir
        self.log_dir = self.paths.log_dir

        self.device = self.hardware.device
        self.log_interval = self.logging.log_interval

    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {config_path}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'name': self.model.name,
                'num_classes': self.model.num_classes,
                'input_size': list(self.model.input_size),
                'encoder_2d': {
                    'name': self.model.encoder_2d_name,
                    'feature_dim': self.model.feature_dim_2d,
                    'depths': self.model.encoder_2d_depths,
                },
                'encoder_3d': {
                    'name': self.model.encoder_3d_name,
                    'feature_dim': self.model.feature_dim_3d,
                    'layers': self.model.encoder_3d_layers,
                },
                'fusion_dim': self.model.fusion_dim,
                'scales': self.model.scales,
                'attention': {
                    'heads': self.model.attention_heads,
                    'dropout': self.model.attention_dropout,
                }
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'max_epochs': self.training.max_epochs,
                'early_stopping': self.training.early_stopping,
                'optimizer': self.training.optimizer,
                'betas': list(self.training.betas),
                'scheduler': self.training.scheduler,
                'eta_min': self.training.eta_min,
            },
            'loss': {
                'alpha_dice': self.loss.alpha_dice,
                'beta_focal': self.loss.beta_focal,
                'gamma_boundary': self.loss.gamma_boundary,
                'delta_constraint': self.loss.delta_constraint,
                'focal_alpha': self.loss.focal_alpha,
                'focal_gamma': self.loss.focal_gamma,
            },
            'data': {
                'root_dir': self.data.root_dir,
                'num_workers': self.data.num_workers,
                'pin_memory': self.data.pin_memory,
                'cache_data': self.data.cache_data,
                'train_ratio': self.data.train_ratio,
                'val_ratio': self.data.val_ratio,
                'test_ratio': self.data.test_ratio,
                'window_level': self.data.window_level,
                'window_width': self.data.window_width,
                'target_spacing': list(self.data.target_spacing),
            },
            'augmentation': {
                'probability': self.augmentation.probability,
                'rotation_range': list(self.augmentation.rotation_range),
                'scaling_range': list(self.augmentation.scaling_range),
                'flip_probability': self.augmentation.flip_probability,
                'noise_std': self.augmentation.noise_std,
                'brightness_range': list(self.augmentation.brightness_range),
                'contrast_range': list(self.augmentation.contrast_range),
                'elastic_alpha': self.augmentation.elastic_alpha,
                'elastic_sigma': self.augmentation.elastic_sigma,
                'elastic_probability': self.augmentation.elastic_probability,
                'blur_sigma_range': list(self.augmentation.blur_sigma_range),
                'blur_probability': self.augmentation.blur_probability,
            },
            'paths': {
                'output_dir': self.paths.output_dir,
                'checkpoint_dir': self.paths.checkpoint_dir,
                'log_dir': self.paths.log_dir,
                'predictions_dir': self.paths.predictions_dir,
                'visualizations_dir': self.paths.visualizations_dir,
            },
            'logging': {
                'level': self.logging.level,
                'log_interval': self.logging.log_interval,
                'val_interval': self.logging.val_interval,
                'save_checkpoint_interval': self.logging.save_checkpoint_interval,
                'visualize_predictions': self.logging.visualize_predictions,
                'use_wandb': self.logging.use_wandb,
                'wandb_project': self.logging.wandb_project,
                'wandb_entity': self.logging.wandb_entity,
            },
            'evaluation': {
                'metrics': self.evaluation.metrics,
                'save_predictions': self.evaluation.save_predictions,
                'generate_plots': self.evaluation.generate_plots,
                'slice_indices': self.evaluation.slice_indices,
                'colormap': self.evaluation.colormap,
            },
            'inference': {
                'batch_size': self.inference.batch_size,
                'use_tta': self.inference.use_tta,
                'save_format': self.inference.save_format,
                'remove_small_objects': self.inference.remove_small_objects,
                'min_object_size': self.inference.min_object_size,
            },
            'hardware': {
                'device': self.hardware.device,
                'mixed_precision': self.hardware.mixed_precision,
                'gradient_accumulation_steps': self.hardware.gradient_accumulation_steps,
                'max_memory_usage': self.hardware.max_memory_usage,
            },
            'organ_labels': self.organ_labels
        }

    def update(self, **kwargs):
        """Update configuration with keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config attribute: {key}")

    def get_experiment_name(self) -> str:
        """Generate experiment name based on configuration"""
        return f"AMNet_{self.model.name}_bs{self.training.batch_size}_lr{self.training.learning_rate}"

    def __str__(self) -> str:
        """String representation of configuration"""
        lines = ["AMNet Configuration:"]
        lines.append(f"  Model: {self.model.name} ({self.model.num_classes} classes)")
        lines.append(f"  Input size: {self.model.input_size}")
        lines.append(f"  Batch size: {self.training.batch_size}")
        lines.append(f"  Learning rate: {self.training.learning_rate}")
        lines.append(f"  Max epochs: {self.training.max_epochs}")
        lines.append(f"  Device: {self.hardware.device}")
        lines.append(f"  Data root: {self.data.root_dir}")
        lines.append(f"  Output dir: {self.paths.output_dir}")
        return "\n".join(lines)