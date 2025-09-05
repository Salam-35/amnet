#!/usr/bin/env python3
"""
Generate AMNet project structure with empty files
Run: python generate_amnet_structure.py
"""

import os
from pathlib import Path


def create_structure():
    # Base directory
    base = Path("")

    # All directories to create
    dirs = [
        "amnet",
        "amnet/data",
        "amnet/models",
        "amnet/losses",
        "amnet/metrics",
        "amnet/training",
        "amnet/evaluation",
        "amnet/inference",
        "amnet/utils",
        "scripts",
        "configs",
        "tests",
        "notebooks",
        "docker",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/predictions",
        "outputs/visualizations"
    ]

    # All files to create
    files = [
        # Root files
        "README.md",
        "requirements.txt",
        "setup.py",
        ".gitignore",
        "LICENSE",

        # Main package
        "amnet/__init__.py",
        "config.py",

        # Data processing
        "amnet/data/__init__.py",
        "amnet/data/dataset.py",
        "amnet/data/transforms.py",
        "amnet/data/preprocessing.py",
        "amnet/data/dataloader.py",

        # Models
        "amnet/models/__init__.py",
        "amnet/models/amnet.py",
        "amnet/models/encoders.py",
        "amnet/models/attention.py",
        "amnet/models/constraints.py",
        "amnet/models/decoder.py",

        # Losses
        "amnet/losses/__init__.py",
        "amnet/losses/dice.py",
        "amnet/losses/focal.py",
        "amnet/losses/boundary.py",
        "amnet/losses/constraint.py",
        "amnet/losses/compound.py",

        # Metrics
        "amnet/metrics/__init__.py",
        "amnet/metrics/segmentation.py",
        "amnet/metrics/surface.py",
        "amnet/metrics/clinical.py",

        # Training
        "amnet/training/__init__.py",
        "amnet/training/trainer.py",
        "amnet/training/callbacks.py",
        "amnet/training/scheduler.py",

        # Evaluation
        "amnet/evaluation/__init__.py",
        "amnet/evaluation/evaluator.py",
        "amnet/evaluation/visualizer.py",
        "amnet/evaluation/clinical_analysis.py",

        # Inference
        "amnet/inference/__init__.py",
        "amnet/inference/predictor.py",
        "amnet/inference/postprocessing.py",

        # Utils
        "amnet/utils/__init__.py",
        "amnet/utils/logging.py",
        "amnet/utils/checkpoints.py",
        "amnet/utils/visualization.py",
        "amnet/utils/io.py",
        "amnet/utils/profiling.py",

        # Scripts
        "scripts/train.py",
        "scripts/evaluate.py",
        "scripts/predict.py",
        "scripts/prepare_data.py",

        # Configs
        "configs/default.yaml",
        "configs/training.yaml",
        "configs/inference.yaml",

        # Tests
        "tests/__init__.py",
        "tests/test_models.py",
        "tests/test_data.py",
        "tests/test_losses.py",
        "tests/test_metrics.py",

        # Notebooks
        "notebooks/data_exploration.ipynb",
        "notebooks/model_analysis.ipynb",
        "notebooks/results_visualization.ipynb",

        # Docker
        "docker/Dockerfile",
        "docker/docker-compose.yml",
    ]

    print(f"Creating AMNet project structure in: {base.absolute()}")

    # Create directories
    for dir_path in dirs:
        full_path = base / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

    # Create empty files
    for file_path in files:
        full_path = base / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch()
        print(f"📄 Created file: {file_path}")

    print(f"\n✅ AMNet project structure created successfully!")
    print(f"📂 Location: {base.absolute()}")
    print(f"📊 Created {len(dirs)} directories and {len(files)} files")

    # Create a quick reference
    with open(base / "STRUCTURE.md", "w") as f:
        f.write("# AMNet Project Structure\n\n")
        f.write("## Files to copy content to:\n\n")
        for file_path in sorted(files):
            if file_path.endswith('.py'):
                f.write(f"- `{file_path}`\n")

    print("\n📋 See STRUCTURE.md for list of Python files to populate")


if __name__ == "__main__":
    create_structure()