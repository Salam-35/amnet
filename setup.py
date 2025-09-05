"""
AMNet Package Setup
Professional Python package configuration
"""

from setuptools import setup, find_packages
import os
from pathlib import Path


# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py"""
    init_file = Path(__file__).parent / "amnet" / "__init__.py"
    if init_file.exists():
        with open(init_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"


# Read README for long description
def get_long_description():
    """Get long description from README.md"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, encoding="utf-8") as f:
            return f.read()
    return "AMNet: Anatomically-aware Multi-scale Network for Abdominal Organ Segmentation"


# Core requirements
install_requires = [
    # Deep Learning Framework
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "torchaudio>=2.0.0",

    # Scientific Computing
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "scikit-image>=0.19.0",

    # Medical Image Processing
    "nibabel>=5.0.0",
    "SimpleITK>=2.2.0",
    "pydicom>=2.3.0",

    # Data Processing
    "pandas>=1.3.0",
    "h5py>=3.7.0",

    # Visualization
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.10.0",

    # Progress and Logging
    "tqdm>=4.64.0",
    "rich>=12.0.0",

    # Configuration
    "pyyaml>=6.0",
    "omegaconf>=2.2.0",

    # Utilities
    "pathlib2>=2.3.0",
    "pillow>=9.0.0",
    "imageio>=2.20.0",

    # Memory and Performance
    "psutil>=5.9.0",
]

# Development dependencies
dev_requires = [
    # Testing
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.8.0",

    # Code Quality
    "black>=22.0.0",
    "flake8>=5.0.0",
    "isort>=5.10.0",
    "mypy>=0.991",

    # Documentation
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.19.0",

    # Development Tools
    "pre-commit>=2.20.0",
    "jupyter>=1.0.0",
    "jupyterlab>=3.4.0",
]

# Optional dependencies for different use cases
extras_require = {
    # Weights & Biases logging
    "wandb": [
        "wandb>=0.13.0",
    ],

    # TensorBoard logging
    "tensorboard": [
        "tensorboard>=2.10.0",
        "tensorboardX>=2.5.0",
    ],

    # Advanced visualizations
    "visualization": [
        "mayavi>=4.8.0",
        "vtk>=9.2.0",
        "pyvista>=0.37.0",
    ],

    # ONNX export
    "onnx": [
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ],

    # Multi-processing
    "multiprocessing": [
        "joblib>=1.2.0",
        "dask>=2022.8.0",
    ],

    # Cloud storage
    "cloud": [
        "boto3>=1.24.0",
        "google-cloud-storage>=2.5.0",
        "azure-storage-blob>=12.14.0",
    ],

    # Development dependencies
    "dev": dev_requires,

    # All optional dependencies
    "all": [
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
        "tensorboardX>=2.5.0",
        "mayavi>=4.8.0",
        "vtk>=9.2.0",
        "pyvista>=0.37.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
        "joblib>=1.2.0",
        "dask>=2022.8.0",
        "boto3>=1.24.0",
        "google-cloud-storage>=2.5.0",
        "azure-storage-blob>=12.14.0",
    ],
}

# Entry points for command-line tools
entry_points = {
    "console_scripts": [
        "amnet-train=amnet.scripts.train:main",
        "amnet-evaluate=amnet.scripts.evaluate:main",
        "amnet-predict=amnet.scripts.predict:main",
        "amnet-prepare-data=amnet.scripts.prepare_data:main",
    ],
}

# Package classifiers
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Image Processing",
]

# Keywords for PyPI search
keywords = [
    "medical imaging",
    "segmentation",
    "deep learning",
    "pytorch",
    "abdominal organs",
    "CT segmentation",
    "computer vision",
    "healthcare AI",
    "medical AI",
]

setup(
    name="amnet",
    version=get_version(),
    author="AMNet Development Team",
    author_email="contact@amnet-research.org",
    description="Anatomically-aware Multi-scale Network for Abdominal Organ Segmentation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/amnet-research/amnet",
    project_urls={
        "Documentation": "https://amnet.readthedocs.io",
        "Bug Reports": "https://github.com/amnet-research/amnet/issues",
        "Source": "https://github.com/amnet-research/amnet",
        "Changelog": "https://github.com/amnet-research/amnet/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    package_data={
        "amnet": [
            "configs/*.yaml",
            "configs/*.yml",
            "data/*.json",
            "models/pretrained/*.pth",
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    python_requires=">=3.8",
    classifiers=classifiers,
    keywords=" ".join(keywords),
    license="MIT",
    zip_safe=False,

    # Additional metadata
    platforms=["any"],

    # Testing configuration
    test_suite="tests",

    # Options for different environments
    options={
        "bdist_wheel": {"universal": False},
    },
)