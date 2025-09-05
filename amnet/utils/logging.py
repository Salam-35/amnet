"""
Logging utilities for AMNet
Professional logging with rich formatting and metrics tracking
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import torch

console = Console()


def setup_logging(log_dir: Optional[Path] = None,
                  level: int = logging.INFO,
                  log_to_file: bool = True) -> logging.Logger:
    """Setup comprehensive logging system"""

    # Create log directory if specified
    if log_dir and log_to_file:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Rich console handler for terminal output
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        markup=True
    )
    console_handler.setLevel(level)

    # Formatter for console
    console_formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler for persistent logging
    if log_dir and log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"amnet_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level)

        # Detailed formatter for file
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")

    # Set specific logger levels
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return root_logger


class MetricsLogger:
    """Professional metrics logging and tracking"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir
        self.metrics_history = []
        self.current_metrics = {}

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_file = log_dir / "metrics.jsonl"
        else:
            self.metrics_file = None

    def log_metrics(self,
                    metrics: Dict[str, Any],
                    step: int,
                    epoch: Optional[int] = None,
                    phase: str = "train"):
        """Log metrics for a specific step/epoch"""

        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics
        }

        # Store in memory
        self.metrics_history.append(entry)
        self.current_metrics[phase] = metrics

        # Save to file
        if self.metrics_file:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')

    def get_best_metric(self, metric_name: str, phase: str = "val") -> Dict[str, Any]:
        """Get best value for a specific metric"""

        best_entry = None
        best_value = float('-inf') if 'loss' not in metric_name.lower() else float('inf')

        for entry in self.metrics_history:
            if entry['phase'] != phase:
                continue

            if metric_name in entry['metrics']:
                value = entry['metrics'][metric_name]

                is_better = (value > best_value if 'loss' not in metric_name.lower()
                             else value < best_value)

                if is_better:
                    best_value = value
                    best_entry = entry

        return best_entry

    def print_metrics_summary(self):
        """Print beautiful metrics summary"""

        table = Table(title="📈 Training Metrics Summary", show_header=True)
        table.add_column("Phase", style="cyan")
        table.add_column("Latest", style="green")
        table.add_column("Best", style="yellow")
        table.add_column("Best Epoch", style="blue")

        # Key metrics to display
        key_metrics = ['total_loss', 'dice_loss', 'mean_dice', 'val_dice']

        for metric in key_metrics:
            for phase in ['train', 'val']:
                if phase in self.current_metrics:
                    current = self.current_metrics[phase].get(metric, 'N/A')
                    best_entry = self.get_best_metric(metric, phase)

                    if best_entry:
                        best_val = best_entry['metrics'][metric]
                        best_epoch = best_entry.get('epoch', 'N/A')

                        table.add_row(
                            f"{phase.title()} {metric}",
                            f"{current:.4f}" if isinstance(current, (int, float)) else str(current),
                            f"{best_val:.4f}",
                            str(best_epoch)
                        )

        console.print(table)


def log_metrics_table(metrics: Dict[str, Dict[str, float]],
                      title: str = "Metrics",
                      console: Console = console):
    """Log metrics in a beautiful table format"""

    table = Table(title=title, show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=12)
    table.add_column("Unit", style="white", width=8)

    for category, category_metrics in metrics.items():
        # Add category header
        table.add_row(f"[bold]{category.upper()}[/bold]", "", "")

        # Add metrics in category
        for metric_name, value in category_metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)

            # Determine unit based on metric name
            unit = ""
            if 'time' in metric_name.lower():
                unit = "s"
            elif 'loss' in metric_name.lower() or 'dice' in metric_name.lower():
                unit = ""
            elif 'hd95' in metric_name.lower() or 'asd' in metric_name.lower():
                unit = "mm"

            table.add_row(f"  {metric_name}", formatted_value, unit)

    console.print(table)


def log_system_info():
    """Log system information for debugging"""

    table = Table(title="🖥️  System Information", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Details", style="green")

    # Python version
    import sys
    table.add_row("Python Version", sys.version.split()[0])

    # PyTorch version
    table.add_row("PyTorch Version", torch.__version__)

    # CUDA information
    if torch.cuda.is_available():
        table.add_row("CUDA Available", "✅ Yes")
        table.add_row("CUDA Version", torch.version.cuda or "Unknown")
        table.add_row("GPU Count", str(torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            table.add_row(f"GPU {i}", f"{gpu_name} ({gpu_memory:.1f} GB)")
    else:
        table.add_row("CUDA Available", "❌ No")

    # Memory information
    import psutil
    memory = psutil.virtual_memory()
    table.add_row("System RAM", f"{memory.total / 1e9:.1f} GB")
    table.add_row("Available RAM", f"{memory.available / 1e9:.1f} GB")

    console.print(table)


class ProgressTracker:
    """Track training progress with rich progress bars"""

    def __init__(self):
        self.progress = None
        self.tasks = {}

    def start_epoch(self, epoch: int, total_epochs: int, total_batches: int):
        """Start tracking a new epoch"""

        if self.progress is None:
            self.progress = Progress(console=console)
            self.progress.start()

        # Create epoch task
        epoch_task = self.progress.add_task(
            f"[green]Epoch {epoch}/{total_epochs}",
            total=total_batches
        )

        self.tasks['epoch'] = epoch_task

        return epoch_task

    def update_batch(self, batch_loss: float, batch_idx: int):
        """Update progress for current batch"""

        if 'epoch' in self.tasks:
            self.progress.update(
                self.tasks['epoch'],
                advance=1,
                description=f"[green]Batch {batch_idx} | Loss: {batch_loss:.4f}"
            )

    def finish_epoch(self):
        """Finish current epoch tracking"""

        if 'epoch' in self.tasks:
            self.progress.remove_task(self.tasks['epoch'])
            del self.tasks['epoch']

    def stop(self):
        """Stop all progress tracking"""

        if self.progress:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()