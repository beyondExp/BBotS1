"""
Logging Utilities for Cognitive SLM Training
Provides structured logging, metrics tracking, and monitoring
"""

import logging
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import time
import functools


def setup_logging(level: int = logging.INFO, log_dir: Optional[str] = None, 
                 log_name: str = "cognitive_slm") -> logging.Logger:
    """
    Setup comprehensive logging for the project
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        log_name: Base name for log files
        
    Returns:
        Configured logger
    """
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log_dir is specified
    if log_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        print(f"Logging to file: {log_file}")
    
    return root_logger


class MetricsTracker:
    """Track and log training metrics"""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Args:
            log_dir: Directory to save metrics
        """
        self.log_dir = log_dir
        self.metrics = {}
        self.step_metrics = []
        
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        else:
            self.metrics_file = None
        
        self.logger = logging.getLogger(__name__)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, 
                  category: str = "training"):
        """
        Log a single metric
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
            category: Metric category
        """
        timestamp = datetime.now().isoformat()
        
        metric_entry = {
            "timestamp": timestamp,
            "category": category,
            "name": name,
            "value": value,
            "step": step
        }
        
        # Store in memory
        if category not in self.metrics:
            self.metrics[category] = {}
        if name not in self.metrics[category]:
            self.metrics[category][name] = []
        
        self.metrics[category][name].append({
            "value": value,
            "step": step,
            "timestamp": timestamp
        })
        
        # Save to file
        if self.metrics_file:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric_entry) + '\n')
        
        # Log to console
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"{category}.{name}: {value:.6f}{step_str}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None,
                   category: str = "training"):
        """
        Log multiple metrics at once
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step (optional)
            category: Metric category
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step, category)
    
    def get_metric_history(self, name: str, category: str = "training") -> list:
        """Get history of a specific metric"""
        return self.metrics.get(category, {}).get(name, [])
    
    def get_latest_metric(self, name: str, category: str = "training") -> Optional[float]:
        """Get latest value of a specific metric"""
        history = self.get_metric_history(name, category)
        return history[-1]["value"] if history else None
    
    def save_summary(self):
        """Save metrics summary to file"""
        if not self.log_dir:
            return
        
        summary_file = os.path.join(self.log_dir, "metrics_summary.json")
        
        summary = {}
        for category, category_metrics in self.metrics.items():
            summary[category] = {}
            for metric_name, metric_history in category_metrics.items():
                values = [entry["value"] for entry in metric_history]
                if values:
                    summary[category][metric_name] = {
                        "count": len(values),
                        "latest": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values)
                    }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Metrics summary saved to {summary_file}")


class PerformanceMonitor:
    """Monitor performance metrics during training"""
    
    def __init__(self, metrics_tracker: MetricsTracker):
        """
        Args:
            metrics_tracker: MetricsTracker instance
        """
        self.metrics_tracker = metrics_tracker
        self.start_times = {}
        self.operation_counts = {}
        self.logger = logging.getLogger(__name__)
    
    def start_timing(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timing(self, operation: str, log_metric: bool = True) -> float:
        """
        End timing an operation and optionally log the duration
        
        Args:
            operation: Operation name
            log_metric: Whether to log the duration as a metric
            
        Returns:
            Duration in seconds
        """
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        # Track operation count
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        
        if log_metric:
            self.metrics_tracker.log_metric(
                f"{operation}_duration",
                duration,
                category="performance"
            )
            
            self.metrics_tracker.log_metric(
                f"{operation}_count",
                self.operation_counts[operation],
                category="performance"
            )
        
        return duration
    
    def time_operation(self, operation: str):
        """Decorator for timing operations"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timing(operation)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timing(operation)
            return wrapper
        return decorator


class TrainingLogger:
    """Comprehensive training logger"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Base logging directory
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(
            level=logging.INFO,
            log_dir=str(self.log_dir),
            log_name=experiment_name
        )
        
        # Setup metrics tracking
        self.metrics_tracker = MetricsTracker(str(self.log_dir))
        self.performance_monitor = PerformanceMonitor(self.metrics_tracker)
        
        # Training state
        self.training_state = {
            "start_time": None,
            "current_epoch": 0,
            "current_step": 0,
            "best_metric": None,
            "best_step": None
        }
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Logging to directory: {self.log_dir}")
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with configuration"""
        self.training_state["start_time"] = time.time()
        
        # Save configuration
        config_file = self.log_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING STARTED: {self.experiment_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration saved to: {config_file}")
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start"""
        self.training_state["current_epoch"] = epoch
        self.logger.info(f"Starting epoch {epoch}")
    
    def log_step(self, step: int, metrics: Dict[str, float], 
                learning_rate: Optional[float] = None):
        """Log training step with metrics"""
        self.training_state["current_step"] = step
        
        # Log metrics
        self.metrics_tracker.log_metrics(metrics, step)
        
        if learning_rate is not None:
            self.metrics_tracker.log_metric("learning_rate", learning_rate, step)
        
        # Check for best metric (assuming lower loss is better)
        if "loss" in metrics:
            current_loss = metrics["loss"]
            if (self.training_state["best_metric"] is None or 
                current_loss < self.training_state["best_metric"]):
                self.training_state["best_metric"] = current_loss
                self.training_state["best_step"] = step
                self.logger.info(f"New best loss: {current_loss:.6f} at step {step}")
    
    def log_evaluation(self, eval_metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics"""
        self.metrics_tracker.log_metrics(eval_metrics, step, category="evaluation")
        
        self.logger.info("Evaluation results:")
        for name, value in eval_metrics.items():
            self.logger.info(f"  {name}: {value:.6f}")
    
    def log_memory_usage(self, memory_stats: Dict[str, float], step: Optional[int] = None):
        """Log memory usage statistics"""
        self.metrics_tracker.log_metrics(memory_stats, step, category="memory")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save model info
        model_info_file = self.log_dir / "model_info.json"
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
    
    def log_training_end(self):
        """Log training completion"""
        if self.training_state["start_time"]:
            total_time = time.time() - self.training_state["start_time"]
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            self.logger.info("=" * 60)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("=" * 60)
            self.logger.info(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.logger.info(f"Final epoch: {self.training_state['current_epoch']}")
            self.logger.info(f"Final step: {self.training_state['current_step']}")
            
            if self.training_state["best_metric"]:
                self.logger.info(f"Best loss: {self.training_state['best_metric']:.6f} "
                               f"at step {self.training_state['best_step']}")
        
        # Save metrics summary
        self.metrics_tracker.save_summary()
        
        # Save training state
        state_file = self.log_dir / "training_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.training_state, f, indent=2, default=str)
    
    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance"""
        return self.performance_monitor
    
    def get_metrics_tracker(self) -> MetricsTracker:
        """Get metrics tracker instance"""
        return self.metrics_tracker


# Convenience function for quick setup
def create_training_logger(log_dir: str, experiment_name: str, 
                         config: Dict[str, Any]) -> TrainingLogger:
    """
    Create and initialize a training logger
    
    Args:
        log_dir: Base logging directory
        experiment_name: Name of the experiment
        config: Training configuration
        
    Returns:
        Configured TrainingLogger instance
    """
    logger = TrainingLogger(log_dir, experiment_name)
    logger.log_training_start(config)
    return logger