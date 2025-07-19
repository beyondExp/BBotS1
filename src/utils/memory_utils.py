"""
Memory Optimization Utilities for RTX 3090 Ti
Provides memory management, monitoring, and optimization tools
"""

import torch
import torch.nn as nn
import gc
import psutil
import time
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """Memory optimization utilities for training on RTX 3090 Ti"""
    
    def __init__(self, target_memory_usage: float = 0.95):
        """
        Args:
            target_memory_usage: Target GPU memory usage (0.0 to 1.0)
        """
        self.target_memory_usage = target_memory_usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_memory = self._get_max_memory()
        
        # Initialize CUDA settings for RTX 3090 Ti
        if torch.cuda.is_available():
            self._setup_cuda_optimizations()
    
    def _get_max_memory(self) -> int:
        """Get maximum GPU memory in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    def _setup_cuda_optimizations(self):
        """Setup CUDA optimizations for RTX 3090 Ti"""
        # Enable TF32 for better performance on RTX 3090 Ti
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        logger.info("CUDA optimizations applied for RTX 3090 Ti")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            free = self.max_memory - reserved
            
            stats.update({
                'gpu_allocated_gb': allocated / (1024**3),
                'gpu_reserved_gb': reserved / (1024**3),
                'gpu_free_gb': free / (1024**3),
                'gpu_utilization': reserved / self.max_memory,
                'gpu_max_gb': self.max_memory / (1024**3)
            })
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        stats.update({
            'cpu_used_gb': cpu_memory.used / (1024**3),
            'cpu_available_gb': cpu_memory.available / (1024**3),
            'cpu_utilization': cpu_memory.percent / 100.0
        })
        
        return stats
    
    def print_memory_stats(self):
        """Print formatted memory statistics"""
        stats = self.get_memory_stats()
        
        print("=" * 60)
        print("Memory Statistics")
        print("=" * 60)
        
        if 'gpu_allocated_gb' in stats:
            print(f"GPU Memory:")
            print(f"  Allocated: {stats['gpu_allocated_gb']:.2f} GB")
            print(f"  Reserved:  {stats['gpu_reserved_gb']:.2f} GB")
            print(f"  Free:      {stats['gpu_free_gb']:.2f} GB")
            print(f"  Max:       {stats['gpu_max_gb']:.2f} GB")
            print(f"  Usage:     {stats['gpu_utilization']:.1%}")
        
        print(f"CPU Memory:")
        print(f"  Used:      {stats['cpu_used_gb']:.2f} GB")
        print(f"  Available: {stats['cpu_available_gb']:.2f} GB")
        print(f"  Usage:     {stats['cpu_utilization']:.1%}")
        print("=" * 60)
    
    def cleanup_memory(self):
        """Cleanup GPU and CPU memory"""
        # Clear Python garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory cleanup completed")
    
    def optimize_batch_size(self, model: nn.Module, sample_input: torch.Tensor, 
                          max_batch_size: int = 32) -> int:
        """
        Find optimal batch size for given model and input
        
        Args:
            model: The model to test
            sample_input: Sample input tensor
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size
        """
        model.eval()
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                # Create batch
                batch_input = sample_input.repeat(batch_size, 1)
                
                # Test forward pass
                with torch.no_grad():
                    _ = model(batch_input)
                
                # Check memory usage
                stats = self.get_memory_stats()
                if stats.get('gpu_utilization', 0) > self.target_memory_usage:
                    break
                
                optimal_batch_size = batch_size
                self.cleanup_memory()
                
            except torch.cuda.OutOfMemoryError:
                break
            except Exception as e:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                break
        
        self.cleanup_memory()
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "operation"):
        """Context manager for monitoring memory usage during operations"""
        start_stats = self.get_memory_stats()
        start_time = time.time()
        
        logger.info(f"Starting {operation_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_stats = self.get_memory_stats()
            
            duration = end_time - start_time
            
            if 'gpu_allocated_gb' in start_stats:
                memory_delta = end_stats['gpu_allocated_gb'] - start_stats['gpu_allocated_gb']
                logger.info(f"{operation_name} completed in {duration:.2f}s, "
                          f"memory change: {memory_delta:+.2f} GB")
            else:
                logger.info(f"{operation_name} completed in {duration:.2f}s")
    
    def setup_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Setup gradient checkpointing for memory efficiency"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
        
        return model
    
    def estimate_model_memory(self, model: nn.Module) -> Dict[str, float]:
        """Estimate model memory usage"""
        param_memory = 0
        buffer_memory = 0
        
        for param in model.parameters():
            param_memory += param.numel() * param.element_size()
        
        for buffer in model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()
        
        total_memory = param_memory + buffer_memory
        
        return {
            'parameters_gb': param_memory / (1024**3),
            'buffers_gb': buffer_memory / (1024**3),
            'total_gb': total_memory / (1024**3),
            'parameters_count': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }


class DynamicBatchSizer:
    """Dynamic batch sizing based on available memory"""
    
    def __init__(self, memory_optimizer: MemoryOptimizer, 
                 safety_margin: float = 0.1):
        """
        Args:
            memory_optimizer: MemoryOptimizer instance
            safety_margin: Safety margin for memory usage (0.0 to 1.0)
        """
        self.memory_optimizer = memory_optimizer
        self.safety_margin = safety_margin
        self.current_batch_size = 1
        self.max_successful_batch_size = 1
        
    def adjust_batch_size(self, current_memory_usage: float) -> int:
        """
        Adjust batch size based on current memory usage
        
        Args:
            current_memory_usage: Current memory utilization (0.0 to 1.0)
            
        Returns:
            New batch size
        """
        target_usage = self.memory_optimizer.target_memory_usage - self.safety_margin
        
        if current_memory_usage < target_usage * 0.8:
            # Increase batch size if memory usage is low
            new_batch_size = min(
                self.current_batch_size + 1,
                self.max_successful_batch_size * 2
            )
        elif current_memory_usage > target_usage:
            # Decrease batch size if memory usage is high
            new_batch_size = max(1, self.current_batch_size - 1)
        else:
            # Keep current batch size
            new_batch_size = self.current_batch_size
        
        return new_batch_size
    
    def update_success(self, batch_size: int):
        """Update maximum successful batch size"""
        self.max_successful_batch_size = max(self.max_successful_batch_size, batch_size)
        self.current_batch_size = batch_size


class MemoryEfficientDataLoader:
    """Memory-efficient data loader for large datasets"""
    
    def __init__(self, dataset, batch_size: int, memory_optimizer: MemoryOptimizer,
                 max_memory_usage: float = 0.8):
        """
        Args:
            dataset: Dataset to load
            batch_size: Initial batch size
            memory_optimizer: MemoryOptimizer instance
            max_memory_usage: Maximum memory usage before reducing batch size
        """
        self.dataset = dataset
        self.memory_optimizer = memory_optimizer
        self.max_memory_usage = max_memory_usage
        self.batch_sizer = DynamicBatchSizer(memory_optimizer)
        self.current_batch_size = batch_size
    
    def __iter__(self):
        """Iterate over dataset with dynamic batch sizing"""
        dataset_size = len(self.dataset)
        current_idx = 0
        
        while current_idx < dataset_size:
            # Check memory usage
            stats = self.memory_optimizer.get_memory_stats()
            current_usage = stats.get('gpu_utilization', 0)
            
            # Adjust batch size if needed
            if current_usage > self.max_memory_usage:
                self.current_batch_size = max(1, self.current_batch_size // 2)
                self.memory_optimizer.cleanup_memory()
            
            # Create batch
            end_idx = min(current_idx + self.current_batch_size, dataset_size)
            batch = [self.dataset[i] for i in range(current_idx, end_idx)]
            
            yield batch
            
            # Update batch size tracking
            self.batch_sizer.update_success(len(batch))
            current_idx = end_idx


def setup_memory_optimization(model: nn.Module, 
                            target_memory_usage: float = 0.95) -> Tuple[nn.Module, MemoryOptimizer]:
    """
    Setup comprehensive memory optimization for a model
    
    Args:
        model: Model to optimize
        target_memory_usage: Target GPU memory usage
        
    Returns:
        Tuple of (optimized_model, memory_optimizer)
    """
    memory_optimizer = MemoryOptimizer(target_memory_usage)
    
    # Setup gradient checkpointing
    model = memory_optimizer.setup_gradient_checkpointing(model)
    
    # Print model memory estimation
    model_stats = memory_optimizer.estimate_model_memory(model)
    logger.info(f"Model memory estimation: {model_stats['total_gb']:.2f} GB")
    logger.info(f"Trainable parameters: {model_stats['trainable_parameters']:,}")
    
    # Initial memory cleanup
    memory_optimizer.cleanup_memory()
    
    return model, memory_optimizer


# Import for convenience
import os