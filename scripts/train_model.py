#!/usr/bin/env python3
"""
Cognitive SLM Training Script
Optimized for RTX 3090 Ti (24GB VRAM)
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb
from accelerate import Accelerator
from datasets import Dataset
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.cognitive_architecture import CognitiveLanguageModel, CognitiveConfig
from src.data.cognitive_data_schemas import CognitiveTrainingExample, DatasetConfiguration
from src.utils.memory_utils import MemoryOptimizer
from src.utils.logging_utils import setup_logging


class CognitiveTrainer(Trainer):
    """Custom trainer for cognitive models with specialized loss functions"""
    
    def __init__(self, *args, **kwargs):
        self.cognitive_loss_weights = kwargs.pop('cognitive_loss_weights', {})
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with cognitive components"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Primary language modeling loss
        lm_loss = outputs.loss
        
        # Initialize total loss
        total_loss = lm_loss
        
        # Add cognitive auxiliary losses if available
        if hasattr(outputs, 'reasoning_outputs') and outputs.reasoning_outputs:
            reasoning_loss = self._compute_reasoning_loss(outputs.reasoning_outputs)
            total_loss += self.cognitive_loss_weights.get('reasoning', 0.1) * reasoning_loss
        
        if hasattr(outputs, 'state_outputs') and outputs.state_outputs:
            state_loss = self._compute_state_loss(outputs.state_outputs)
            total_loss += self.cognitive_loss_weights.get('state', 0.05) * state_loss
        
        if hasattr(outputs, 'tool_outputs') and outputs.tool_outputs:
            tool_loss = self._compute_tool_loss(outputs.tool_outputs)
            total_loss += self.cognitive_loss_weights.get('tool', 0.1) * tool_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _compute_reasoning_loss(self, reasoning_outputs):
        """Compute reasoning-specific loss"""
        # Placeholder for reasoning loss computation
        # In practice, this would compute loss on reasoning step predictions
        return torch.tensor(0.0, device=self.model.device)
    
    def _compute_state_loss(self, state_outputs):
        """Compute state transition loss"""
        # Placeholder for state transition loss
        return torch.tensor(0.0, device=self.model.device)
    
    def _compute_tool_loss(self, tool_outputs):
        """Compute tool calling loss"""
        # Placeholder for tool calling loss
        return torch.tensor(0.0, device=self.model.device)


def setup_model_and_tokenizer(config_path: str, use_quantization: bool = True):
    """Setup model and tokenizer with memory optimizations"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = CognitiveConfig(**config_dict['model_config'])
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Setup model with quantization if requested
    if use_quantization:
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = CognitiveLanguageModel.from_pretrained(
            None,  # We're training from scratch
            config=model_config,
            quantization_config=bnb_config if use_quantization else None,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if use_quantization:
            model = prepare_model_for_kbit_training(model)
    else:
        model = CognitiveLanguageModel(model_config)
    
    return model, tokenizer, model_config


def setup_lora(model, config_dict: Dict[str, Any]):
    """Setup LoRA for parameter-efficient fine-tuning"""
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    return model


def create_data_collator(tokenizer, max_length: int = 2048):
    """Create data collator for cognitive training"""
    
    class CognitiveDataCollator:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __call__(self, examples):
            # Extract text from examples
            texts = []
            for example in examples:
                if isinstance(example, dict) and 'text' in example:
                    texts.append(example['text'])
                elif isinstance(example, str):
                    texts.append(example)
                else:
                    texts.append(str(example))
            
            # Tokenize
            batch = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for language modeling)
            batch["labels"] = batch["input_ids"].clone()
            
            return batch
    
    return CognitiveDataCollator(tokenizer, max_length)


def load_dataset(data_path: str, tokenizer, max_examples: Optional[int] = None):
    """Load and process cognitive training dataset"""
    
    # Load data (implement according to your data format)
    # This is a placeholder - replace with actual data loading
    examples = []
    
    # Example: Load from JSON files
    if os.path.isfile(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
            for item in data:
                if isinstance(item, dict) and 'messages' in item:
                    # Convert conversation to text
                    text = ""
                    for message in item['messages']:
                        role = message.get('role', 'unknown')
                        content = message.get('content', '')
                        text += f"<|{role}|>\n{content}\n<|endoftext|>\n"
                    examples.append({'text': text})
    
    # Limit examples if specified
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
    
    # Create dataset
    dataset = Dataset.from_list(examples)
    
    return dataset


def setup_training_arguments(config_dict: Dict[str, Any], output_dir: str):
    """Setup training arguments optimized for RTX 3090 Ti"""
    
    training_config = config_dict.get('training_config', {})
    hardware_config = config_dict.get('hardware_optimizations', {})
    
    # Calculate effective batch size
    batch_size = hardware_config.get('batch_size_config', {}).get('base_batch_size', 4)
    gradient_accumulation = hardware_config.get('batch_size_config', {}).get('gradient_accumulation_steps', 8)
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Batch size and gradient accumulation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        
        # Learning rate and schedule
        learning_rate=training_config.get('learning_rate', 5e-5),
        lr_scheduler_type=training_config.get('lr_scheduler', 'cosine'),
        warmup_steps=training_config.get('warmup_steps', 1000),
        max_steps=training_config.get('max_steps', 50000),
        
        # Optimization
        optim="adamw_torch",
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=hardware_config.get('batch_size_config', {}).get('max_grad_norm', 1.0),
        
        # Memory optimization
        fp16=False,
        bf16=hardware_config.get('mixed_precision') == 'bf16',
        gradient_checkpointing=hardware_config.get('gradient_checkpointing', True),
        dataloader_pin_memory=hardware_config.get('memory_management', {}).get('pin_memory', True),
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps", 
        save_steps=1000,
        save_total_limit=3,
        
        # Logging
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        report_to=["wandb", "tensorboard"],
        
        # Performance
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Reproducibility
        seed=42,
        data_seed=42,
    )
    
    return args


def setup_wandb(config_dict: Dict[str, Any], model_name: str):
    """Setup Weights & Biases logging"""
    
    logging_config = config_dict.get('logging_config', {})
    
    wandb.init(
        project=logging_config.get('wandb_project', 'cognitive-slm'),
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config_dict,
        tags=["cognitive", "slm", "rtx3090ti"]
    )


def monitor_memory_usage():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        memory_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Free: {memory_free:.2f}GB")
        
        # Log to wandb if available
        try:
            wandb.log({
                "gpu_memory_allocated_gb": memory_allocated,
                "gpu_memory_reserved_gb": memory_reserved,
                "gpu_memory_free_gb": memory_free
            })
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Train Cognitive SLM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--use-quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--max-examples", type=int, help="Maximum number of training examples")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data")
    parser.add_argument("--resume-from", type=str, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup model and tokenizer
    logger.info("Setting up model and tokenizer...")
    model, tokenizer, model_config = setup_model_and_tokenizer(
        args.config, 
        use_quantization=args.use_quantization
    )
    
    # Setup LoRA if requested
    if args.use_lora:
        logger.info("Setting up LoRA...")
        model = setup_lora(model, config_dict)
    
    # Monitor initial memory
    monitor_memory_usage()
    
    # Load dataset
    logger.info("Loading training dataset...")
    train_dataset = load_dataset(args.data, tokenizer, args.max_examples)
    
    eval_dataset = None
    if args.eval_data:
        logger.info("Loading evaluation dataset...")
        eval_dataset = load_dataset(args.eval_data, tokenizer, args.max_examples // 10 if args.max_examples else None)
    
    # Setup data collator
    data_collator = create_data_collator(tokenizer)
    
    # Setup training arguments
    training_args = setup_training_arguments(config_dict, args.output)
    
    # Setup wandb
    model_name = config_dict.get('model_config', {}).get('name', 'cognitive-1.5b')
    setup_wandb(config_dict, model_name)
    
    # Setup trainer
    cognitive_loss_weights = config_dict.get('training_config', {}).get('loss_config', {}).get('auxiliary_losses', {})
    
    trainer = CognitiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        cognitive_loss_weights=cognitive_loss_weights
    )
    
    # Add callback for memory monitoring
    class MemoryCallback:
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                monitor_memory_usage()
    
    trainer.add_callback(MemoryCallback())
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming training from {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        logger.info("Starting training...")
        trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output, "final_model"))
    
    # Save configuration
    with open(os.path.join(args.output, "final_model", "config.yaml"), 'w') as f:
        yaml.dump(config_dict, f)
    
    logger.info("Training completed!")
    
    # Final memory check
    monitor_memory_usage()
    
    # Finish wandb
    wandb.finish()


if __name__ == "__main__":
    main()