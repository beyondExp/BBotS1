#!/usr/bin/env python3
"""Test script for the cognitive model"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from src.models.cognitive_architecture import CognitiveConfig, CognitiveLanguageModel
    
    print("✅ Imports successful")
    
    # Create a small test configuration
    config = CognitiveConfig(
        vocab_size=1000,  # Small for testing
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512
    )
    
    print("✅ Config created successfully")
    print(f"   - vocab_size: {config.vocab_size}")
    print(f"   - hidden_size: {config.hidden_size}")
    print(f"   - num_layers: {config.num_hidden_layers}")
    
    # Create the model
    model = CognitiveLanguageModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("✅ Model created successfully!")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model type: {type(model).__name__}")
    
    # Test a simple forward pass
    import torch
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    print(f"✅ Testing forward pass with input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
        
    print(f"✅ Forward pass successful!")
    print(f"   - Output logits shape: {outputs.logits.shape}")
    print(f"   - Expected shape: ({batch_size}, {seq_length}, {config.vocab_size})")
    
    if outputs.logits.shape == (batch_size, seq_length, config.vocab_size):
        print("✅ All tests passed! Model is working correctly.")
    else:
        print("❌ Output shape mismatch")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()