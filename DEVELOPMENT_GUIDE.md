# Cognitive SLM Development Guide: Complete Implementation

## 🎯 Project Overview

This project implements a **Cognitive Small Language Model (SLM)** optimized for **RTX 3090 Ti** hardware with **24GB VRAM**. The model features advanced cognitive capabilities including reasoning, planning, working memory, state tracking, and **tool calling**.

### Key Features
- **1.5B parameter cognitive architecture**
- **Tool calling system** with safe execution environment
- **Working memory** for context retention
- **Reasoning modules** with step-by-step thinking
- **State tracking** for planning tasks
- **Memory optimization** for RTX 3090 Ti
- **Comprehensive training pipeline**

## 🏗️ Architecture Components

### 1. Model Architecture (`src/models/cognitive_architecture.py`)

#### Core Components:
- **CognitiveLanguageModel**: Main model class with cognitive modules
- **ToolCallingModule**: Handles function discovery and execution
- **WorkingMemoryModule**: Maintains context across interactions
- **ReasoningModule**: Specialized attention for reasoning tasks
- **StateTrackingModule**: Manages planning and state transitions

#### Key Features:
```python
# Model instantiation
config = CognitiveConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=16,
    max_tools=128,
    working_memory_size=512
)
model = CognitiveLanguageModel(config)

# Tool registration
model.register_tool("calculate", math_function, "Mathematical calculations", params)
```

### 2. Tool Calling System (`src/tools/tool_execution_engine.py`)

#### Execution Modes:
- **SAFE**: Restricted sandbox (recommended)
- **RESTRICTED**: Limited permissions
- **DANGEROUS**: Full access (use with caution)

#### Built-in Tools:
- **calculate**: Mathematical computations
- **search_knowledge**: Information retrieval
- **analyze_data**: Statistical analysis
- **code_execute**: Python code execution
- **text_analyze**: Text processing

#### Usage Example:
```python
from src.tools.tool_execution_engine import create_tool_engine, ExecutionMode

engine = create_tool_engine(ExecutionMode.SAFE)
tool_call = ToolCall("calculate", {"expression": "2 + 3 * 4"})
result = engine.execute_tool(tool_call)
```

### 3. Data Schemas (`src/data/cognitive_data_schemas.py`)

#### Training Data Structure:
```python
@dataclass
class CognitiveTrainingExample:
    task_type: ReasoningType  # CHAIN_OF_THOUGHT, TOOL_CALLING, PLANNING
    difficulty: DifficultyLevel  # BASIC, INTERMEDIATE, ADVANCED, EXPERT
    context: str
    instruction: str
    reasoning_steps: List[ReasoningStep]
    tool_calls: List[ToolCall]
    final_answer: str
    confidence_score: float
```

## 🚀 Development Workflow

### Phase 1: Environment Setup (Week 1)

#### 1. Clone and Setup Project:
```bash
git clone <repository-url>
cd cognitive-slm
cp .env.example .env
# Edit .env with your specific settings
```

#### 2. Install Dependencies:
```bash
# For RTX 3090 Ti optimization
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Verify CUDA setup
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name()}')"
```

#### 3. Hardware Optimization:
```bash
# Set environment variables for RTX 3090 Ti
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Phase 2: Data Generation (Weeks 2-3)

#### 1. Generate Synthetic Training Data:
```bash
# Generate 10,000 training examples
python scripts/generate_data.py \
    --config configs/model_configs/cognitive_1.5b.yaml \
    --output data/synthetic/cognitive_training.json \
    --num-examples 10000 \
    --split

# This creates:
# - data/synthetic/cognitive_training_train.json (8,000 examples)
# - data/synthetic/cognitive_training_val.json (1,000 examples)  
# - data/synthetic/cognitive_training_test.json (1,000 examples)
```

#### 2. Data Quality Validation:
```python
from src.data.cognitive_data_schemas import CognitiveTrainingExample
import json

# Load and validate data
with open('data/synthetic/cognitive_training_train.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} training examples")
print(f"Task distribution: {[ex['metadata']['task_type'] for ex in data[:10]]}")
```

### Phase 3: Model Training (Weeks 4-8)

#### 1. Basic Training (No LoRA):
```bash
python scripts/train_model.py \
    --config configs/model_configs/cognitive_1.5b.yaml \
    --data data/synthetic/cognitive_training_train.json \
    --eval-data data/synthetic/cognitive_training_val.json \
    --output models/cognitive-1.5b-base \
    --max-examples 5000
```

#### 2. Memory-Efficient Training (With LoRA + Quantization):
```bash
python scripts/train_model.py \
    --config configs/model_configs/cognitive_1.5b.yaml \
    --data data/synthetic/cognitive_training_train.json \
    --eval-data data/synthetic/cognitive_training_val.json \
    --output models/cognitive-1.5b-lora \
    --use-lora \
    --use-quantization \
    --max-examples 10000
```

#### 3. Monitor Training:
```bash
# View logs
tail -f models/cognitive-1.5b-lora/logs/cognitive_slm_*.log

# Monitor with tensorboard
tensorboard --logdir models/cognitive-1.5b-lora/logs

# Check wandb dashboard
# Visit: https://wandb.ai/your-username/cognitive-slm
```

### Phase 4: Evaluation & Testing (Weeks 9-10)

#### 1. Model Evaluation:
```bash
python scripts/evaluate_model.py \
    --model-path models/cognitive-1.5b-lora/final_model \
    --test-data data/synthetic/cognitive_training_test.json \
    --output-dir evaluation_results
```

#### 2. Interactive Testing:
```python
from src.models.cognitive_architecture import CognitiveLanguageModel
from src.tools.tool_execution_engine import create_tool_engine
from transformers import AutoTokenizer

# Load model
model = CognitiveLanguageModel.from_pretrained("models/cognitive-1.5b-lora/final_model")
tokenizer = AutoTokenizer.from_pretrained("models/cognitive-1.5b-lora/final_model")

# Setup tool engine
tool_engine = create_tool_engine()

# Register tools with model
model.tool_registry = tool_engine.get_available_tools()

# Test reasoning
prompt = "Calculate the compound interest on $1000 at 5% for 3 years."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, do_sample=True, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🧠 Training Data Formats

### 1. Chain-of-Thought Example:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a cognitive AI system capable of reasoning, planning, and problem-solving. Focus on step-by-step logical reasoning."
    },
    {
      "role": "user",
      "content": "Context: Sarah has 3 boxes of apples. Each box contains 12 apples.\n\nTask: How many apples does Sarah have in total?"
    },
    {
      "role": "assistant", 
      "content": "**Reasoning Process:**\n\nStep 1: Identify the quantities\nReasoning: Sarah has 3 boxes, each containing 12 apples\nMemory: boxes=3, apples_per_box=12\n\nStep 2: Calculate total apples\nReasoning: Total = number of boxes × apples per box = 3 × 12 = 36\nMemory: total_apples=36\n\n**Final Answer:** Sarah has 36 apples in total."
    }
  ],
  "metadata": {
    "task_type": "chain_of_thought",
    "difficulty": "basic",
    "domain": "mathematics",
    "has_tools": false
  }
}
```

### 2. Tool Calling Example:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a cognitive AI system with access to external tools.\n\nAvailable Tools:\n- calculate: Perform mathematical calculations\n  Parameters: {\"expression\": \"string\"}"
    },
    {
      "role": "user",
      "content": "Context: You need to solve a complex calculation.\n\nTask: Calculate the compound interest on $1000 at 5% annual rate for 3 years."
    },
    {
      "role": "assistant",
      "content": "**Reasoning Process:**\n\nStep 1: Set up compound interest calculation\nReasoning: Need to use formula A = P(1 + r)^t where P=1000, r=0.05, t=3\n\n**Tool Usage:**\n- Calling calculate({\"expression\": \"1000 * (1 + 0.05) ** 3\"})\n  Reasoning: Using compound interest formula\n  Result: {\"success\": true, \"result\": 1157.625}\n\n**Final Answer:** The compound interest after 3 years is $1157.63."
    }
  ],
  "metadata": {
    "task_type": "tool_calling",
    "difficulty": "intermediate", 
    "domain": "mathematics",
    "has_tools": true,
    "num_tool_calls": 1
  }
}
```

### 3. Planning Example:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a cognitive AI system capable of creating executable plans with clear state transitions."
    },
    {
      "role": "user",
      "content": "Context: You need to organize a birthday party for 20 people in 4 hours.\n\nTask: Create an efficient step-by-step plan."
    },
    {
      "role": "assistant",
      "content": "**Planning Steps:**\n\n1. initial_planning → shopping_preparation\n   Action: Make shopping list and check supplies\n   Rationale: Need to know what to buy before shopping\n\n2. shopping_preparation → shopping_execution  \n   Action: Go shopping for party supplies and food\n   Rationale: Get all items in one trip to save time\n\n3. shopping_execution → venue_preparation\n   Action: Set up decorations and arrange furniture\n   Rationale: Prepare the space for guests\n\n4. venue_preparation → party_ready\n   Action: Final setup and welcome guests\n   Rationale: Complete preparations for party start\n\n**Final Answer:** Complete plan with 4 main phases totaling 280 minutes, leaving 20-minute buffer."
    }
  ],
  "metadata": {
    "task_type": "planning",
    "difficulty": "intermediate",
    "domain": "planning"
  }
}
```

## ⚡ Memory Optimization for RTX 3090 Ti

### Hardware Configuration:
- **VRAM**: 24GB GDDR6X
- **Optimal Model Size**: 1-3B parameters
- **Batch Size**: 2-8 (depending on sequence length)
- **Memory Target**: 95% utilization (22.8GB)

### Optimization Techniques:

#### 1. Gradient Checkpointing:
```python
# Automatically enabled in model config
model.gradient_checkpointing_enable()
# Saves ~40% memory at cost of ~20% speed
```

#### 2. Mixed Precision Training:
```yaml
# In config file
hardware_optimizations:
  mixed_precision: "bf16"  # Use BF16 for RTX 3090 Ti
  attention_softmax_in_fp32: true  # Stability
```

#### 3. QLoRA (4-bit Quantization):
```python
# Enable in training script
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

#### 4. Dynamic Batch Sizing:
```python
from src.utils.memory_utils import MemoryOptimizer

optimizer = MemoryOptimizer(target_memory_usage=0.95)
optimal_batch_size = optimizer.optimize_batch_size(model, sample_input)
print(f"Optimal batch size: {optimal_batch_size}")
```

#### 5. Memory Monitoring:
```python
from src.utils.memory_utils import MemoryOptimizer

optimizer = MemoryOptimizer()

# Monitor during training
with optimizer.memory_monitor("training_step"):
    loss = model(**batch)
    loss.backward()

# Print stats
optimizer.print_memory_stats()
```

## 🧪 Testing & Validation

### 1. Unit Tests:
```bash
# Run all tests
python -m pytest tests/unit/ -v

# Test specific components
python -m pytest tests/unit/test_cognitive_architecture.py -v
python -m pytest tests/unit/test_tool_execution.py -v
```

### 2. Integration Tests:
```bash
# End-to-end training test
python -m pytest tests/integration/test_training_pipeline.py -v

# Tool calling integration
python -m pytest tests/integration/test_tool_calling.py -v
```

### 3. Performance Tests:
```bash
# Memory usage tests
python tests/performance/test_memory_usage.py

# Training speed benchmarks
python tests/performance/test_training_speed.py
```

## 📊 Performance Targets

### Training Metrics:
- **Training Loss**: < 2.5 after convergence
- **Validation Perplexity**: < 15
- **Memory Usage**: < 22GB VRAM during training
- **Training Speed**: > 500 tokens/sec
- **Convergence**: Within 10,000 steps

### Cognitive Benchmarks:
- **Chain-of-Thought**: > 85% accuracy on math problems
- **Planning Tasks**: > 80% success rate on multi-step problems
- **Tool Usage**: > 90% correct function calls
- **Reasoning Quality**: > 4.0/5.0 human evaluation score

### Tool Calling Metrics:
- **Function Selection**: > 95% accuracy
- **Parameter Generation**: > 90% valid parameters
- **Execution Success**: > 95% successful tool calls
- **Safety**: 0% dangerous operations in SAFE mode

## 🔧 Troubleshooting

### Common Issues:

#### 1. CUDA Out of Memory:
```bash
# Solution 1: Reduce batch size
--batch-size 2

# Solution 2: Enable quantization
--use-quantization

# Solution 3: Use gradient accumulation
gradient_accumulation_steps: 16
```

#### 2. Training Not Converging:
```bash
# Check learning rate
learning_rate: 5e-5  # Try 1e-4 or 2e-5

# Increase warmup steps
warmup_steps: 2000

# Check data quality
python scripts/validate_data.py --data path/to/data.json
```

#### 3. Tool Calling Errors:
```python
# Debug tool execution
engine = create_tool_engine(ExecutionMode.SAFE)
stats = engine.get_execution_stats()
print(f"Success rate: {stats['success_rate']:.2%}")

# Check tool registration
available_tools = engine.get_available_tools()
print(f"Available tools: {list(available_tools.keys())}")
```

#### 4. Memory Leaks:
```python
# Regular cleanup
from src.utils.memory_utils import MemoryOptimizer
optimizer = MemoryOptimizer()
optimizer.cleanup_memory()

# Monitor memory growth
optimizer.print_memory_stats()
```

## 🚀 Advanced Features

### 1. Custom Tool Development:
```python
def custom_search_tool(query: str, domain: str = "general") -> Dict[str, Any]:
    """Custom search implementation"""
    # Your search logic here
    results = search_external_api(query, domain)
    return {"results": results, "query": query}

# Register with engine
engine.register_tool(
    "custom_search",
    custom_search_tool,
    "Search external knowledge base",
    ["query", "domain"],
    ExecutionMode.RESTRICTED
)
```

### 2. Multi-GPU Training (Future):
```bash
# For multiple RTX 3090 Ti cards
torchrun --nproc_per_node=2 scripts/train_model.py \
    --config configs/model_configs/cognitive_3b.yaml \
    --data data/synthetic/large_dataset.json \
    --output models/cognitive-3b-multi
```

### 3. Model Serving:
```python
from transformers import pipeline
from src.models.cognitive_architecture import CognitiveLanguageModel

# Create inference pipeline
model = CognitiveLanguageModel.from_pretrained("models/cognitive-1.5b-final")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Serve via API
@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json["prompt"]
    response = pipe(prompt, max_length=512, do_sample=True)
    return {"response": response[0]["generated_text"]}
```

## 📈 Scaling Recommendations

### For Larger Models (3B+):
1. **Multi-GPU Setup**: Use 2x RTX 3090 Ti for 3B models
2. **DeepSpeed Integration**: Enable ZeRO-2 or ZeRO-3
3. **Gradient Accumulation**: Increase to 32+ steps
4. **Checkpoint Frequency**: Save every 500 steps

### For Production Deployment:
1. **Model Quantization**: Use 8-bit inference
2. **Model Pruning**: Remove unused cognitive modules
3. **TensorRT Optimization**: Convert to TensorRT for faster inference
4. **Batch Processing**: Implement dynamic batching

## 📚 Additional Resources

### Documentation:
- [Model Architecture Details](docs/architecture.md)
- [Training Configuration Guide](docs/training.md)
- [Tool Calling API Reference](docs/tool_calling.md)
- [Performance Optimization](docs/optimization.md)

### Community:
- [GitHub Issues](https://github.com/your-org/cognitive-slm/issues)
- [Discussion Forum](https://github.com/your-org/cognitive-slm/discussions)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Ready to start building your cognitive SLM? Follow this guide step by step, and you'll have a powerful reasoning model running on your RTX 3090 Ti in no time! 🚀**