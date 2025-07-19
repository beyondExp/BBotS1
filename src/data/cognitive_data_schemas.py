"""
Comprehensive Data Schemas for Cognitive SLM Training
Supports reasoning, planning, tool calling, and memory tasks
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Union, Any
from enum import Enum
import json
import uuid
from datetime import datetime
import torch


class ReasoningType(Enum):
    """Types of reasoning tasks"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    PLANNING = "planning"
    STATE_TRANSITION = "state_transition"
    PROBLEM_SOLVING = "problem_solving"
    MEMORY_RECALL = "memory_recall"
    TOOL_CALLING = "tool_calling"
    META_COGNITION = "meta_cognition"
    CAUSAL_REASONING = "causal_reasoning"


class DifficultyLevel(Enum):
    """Difficulty levels for cognitive tasks"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ToolType(Enum):
    """Types of tools available for calling"""
    MATHEMATICS = "mathematics"
    SEARCH = "search"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATIONS = "file_operations"
    API_CALLS = "api_calls"
    CODE_EXECUTION = "code_execution"
    VISUALIZATION = "visualization"
    TEXT_ANALYSIS = "text_analysis"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Optional[Any] = None
    constraints: Optional[Dict] = None  # min, max, pattern, enum, etc.
    
    def validate(self, value: Any) -> bool:
        """Validate a parameter value"""
        # Basic type checking
        if self.type == "string" and not isinstance(value, str):
            return False
        elif self.type == "integer" and not isinstance(value, int):
            return False
        elif self.type == "float" and not isinstance(value, (int, float)):
            return False
        elif self.type == "boolean" and not isinstance(value, bool):
            return False
        elif self.type == "array" and not isinstance(value, list):
            return False
        elif self.type == "object" and not isinstance(value, dict):
            return False
        
        # Constraint checking
        if self.constraints:
            if "min" in self.constraints and value < self.constraints["min"]:
                return False
            if "max" in self.constraints and value > self.constraints["max"]:
                return False
            if "enum" in self.constraints and value not in self.constraints["enum"]:
                return False
            if "pattern" in self.constraints and self.type == "string":
                import re
                if not re.match(self.constraints["pattern"], value):
                    return False
        
        return True


@dataclass
class ToolDefinition:
    """Complete tool definition for model training"""
    name: str
    tool_type: ToolType
    description: str
    parameters: List[ToolParameter]
    return_type: str
    return_description: str
    examples: List[Dict] = field(default_factory=list)
    safety_level: Literal["safe", "restricted", "dangerous"] = "safe"
    execution_timeout: int = 30  # seconds
    
    def to_schema(self) -> Dict:
        """Convert to JSON schema format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **(param.constraints or {})
                    } for param in self.parameters
                },
                "required": [param.name for param in self.parameters if param.required]
            },
            "returns": {
                "type": self.return_type,
                "description": self.return_description
            }
        }


@dataclass
class ToolCall:
    """A specific tool call with parameters"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning: Optional[str] = None  # Why this tool was called
    expected_outcome: Optional[str] = None  # What the model expects


@dataclass
class ToolResult:
    """Result of a tool execution"""
    call_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_embedding_format(self) -> str:
        """Convert to string format for model consumption"""
        if self.success:
            return f"TOOL_RESULT[{self.call_id}]: {json.dumps(self.result)}"
        else:
            return f"TOOL_ERROR[{self.call_id}]: {self.error_message}"


@dataclass
class ReasoningStep:
    """A single step in reasoning process"""
    step_number: int
    description: str
    reasoning: str
    memory_update: Optional[str] = None
    confidence: Optional[float] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    state_change: Optional[Dict] = None
    
    def to_training_format(self) -> str:
        """Convert to model training format"""
        step_text = f"Step {self.step_number}: {self.description}\n"
        step_text += f"Reasoning: {self.reasoning}\n"
        
        if self.tool_calls:
            step_text += "Tool Calls:\n"
            for tool_call in self.tool_calls:
                step_text += f"  - {tool_call.tool_name}({json.dumps(tool_call.parameters)})\n"
                if tool_call.reasoning:
                    step_text += f"    Reason: {tool_call.reasoning}\n"
        
        if self.memory_update:
            step_text += f"Memory Update: {self.memory_update}\n"
        
        if self.confidence is not None:
            step_text += f"Confidence: {self.confidence:.2f}\n"
            
        return step_text


@dataclass
class StateTransition:
    """State transition for planning tasks"""
    from_state: str
    to_state: str
    action: str
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: Optional[float] = None
    duration: Optional[float] = None
    rationale: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


@dataclass
class CognitiveTrainingExample:
    """Core training data structure for cognitive capabilities"""
    
    # Identifiers
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    domain: str = "general"
    
    # Input data
    context: str = ""
    instruction: str = ""
    constraints: List[str] = field(default_factory=list)
    available_tools: List[ToolDefinition] = field(default_factory=list)
    
    # Reasoning process
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    state_transitions: List[StateTransition] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    
    # Memory management
    initial_memory: Optional[List[str]] = None
    memory_updates: List[str] = field(default_factory=list)
    working_memory_usage: List[str] = field(default_factory=list)
    
    # Output
    final_answer: str = ""
    confidence_score: Optional[float] = None
    alternative_solutions: List[str] = field(default_factory=list)
    
    # Metadata
    generated_by: str = "unknown"
    verified: bool = False
    quality_score: Optional[float] = None
    creation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        """Validate the training example"""
        errors = []
        
        if not self.context.strip():
            errors.append("Context cannot be empty")
        
        if not self.instruction.strip():
            errors.append("Instruction cannot be empty")
        
        if not self.final_answer.strip():
            errors.append("Final answer cannot be empty")
        
        if self.confidence_score is not None and not (0 <= self.confidence_score <= 1):
            errors.append("Confidence score must be between 0 and 1")
        
        if self.quality_score is not None and not (0 <= self.quality_score <= 1):
            errors.append("Quality score must be between 0 and 1")
        
        # Validate tool calls
        for tool_call in self.tool_calls:
            # Find corresponding tool definition
            tool_def = next((t for t in self.available_tools if t.name == tool_call.tool_name), None)
            if not tool_def:
                errors.append(f"Tool '{tool_call.tool_name}' not found in available tools")
                continue
            
            # Validate parameters
            for param in tool_def.parameters:
                if param.required and param.name not in tool_call.parameters:
                    errors.append(f"Required parameter '{param.name}' missing for tool '{tool_call.tool_name}'")
                elif param.name in tool_call.parameters:
                    if not param.validate(tool_call.parameters[param.name]):
                        errors.append(f"Invalid parameter '{param.name}' for tool '{tool_call.tool_name}'")
        
        return errors
    
    def to_training_format(self) -> Dict:
        """Convert to model training format"""
        return {
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_prompt()
                },
                {
                    "role": "user", 
                    "content": self._build_user_prompt()
                },
                {
                    "role": "assistant",
                    "content": self._build_assistant_response()
                }
            ],
            "metadata": {
                "id": self.id,
                "task_type": self.task_type.value,
                "difficulty": self.difficulty.value,
                "domain": self.domain,
                "quality_score": self.quality_score,
                "has_tools": len(self.available_tools) > 0,
                "num_reasoning_steps": len(self.reasoning_steps),
                "num_tool_calls": len(self.tool_calls)
            }
        }
    
    def _build_system_prompt(self) -> str:
        """Build cognitive system prompt"""
        system_parts = [
            "You are a cognitive AI system capable of reasoning, planning, and problem-solving."
        ]
        
        # Add task-specific instructions
        if self.task_type == ReasoningType.CHAIN_OF_THOUGHT:
            system_parts.append("Focus on step-by-step logical reasoning.")
        elif self.task_type == ReasoningType.PLANNING:
            system_parts.append("Focus on creating executable plans with clear state transitions.")
        elif self.task_type == ReasoningType.TOOL_CALLING:
            system_parts.append("You have access to external tools. Use them when appropriate.")
        
        # Add available tools
        if self.available_tools:
            system_parts.append("\nAvailable Tools:")
            for tool in self.available_tools:
                tool_schema = tool.to_schema()
                system_parts.append(f"- {tool.name}: {tool.description}")
                system_parts.append(f"  Parameters: {json.dumps(tool_schema['parameters'], indent=2)}")
        
        # Add constraints
        if self.constraints:
            system_parts.append(f"\nConstraints: {'; '.join(self.constraints)}")
        
        system_parts.extend([
            "\nInstructions:",
            "1. Think step-by-step through the problem",
            "2. Show your reasoning process clearly", 
            "3. Use available tools when needed",
            "4. Track important information in working memory",
            "5. Provide a clear final answer"
        ])
        
        return "\n".join(system_parts)
    
    def _build_user_prompt(self) -> str:
        """Build user prompt"""
        user_parts = []
        
        if self.context:
            user_parts.append(f"Context: {self.context}")
        
        user_parts.append(f"Task: {self.instruction}")
        
        if self.initial_memory:
            user_parts.append(f"Initial Memory: {'; '.join(self.initial_memory)}")
        
        return "\n\n".join(user_parts)
    
    def _build_assistant_response(self) -> str:
        """Build the expected assistant response"""
        response_parts = []
        
        # Add reasoning process
        if self.reasoning_steps:
            response_parts.append("**Reasoning Process:**")
            for step in self.reasoning_steps:
                response_parts.append(step.to_training_format())
        
        # Add planning steps
        if self.state_transitions:
            response_parts.append("**Planning Steps:**")
            for i, transition in enumerate(self.state_transitions, 1):
                transition_text = f"{i}. {transition.from_state} → {transition.to_state}"
                transition_text += f"\n   Action: {transition.action}"
                if transition.rationale:
                    transition_text += f"\n   Rationale: {transition.rationale}"
                if transition.tool_calls:
                    transition_text += "\n   Tool Calls:"
                    for tool_call in transition.tool_calls:
                        transition_text += f"\n     - {tool_call.tool_name}({json.dumps(tool_call.parameters)})"
                response_parts.append(transition_text)
        
        # Add tool usage
        if self.tool_calls:
            response_parts.append("**Tool Usage:**")
            for tool_call in self.tool_calls:
                tool_text = f"- Calling {tool_call.tool_name}({json.dumps(tool_call.parameters)})"
                if tool_call.reasoning:
                    tool_text += f"\n  Reasoning: {tool_call.reasoning}"
                
                # Add corresponding result if available
                result = next((r for r in self.tool_results if r.call_id == tool_call.call_id), None)
                if result:
                    tool_text += f"\n  Result: {result.to_embedding_format()}"
                
                response_parts.append(tool_text)
        
        # Add memory updates
        if self.memory_updates:
            response_parts.append("**Memory Updates:**")
            for update in self.memory_updates:
                response_parts.append(f"- {update}")
        
        # Add final answer
        response_parts.append(f"**Final Answer:** {self.final_answer}")
        
        # Add confidence if available
        if self.confidence_score is not None:
            response_parts.append(f"**Confidence:** {self.confidence_score:.2f}")
        
        return "\n\n".join(response_parts)


@dataclass
class DatasetConfiguration:
    """Configuration for dataset generation"""
    target_size: int
    task_distribution: Dict[ReasoningType, float]
    difficulty_distribution: Dict[DifficultyLevel, float]
    domain_distribution: Dict[str, float]
    tool_usage_ratio: float = 0.4  # Percentage of examples that use tools
    synthetic_ratio: float = 0.8
    quality_threshold: float = 0.7
    max_reasoning_steps: int = 10
    max_tools_per_example: int = 5
    
    def validate(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        if abs(sum(self.task_distribution.values()) - 1.0) > 1e-6:
            errors.append("Task distribution must sum to 1.0")
        
        if abs(sum(self.difficulty_distribution.values()) - 1.0) > 1e-6:
            errors.append("Difficulty distribution must sum to 1.0")
        
        if abs(sum(self.domain_distribution.values()) - 1.0) > 1e-6:
            errors.append("Domain distribution must sum to 1.0")
        
        if not (0 <= self.tool_usage_ratio <= 1):
            errors.append("Tool usage ratio must be between 0 and 1")
        
        if not (0 <= self.synthetic_ratio <= 1):
            errors.append("Synthetic ratio must be between 0 and 1")
        
        if not (0 <= self.quality_threshold <= 1):
            errors.append("Quality threshold must be between 0 and 1")
        
        return errors


@dataclass 
class DatasetSplit:
    """Dataset split configuration"""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    def validate(self) -> List[str]:
        """Validate split ratios"""
        errors = []
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            errors.append("Split ratios must sum to 1.0")
        return errors


# Built-in tool definitions for common cognitive tasks
BUILTIN_TOOLS = [
    ToolDefinition(
        name="calculate",
        tool_type=ToolType.MATHEMATICS,
        description="Perform mathematical calculations",
        parameters=[
            ToolParameter("expression", "string", "Mathematical expression to evaluate", True)
        ],
        return_type="number",
        return_description="Result of the calculation",
        examples=[
            {"input": {"expression": "2 + 3 * 4"}, "output": 14},
            {"input": {"expression": "sqrt(16)"}, "output": 4}
        ]
    ),
    ToolDefinition(
        name="search_knowledge",
        tool_type=ToolType.SEARCH,
        description="Search for information in knowledge base",
        parameters=[
            ToolParameter("query", "string", "Search query", True),
            ToolParameter("max_results", "integer", "Maximum number of results", False, 5)
        ],
        return_type="array",
        return_description="List of relevant information",
        examples=[
            {"input": {"query": "photosynthesis"}, "output": ["Process by which plants convert sunlight to energy..."]}
        ]
    ),
    ToolDefinition(
        name="analyze_data",
        tool_type=ToolType.DATA_PROCESSING,
        description="Analyze structured data",
        parameters=[
            ToolParameter("data", "array", "Data to analyze", True),
            ToolParameter("analysis_type", "string", "Type of analysis", True, constraints={"enum": ["mean", "median", "mode", "std", "summary"]})
        ],
        return_type="object",
        return_description="Analysis results",
        examples=[
            {"input": {"data": [1, 2, 3, 4, 5], "analysis_type": "mean"}, "output": {"mean": 3.0}}
        ]
    ),
    ToolDefinition(
        name="code_execute",
        tool_type=ToolType.CODE_EXECUTION,
        description="Execute Python code safely",
        parameters=[
            ToolParameter("code", "string", "Python code to execute", True),
            ToolParameter("timeout", "integer", "Execution timeout in seconds", False, 10)
        ],
        return_type="object",
        return_description="Execution result including output and any errors",
        safety_level="restricted",
        examples=[
            {"input": {"code": "print('Hello, World!')"}, "output": {"stdout": "Hello, World!\n", "stderr": "", "success": True}}
        ]
    )
]


def get_builtin_tools_by_type(tool_type: ToolType) -> List[ToolDefinition]:
    """Get built-in tools by type"""
    return [tool for tool in BUILTIN_TOOLS if tool.tool_type == tool_type]


def create_example_dataset_config() -> DatasetConfiguration:
    """Create example dataset configuration"""
    return DatasetConfiguration(
        target_size=10000,
        task_distribution={
            ReasoningType.CHAIN_OF_THOUGHT: 0.3,
            ReasoningType.PLANNING: 0.2,
            ReasoningType.TOOL_CALLING: 0.2,
            ReasoningType.PROBLEM_SOLVING: 0.15,
            ReasoningType.MEMORY_RECALL: 0.1,
            ReasoningType.META_COGNITION: 0.05
        },
        difficulty_distribution={
            DifficultyLevel.BASIC: 0.3,
            DifficultyLevel.INTERMEDIATE: 0.4,
            DifficultyLevel.ADVANCED: 0.25,
            DifficultyLevel.EXPERT: 0.05
        },
        domain_distribution={
            "mathematics": 0.25,
            "science": 0.2,
            "logic": 0.15,
            "planning": 0.15,
            "general": 0.1,
            "programming": 0.1,
            "analysis": 0.05
        },
        tool_usage_ratio=0.4,
        synthetic_ratio=0.8,
        quality_threshold=0.7
    )