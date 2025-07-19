#!/usr/bin/env python3
"""
Synthetic Data Generation for Cognitive SLM Training
Generates diverse cognitive training examples with tool calling
"""

import os
import sys
import argparse
import yaml
import json
import random
from typing import List, Dict, Any
from datetime import datetime
import uuid

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.cognitive_data_schemas import (
    CognitiveTrainingExample, ReasoningType, DifficultyLevel, ToolType,
    ToolDefinition, ToolCall, ToolResult, ReasoningStep, StateTransition,
    BUILTIN_TOOLS, create_example_dataset_config
)


class SyntheticDataGenerator:
    """Generate synthetic cognitive training data"""
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = create_example_dataset_config()
        if 'dataset_config' in self.config:
            # Update with config file values
            for key, value in self.config['dataset_config'].items():
                if hasattr(self.dataset_config, key):
                    setattr(self.dataset_config, key, value)
        
        self.available_tools = BUILTIN_TOOLS.copy()
        self.examples = []
    
    def generate_chain_of_thought_example(self, difficulty: DifficultyLevel, 
                                        domain: str) -> CognitiveTrainingExample:
        """Generate a chain-of-thought reasoning example"""
        
        if domain == "mathematics":
            if difficulty == DifficultyLevel.BASIC:
                context = "Sarah has 3 boxes of apples. Each box contains 12 apples. She gives away 8 apples to her neighbor and eats 5 apples herself."
                instruction = "How many apples does Sarah have left?"
                
                reasoning_steps = [
                    ReasoningStep(
                        step_number=1,
                        description="Calculate total apples initially",
                        reasoning="3 boxes × 12 apples per box = 36 apples",
                        memory_update="Total apples: 36",
                        confidence=0.95
                    ),
                    ReasoningStep(
                        step_number=2,
                        description="Calculate apples given away and eaten",
                        reasoning="Given away: 8 apples, Eaten: 5 apples, Total used: 8 + 5 = 13 apples",
                        memory_update="Used apples: 13",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=3,
                        description="Calculate remaining apples",
                        reasoning="Remaining = Initial - Used = 36 - 13 = 23 apples",
                        memory_update="Final count: 23",
                        confidence=0.95
                    )
                ]
                
                final_answer = "Sarah has 23 apples left."
                
            elif difficulty == DifficultyLevel.INTERMEDIATE:
                context = "A train travels 120 km in the first 2 hours at a constant speed. Then it increases its speed by 20 km/h and travels for another 3 hours."
                instruction = "What is the total distance traveled and the average speed for the entire journey?"
                
                reasoning_steps = [
                    ReasoningStep(
                        step_number=1,
                        description="Calculate initial speed",
                        reasoning="Speed = Distance / Time = 120 km / 2 hours = 60 km/h",
                        memory_update="Initial speed: 60 km/h",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=2,
                        description="Calculate increased speed",
                        reasoning="New speed = 60 + 20 = 80 km/h",
                        memory_update="New speed: 80 km/h",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=3,
                        description="Calculate distance in second part",
                        reasoning="Distance = Speed × Time = 80 km/h × 3 hours = 240 km",
                        memory_update="Second distance: 240 km",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=4,
                        description="Calculate total distance and average speed",
                        reasoning="Total distance = 120 + 240 = 360 km, Total time = 2 + 3 = 5 hours, Average speed = 360 / 5 = 72 km/h",
                        memory_update="Total: 360 km, 72 km/h average",
                        confidence=0.95
                    )
                ]
                
                final_answer = "The train traveled a total distance of 360 km with an average speed of 72 km/h."
                
            else:  # ADVANCED
                context = "A circular swimming pool has a radius of 10 meters. A rectangular deck surrounds the pool with a uniform width of 2 meters on all sides."
                instruction = "Calculate the area of the deck (excluding the pool area)."
                
                reasoning_steps = [
                    ReasoningStep(
                        step_number=1,
                        description="Calculate pool area",
                        reasoning="Pool area = π × r² = π × 10² = 100π square meters",
                        memory_update="Pool area: 100π m²",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=2,
                        description="Determine outer rectangle dimensions",
                        reasoning="Pool diameter = 20m, deck width = 2m each side, so rectangle = (20 + 4) × (20 + 4) = 24 × 24",
                        memory_update="Rectangle: 24m × 24m",
                        confidence=0.85
                    ),
                    ReasoningStep(
                        step_number=3,
                        description="Calculate total rectangle area",
                        reasoning="Rectangle area = 24 × 24 = 576 square meters",
                        memory_update="Rectangle area: 576 m²",
                        confidence=0.9
                    ),
                    ReasoningStep(
                        step_number=4,
                        description="Calculate deck area",
                        reasoning="Deck area = Rectangle area - Pool area = 576 - 100π ≈ 576 - 314.16 = 261.84 square meters",
                        memory_update="Deck area: 576 - 100π m²",
                        confidence=0.9
                    )
                ]
                
                final_answer = "The deck area is 576 - 100π square meters, which is approximately 261.84 square meters."
        
        elif domain == "science":
            context = "A ball is dropped from a height of 20 meters. Assuming no air resistance and g = 9.8 m/s²."
            instruction = "How long does it take for the ball to hit the ground and what is its velocity when it hits?"
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Identify the physics equation",
                    reasoning="For free fall: h = ½gt², where h = height, g = acceleration, t = time",
                    memory_update="Equation: h = ½gt²",
                    confidence=0.95
                ),
                ReasoningStep(
                    step_number=2,
                    description="Solve for time",
                    reasoning="20 = ½ × 9.8 × t², so t² = 40/9.8 ≈ 4.08, therefore t ≈ 2.02 seconds",
                    memory_update="Time: 2.02 seconds",
                    confidence=0.9
                ),
                ReasoningStep(
                    step_number=3,
                    description="Calculate final velocity",
                    reasoning="v = gt = 9.8 × 2.02 ≈ 19.8 m/s",
                    memory_update="Final velocity: 19.8 m/s",
                    confidence=0.9
                )
            ]
            
            final_answer = "The ball takes approximately 2.02 seconds to hit the ground and has a velocity of 19.8 m/s when it hits."
        
        else:  # general domain
            context = "You need to organize a dinner party for 8 people. Each person eats on average 2 slices of pizza, and each pizza has 8 slices."
            instruction = "How many pizzas should you order, and what's the cost if each pizza costs $15?"
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Calculate total slices needed",
                    reasoning="8 people × 2 slices per person = 16 slices needed",
                    memory_update="Total slices: 16",
                    confidence=0.95
                ),
                ReasoningStep(
                    step_number=2,
                    description="Calculate pizzas needed",
                    reasoning="16 slices ÷ 8 slices per pizza = 2 pizzas exactly",
                    memory_update="Pizzas needed: 2",
                    confidence=0.95
                ),
                ReasoningStep(
                    step_number=3,
                    description="Calculate total cost",
                    reasoning="2 pizzas × $15 per pizza = $30 total",
                    memory_update="Total cost: $30",
                    confidence=0.95
                )
            ]
            
            final_answer = "You should order 2 pizzas for a total cost of $30."
        
        return CognitiveTrainingExample(
            task_type=ReasoningType.CHAIN_OF_THOUGHT,
            difficulty=difficulty,
            domain=domain,
            context=context,
            instruction=instruction,
            reasoning_steps=reasoning_steps,
            final_answer=final_answer,
            confidence_score=0.9,
            generated_by="synthetic_generator",
            quality_score=0.85
        )
    
    def generate_tool_calling_example(self, difficulty: DifficultyLevel,
                                    domain: str) -> CognitiveTrainingExample:
        """Generate a tool calling example"""
        
        # Select appropriate tools
        if domain == "mathematics":
            available_tools = [tool for tool in BUILTIN_TOOLS if tool.tool_type == ToolType.MATHEMATICS]
        else:
            available_tools = BUILTIN_TOOLS  # Use all available tools
        
        if domain == "mathematics":
            context = "You need to solve a complex mathematical problem involving multiple calculations."
            instruction = "Calculate the compound interest on $1000 invested at 5% annual interest rate for 3 years, compounded annually. Then find the square root of the final amount."
            
            # First tool call - calculate compound interest
            calc_call_1 = ToolCall(
                tool_name="calculate",
                parameters={"expression": "1000 * (1 + 0.05) ** 3"},
                reasoning="Using compound interest formula: A = P(1 + r)^t"
            )
            
            calc_result_1 = ToolResult(
                call_id=calc_call_1.call_id,
                success=True,
                result={"success": True, "result": 1157.625, "expression": "1000 * (1 + 0.05) ** 3"}
            )
            
            # Second tool call - square root
            calc_call_2 = ToolCall(
                tool_name="calculate", 
                parameters={"expression": "math.sqrt(1157.625)"},
                reasoning="Finding square root of the compound interest result"
            )
            
            calc_result_2 = ToolResult(
                call_id=calc_call_2.call_id,
                success=True,
                result={"success": True, "result": 34.023, "expression": "math.sqrt(1157.625)"}
            )
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Set up compound interest calculation",
                    reasoning="Need to use formula A = P(1 + r)^t where P=1000, r=0.05, t=3",
                    tool_calls=[calc_call_1],
                    confidence=0.9
                ),
                ReasoningStep(
                    step_number=2,
                    description="Calculate square root of result",
                    reasoning="Take square root of $1157.625 to get final answer",
                    tool_calls=[calc_call_2],
                    confidence=0.9
                )
            ]
            
            tool_calls = [calc_call_1, calc_call_2]
            tool_results = [calc_result_1, calc_result_2]
            
            final_answer = "The compound interest after 3 years is $1157.63, and the square root of this amount is approximately 34.02."
        
        elif domain == "data_analysis":
            context = "You have collected test scores from a class and need to analyze the data."
            instruction = "Analyze the following test scores and provide summary statistics: [85, 92, 78, 96, 88, 91, 84, 89, 93, 87]"
            
            data_call = ToolCall(
                tool_name="analyze_data",
                parameters={
                    "data": [85, 92, 78, 96, 88, 91, 84, 89, 93, 87],
                    "analysis_type": "summary"
                },
                reasoning="Get comprehensive statistics for the test scores"
            )
            
            data_result = ToolResult(
                call_id=data_call.call_id,
                success=True,
                result={
                    "success": True,
                    "analysis_type": "summary",
                    "result": {
                        "count": 10,
                        "min": 78,
                        "max": 96,
                        "mean": 88.3,
                        "sum": 883
                    },
                    "data_size": 10
                }
            )
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Analyze the test score data",
                    reasoning="Use data analysis tool to compute summary statistics",
                    tool_calls=[data_call],
                    confidence=0.95
                )
            ]
            
            tool_calls = [data_call]
            tool_results = [data_result]
            
            final_answer = "The test scores analysis shows: 10 students, scores ranging from 78 to 96, with a mean of 88.3 and total sum of 883."
        
        else:  # general domain
            context = "You need to search for information and then analyze some text."
            instruction = "Search for information about 'machine learning' and then analyze the sentiment of this text: 'I love working with AI systems, they are amazing and helpful!'"
            
            search_call = ToolCall(
                tool_name="search_knowledge",
                parameters={"query": "machine learning", "max_results": 3},
                reasoning="Get information about machine learning"
            )
            
            search_result = ToolResult(
                call_id=search_call.call_id,
                success=True,
                result={
                    "success": True,
                    "query": "machine learning", 
                    "results": [
                        "Information about machine learning - Result 1",
                        "Information about machine learning - Result 2",
                        "Information about machine learning - Result 3"
                    ],
                    "total_found": 3
                }
            )
            
            text_call = ToolCall(
                tool_name="text_analyze",
                parameters={
                    "text": "I love working with AI systems, they are amazing and helpful!",
                    "analysis_type": "sentiment"
                },
                reasoning="Analyze sentiment of the given text"
            )
            
            text_result = ToolResult(
                call_id=text_call.call_id,
                success=True,
                result={
                    "success": True,
                    "analysis_type": "sentiment",
                    "result": {
                        "sentiment": "positive",
                        "positive_indicators": 3,
                        "negative_indicators": 0
                    }
                }
            )
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Search for machine learning information",
                    reasoning="Use search tool to gather relevant information",
                    tool_calls=[search_call],
                    confidence=0.85
                ),
                ReasoningStep(
                    step_number=2,
                    description="Analyze text sentiment",
                    reasoning="Use text analysis tool to determine sentiment",
                    tool_calls=[text_call],
                    confidence=0.9
                )
            ]
            
            tool_calls = [search_call, text_call]
            tool_results = [search_result, text_result]
            
            final_answer = "Found 3 machine learning information sources. The text sentiment analysis shows positive sentiment with 3 positive indicators and 0 negative indicators."
        
        return CognitiveTrainingExample(
            task_type=ReasoningType.TOOL_CALLING,
            difficulty=difficulty,
            domain=domain,
            context=context,
            instruction=instruction,
            available_tools=available_tools,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            tool_results=tool_results,
            final_answer=final_answer,
            confidence_score=0.88,
            generated_by="synthetic_generator",
            quality_score=0.82
        )
    
    def generate_planning_example(self, difficulty: DifficultyLevel,
                                domain: str) -> CognitiveTrainingExample:
        """Generate a planning task example"""
        
        if domain == "planning":
            context = "You need to organize a birthday party for 20 people. You have 4 hours to prepare everything."
            instruction = "Create a step-by-step plan to organize the party efficiently."
            
            state_transitions = [
                StateTransition(
                    from_state="initial_planning",
                    to_state="shopping_preparation",
                    action="Make shopping list and check supplies",
                    preconditions=["guest count confirmed", "time available"],
                    effects=["shopping list ready"],
                    duration=30,
                    rationale="Need to know what to buy before going shopping"
                ),
                StateTransition(
                    from_state="shopping_preparation", 
                    to_state="shopping_execution",
                    action="Go shopping for party supplies and food",
                    preconditions=["shopping list ready", "transportation available"],
                    effects=["supplies acquired", "food acquired"],
                    duration=90,
                    rationale="Get all necessary items in one trip to save time"
                ),
                StateTransition(
                    from_state="shopping_execution",
                    to_state="venue_preparation",
                    action="Set up decorations and arrange furniture",
                    preconditions=["supplies acquired", "venue access"],
                    effects=["venue decorated", "seating arranged"],
                    duration=60,
                    rationale="Prepare the space for guests"
                ),
                StateTransition(
                    from_state="venue_preparation",
                    to_state="food_preparation",
                    action="Prepare food and drinks",
                    preconditions=["food acquired", "kitchen access"],
                    effects=["food ready", "drinks ready"],
                    duration=90,
                    rationale="Have food ready before guests arrive"
                ),
                StateTransition(
                    from_state="food_preparation",
                    to_state="party_ready",
                    action="Final setup and welcome guests",
                    preconditions=["venue decorated", "food ready"],
                    effects=["party started"],
                    duration=10,
                    rationale="Final preparations and guest reception"
                )
            ]
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Analyze time constraints and requirements",
                    reasoning="4 hours total, 20 people, need food, decorations, and venue setup",
                    memory_update="Time: 4h, People: 20, Tasks: food, decor, setup",
                    confidence=0.9
                ),
                ReasoningStep(
                    step_number=2,
                    description="Prioritize tasks by dependencies",
                    reasoning="Shopping must come first, then venue setup, then food prep",
                    memory_update="Order: shopping → venue → food → final",
                    confidence=0.85
                ),
                ReasoningStep(
                    step_number=3,
                    description="Allocate time for each task",
                    reasoning="Shopping 90min, venue 60min, food 90min, final 10min = 250min total",
                    memory_update="Time allocation planned, 30min buffer remaining",
                    confidence=0.8
                )
            ]
            
            final_answer = "Plan: 1) Make shopping list (30min), 2) Shop for supplies and food (90min), 3) Set up venue and decorations (60min), 4) Prepare food and drinks (90min), 5) Final setup and greet guests (10min). Total: 280 minutes with 20-minute buffer."
        
        else:  # general domain
            context = "You're moving to a new apartment and need to pack everything efficiently."
            instruction = "Create a plan to pack your belongings systematically."
            
            state_transitions = [
                StateTransition(
                    from_state="unprepared",
                    to_state="supplies_ready",
                    action="Gather packing supplies",
                    effects=["boxes available", "tape available", "markers available"],
                    rationale="Need supplies before starting to pack"
                ),
                StateTransition(
                    from_state="supplies_ready",
                    to_state="non_essentials_packed",
                    action="Pack non-essential items first",
                    effects=["decorations packed", "books packed", "seasonal items packed"],
                    rationale="Pack items you don't need immediately"
                ),
                StateTransition(
                    from_state="non_essentials_packed",
                    to_state="essentials_packed",
                    action="Pack essential items last",
                    effects=["clothes packed", "toiletries packed", "important documents secured"],
                    rationale="Keep essentials accessible until the last moment"
                )
            ]
            
            reasoning_steps = [
                ReasoningStep(
                    step_number=1,
                    description="Categorize belongings by necessity",
                    reasoning="Separate essential daily items from non-essential items",
                    memory_update="Categories: essential vs non-essential",
                    confidence=0.9
                ),
                ReasoningStep(
                    step_number=2,
                    description="Plan packing order",
                    reasoning="Pack non-essentials first to avoid disrupting daily routine",
                    memory_update="Order: supplies → non-essentials → essentials",
                    confidence=0.85
                )
            ]
            
            final_answer = "Packing plan: 1) Gather all packing supplies, 2) Pack non-essential items (decorations, books, seasonal items), 3) Pack essential items last (clothes, toiletries, documents). This minimizes disruption to daily life."
        
        return CognitiveTrainingExample(
            task_type=ReasoningType.PLANNING,
            difficulty=difficulty,
            domain=domain,
            context=context,
            instruction=instruction,
            reasoning_steps=reasoning_steps,
            state_transitions=state_transitions,
            final_answer=final_answer,
            confidence_score=0.85,
            generated_by="synthetic_generator",
            quality_score=0.8
        )
    
    def generate_examples(self, num_examples: int) -> List[CognitiveTrainingExample]:
        """Generate a set of training examples"""
        examples = []
        
        for _ in range(num_examples):
            # Sample task type based on distribution
            task_type = random.choices(
                list(self.dataset_config.task_distribution.keys()),
                weights=list(self.dataset_config.task_distribution.values())
            )[0]
            
            # Sample difficulty based on distribution
            difficulty = random.choices(
                list(self.dataset_config.difficulty_distribution.keys()),
                weights=list(self.dataset_config.difficulty_distribution.values())
            )[0]
            
            # Sample domain based on distribution
            domain = random.choices(
                list(self.dataset_config.domain_distribution.keys()),
                weights=list(self.dataset_config.domain_distribution.values())
            )[0]
            
            # Generate example based on task type
            if task_type == ReasoningType.CHAIN_OF_THOUGHT:
                example = self.generate_chain_of_thought_example(difficulty, domain)
            elif task_type == ReasoningType.TOOL_CALLING:
                example = self.generate_tool_calling_example(difficulty, domain)
            elif task_type == ReasoningType.PLANNING:
                example = self.generate_planning_example(difficulty, domain)
            else:
                # Default to chain of thought for other types
                example = self.generate_chain_of_thought_example(difficulty, domain)
            
            # Validate example
            errors = example.validate()
            if not errors:
                examples.append(example)
            else:
                print(f"Warning: Generated example has validation errors: {errors}")
        
        return examples
    
    def save_examples(self, examples: List[CognitiveTrainingExample], output_path: str):
        """Save examples to file"""
        # Convert examples to training format
        training_data = []
        for example in examples:
            training_format = example.to_training_format()
            training_data.append(training_format)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        print(f"Saved {len(examples)} examples to {output_path}")
    
    def generate_and_save(self, num_examples: int, output_path: str):
        """Generate examples and save to file"""
        print(f"Generating {num_examples} cognitive training examples...")
        examples = self.generate_examples(num_examples)
        
        print(f"Successfully generated {len(examples)} examples")
        self.save_examples(examples, output_path)
        
        # Print statistics
        task_counts = {}
        difficulty_counts = {}
        domain_counts = {}
        
        for example in examples:
            task_counts[example.task_type.value] = task_counts.get(example.task_type.value, 0) + 1
            difficulty_counts[example.difficulty.value] = difficulty_counts.get(example.difficulty.value, 0) + 1
            domain_counts[example.domain] = domain_counts.get(example.domain, 0) + 1
        
        print("\nGenerated data statistics:")
        print(f"Task types: {task_counts}")
        print(f"Difficulties: {difficulty_counts}")
        print(f"Domains: {domain_counts}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cognitive training data")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of examples to generate")
    parser.add_argument("--split", action="store_true", help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    generator = SyntheticDataGenerator(args.config)
    
    if args.split:
        # Generate and split data
        examples = generator.generate_examples(args.num_examples)
        
        # Split data
        random.shuffle(examples)
        train_size = int(0.8 * len(examples))
        val_size = int(0.1 * len(examples))
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        # Save splits
        base_path = args.output.rsplit('.', 1)[0]
        generator.save_examples(train_examples, f"{base_path}_train.json")
        generator.save_examples(val_examples, f"{base_path}_val.json") 
        generator.save_examples(test_examples, f"{base_path}_test.json")
        
        print(f"Split data: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")
    else:
        # Generate single file
        generator.generate_and_save(args.num_examples, args.output)


if __name__ == "__main__":
    main()