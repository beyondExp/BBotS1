"""
Tool Execution Engine for Cognitive SLM
Provides safe, sandboxed execution of external tools and functions
"""

import json
import time
import threading
import subprocess
import tempfile
import os
import sys
import importlib
import inspect
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback
import ast
import math
import re
from datetime import datetime

from src.data.cognitive_data_schemas import ToolDefinition, ToolCall, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Tool execution modes"""
    SAFE = "safe"           # Restricted sandbox environment
    RESTRICTED = "restricted" # Limited permissions
    DANGEROUS = "dangerous"   # Full access (use with extreme caution)


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors"""
    pass


class SafeEvaluator:
    """Safe evaluation environment for mathematical and simple expressions"""
    
    ALLOWED_NAMES = {
        'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes', 'chr',
        'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
        'hex', 'int', 'len', 'list', 'map', 'max', 'min', 'oct', 'ord',
        'pow', 'range', 'reversed', 'round', 'set', 'slice', 'sorted',
        'str', 'sum', 'tuple', 'zip'
    }
    
    ALLOWED_MODULES = {
        'math': ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh',
                'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp',
                'hypot', 'ldexp', 'log', 'log10', 'modf', 'pi', 'pow',
                'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh'],
        'datetime': ['datetime', 'date', 'time', 'timedelta'],
        'json': ['loads', 'dumps'],
        're': ['match', 'search', 'findall', 'sub', 'split']
    }
    
    def __init__(self):
        self.safe_dict = {"__builtins__": {}}
        
        # Add allowed built-in functions
        for name in self.ALLOWED_NAMES:
            if hasattr(__builtins__, name):
                self.safe_dict[name] = getattr(__builtins__, name)
        
        # Add allowed modules
        for module_name, allowed_attrs in self.ALLOWED_MODULES.items():
            try:
                module = importlib.import_module(module_name)
                module_dict = {}
                for attr in allowed_attrs:
                    if hasattr(module, attr):
                        module_dict[attr] = getattr(module, attr)
                self.safe_dict[module_name] = type(sys)('safe_' + module_name)
                for attr, value in module_dict.items():
                    setattr(self.safe_dict[module_name], attr, value)
            except ImportError:
                logger.warning(f"Module {module_name} not available")
    
    def evaluate(self, expression: str) -> Any:
        """Safely evaluate an expression"""
        try:
            # Parse the expression to check for dangerous operations
            parsed = ast.parse(expression, mode='eval')
            self._check_ast_safety(parsed)
            
            # Evaluate in safe environment
            return eval(expression, self.safe_dict)
        except Exception as e:
            raise ToolExecutionError(f"Safe evaluation failed: {str(e)}")
    
    def _check_ast_safety(self, node):
        """Check if AST node is safe to execute"""
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom, ast.Exec)):
                raise ToolExecutionError("Import statements not allowed")
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    # Check method calls
                    attr_name = child.func.attr
                    if attr_name.startswith('_'):
                        raise ToolExecutionError(f"Private method access not allowed: {attr_name}")
            elif isinstance(child, ast.Attribute):
                if child.attr.startswith('_'):
                    raise ToolExecutionError(f"Private attribute access not allowed: {child.attr}")


class ToolSandbox:
    """Sandboxed environment for tool execution"""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE,
                 timeout: int = 30, max_memory: int = 100 * 1024 * 1024):  # 100MB
        """
        Args:
            execution_mode: Execution mode for safety level
            timeout: Maximum execution time in seconds
            max_memory: Maximum memory usage in bytes
        """
        self.execution_mode = execution_mode
        self.timeout = timeout
        self.max_memory = max_memory
        self.evaluator = SafeEvaluator()
        
    def execute_python_code(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python code safely"""
        if self.execution_mode == ExecutionMode.SAFE:
            return self._execute_safe_python(code, globals_dict)
        elif self.execution_mode == ExecutionMode.RESTRICTED:
            return self._execute_restricted_python(code, globals_dict)
        else:
            return self._execute_dangerous_python(code, globals_dict)
    
    def _execute_safe_python(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python code in safe mode"""
        try:
            # Parse and check AST
            parsed = ast.parse(code)
            self.evaluator._check_ast_safety(parsed)
            
            # Prepare execution environment
            exec_globals = self.evaluator.safe_dict.copy()
            if globals_dict:
                # Only allow safe globals
                for key, value in globals_dict.items():
                    if not key.startswith('_') and not callable(value):
                        exec_globals[key] = value
            
            exec_locals = {}
            
            # Capture stdout
            import io
            old_stdout = sys.stdout
            stdout_capture = io.StringIO()
            sys.stdout = stdout_capture
            
            try:
                # Execute with timeout
                def target():
                    exec(code, exec_globals, exec_locals)
                
                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout=self.timeout)
                
                if thread.is_alive():
                    raise ToolExecutionError("Execution timeout")
                
                stdout_content = stdout_capture.getvalue()
                
                return {
                    "success": True,
                    "stdout": stdout_content,
                    "stderr": "",
                    "locals": {k: v for k, v in exec_locals.items() if not k.startswith('_')},
                    "result": exec_locals.get('result', None)
                }
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "locals": {},
                "result": None,
                "error": str(e)
            }
    
    def _execute_restricted_python(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python code in restricted mode"""
        # Similar to safe mode but with more permissions
        # This is a placeholder - implement based on specific needs
        return self._execute_safe_python(code, globals_dict)
    
    def _execute_dangerous_python(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute Python code with full permissions (dangerous!)"""
        logger.warning("Executing code in DANGEROUS mode - use with extreme caution!")
        
        try:
            exec_globals = globals() if globals_dict is None else globals_dict
            exec_locals = {}
            
            # Capture stdout and stderr
            import io
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
            sys.stdout, sys.stderr = stdout_capture, stderr_capture
            
            try:
                exec(code, exec_globals, exec_locals)
                
                return {
                    "success": True,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "locals": exec_locals,
                    "result": exec_locals.get('result', None)
                }
                
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "locals": {},
                "result": None,
                "error": str(e)
            }


class BuiltinTools:
    """Collection of built-in tools for cognitive tasks"""
    
    @staticmethod
    def calculate(expression: str, sandbox: ToolSandbox) -> Dict[str, Any]:
        """Perform mathematical calculations"""
        try:
            result = sandbox.evaluator.evaluate(expression)
            return {
                "success": True,
                "result": result,
                "expression": expression
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }
    
    @staticmethod
    def search_knowledge(query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for information (placeholder implementation)"""
        # This would integrate with actual knowledge bases
        mock_results = [
            f"Information about {query} - Result {i+1}"
            for i in range(min(max_results, 3))
        ]
        
        return {
            "success": True,
            "query": query,
            "results": mock_results,
            "total_found": len(mock_results)
        }
    
    @staticmethod
    def analyze_data(data: List[Union[int, float]], analysis_type: str) -> Dict[str, Any]:
        """Analyze numerical data"""
        try:
            if not data:
                return {"success": False, "error": "Empty data provided"}
            
            if analysis_type == "mean":
                result = sum(data) / len(data)
            elif analysis_type == "median":
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    result = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                else:
                    result = sorted_data[n//2]
            elif analysis_type == "mode":
                from collections import Counter
                counts = Counter(data)
                result = counts.most_common(1)[0][0]
            elif analysis_type == "std":
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                result = variance ** 0.5
            elif analysis_type == "summary":
                result = {
                    "count": len(data),
                    "min": min(data),
                    "max": max(data),
                    "mean": sum(data) / len(data),
                    "sum": sum(data)
                }
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result,
                "data_size": len(data)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }
    
    @staticmethod
    def code_execute(code: str, timeout: int = 10, sandbox: ToolSandbox = None) -> Dict[str, Any]:
        """Execute Python code safely"""
        if sandbox is None:
            sandbox = ToolSandbox(ExecutionMode.SAFE, timeout)
        
        return sandbox.execute_python_code(code)
    
    @staticmethod
    def text_analyze(text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze text content"""
        try:
            if analysis_type == "summary":
                words = text.split()
                sentences = text.split('.')
                result = {
                    "character_count": len(text),
                    "word_count": len(words),
                    "sentence_count": len([s for s in sentences if s.strip()]),
                    "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
                }
            elif analysis_type == "sentiment":
                # Simple sentiment analysis (placeholder)
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiment = "positive"
                elif neg_count > pos_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                result = {
                    "sentiment": sentiment,
                    "positive_indicators": pos_count,
                    "negative_indicators": neg_count
                }
            else:
                return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_type": analysis_type
            }


class ToolExecutionEngine:
    """Main tool execution engine"""
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.SAFE,
                 default_timeout: int = 30):
        """
        Args:
            execution_mode: Default execution mode
            default_timeout: Default timeout for tool execution
        """
        self.execution_mode = execution_mode
        self.default_timeout = default_timeout
        self.sandbox = ToolSandbox(execution_mode, default_timeout)
        self.registered_tools = {}
        self.execution_history = []
        
        # Register built-in tools
        self._register_builtin_tools()
        
        logger.info(f"ToolExecutionEngine initialized in {execution_mode.value} mode")
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        builtin_tools = {
            "calculate": {
                "function": BuiltinTools.calculate,
                "description": "Perform mathematical calculations",
                "parameters": ["expression"],
                "execution_mode": ExecutionMode.SAFE
            },
            "search_knowledge": {
                "function": BuiltinTools.search_knowledge,
                "description": "Search for information in knowledge base",
                "parameters": ["query", "max_results"],
                "execution_mode": ExecutionMode.SAFE
            },
            "analyze_data": {
                "function": BuiltinTools.analyze_data,
                "description": "Analyze numerical data",
                "parameters": ["data", "analysis_type"],
                "execution_mode": ExecutionMode.SAFE
            },
            "code_execute": {
                "function": BuiltinTools.code_execute,
                "description": "Execute Python code safely",
                "parameters": ["code", "timeout"],
                "execution_mode": ExecutionMode.RESTRICTED
            },
            "text_analyze": {
                "function": BuiltinTools.text_analyze,
                "description": "Analyze text content",
                "parameters": ["text", "analysis_type"],
                "execution_mode": ExecutionMode.SAFE
            }
        }
        
        for name, tool_info in builtin_tools.items():
            self.registered_tools[name] = tool_info
    
    def register_tool(self, name: str, function: Callable,
                     description: str, parameters: List[str],
                     execution_mode: ExecutionMode = ExecutionMode.SAFE):
        """Register a custom tool"""
        self.registered_tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters,
            "execution_mode": execution_mode
        }
        logger.info(f"Registered tool: {name}")
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        start_time = time.time()
        
        try:
            # Check if tool exists
            if tool_call.tool_name not in self.registered_tools:
                raise ToolExecutionError(f"Tool '{tool_call.tool_name}' not found")
            
            tool_info = self.registered_tools[tool_call.tool_name]
            function = tool_info["function"]
            
            # Prepare parameters
            if tool_call.tool_name in ["calculate", "code_execute"]:
                # These tools need the sandbox
                result = function(sandbox=self.sandbox, **tool_call.parameters)
            else:
                result = function(**tool_call.parameters)
            
            execution_time = time.time() - start_time
            
            # Create successful result
            tool_result = ToolResult(
                call_id=tool_call.call_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Tool execution failed: {str(e)}"
            logger.error(f"Error executing tool '{tool_call.tool_name}': {error_message}")
            
            tool_result = ToolResult(
                call_id=tool_call.call_id,
                success=False,
                result=None,
                error_message=error_message,
                execution_time=execution_time
            )
        
        # Log execution
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_call.tool_name,
            "call_id": tool_call.call_id,
            "success": tool_result.success,
            "execution_time": tool_result.execution_time,
            "error": tool_result.error_message
        })
        
        return tool_result
    
    def execute_multiple_tools(self, tool_calls: List[ToolCall],
                             parallel: bool = False) -> List[ToolResult]:
        """Execute multiple tool calls"""
        if not parallel:
            return [self.execute_tool(call) for call in tool_calls]
        
        # Parallel execution
        results = []
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as executor:
            future_to_call = {
                executor.submit(self.execute_tool, call): call 
                for call in tool_calls
            }
            
            for future in future_to_call:
                try:
                    result = future.result(timeout=self.default_timeout)
                    results.append(result)
                except TimeoutError:
                    call = future_to_call[future]
                    error_result = ToolResult(
                        call_id=call.call_id,
                        success=False,
                        result=None,
                        error_message="Tool execution timeout",
                        execution_time=self.default_timeout
                    )
                    results.append(error_result)
        
        return results
    
    def get_available_tools(self) -> Dict[str, Dict]:
        """Get information about available tools"""
        return {
            name: {
                "description": info["description"],
                "parameters": info["parameters"],
                "execution_mode": info["execution_mode"].value
            }
            for name, info in self.registered_tools.items()
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for entry in self.execution_history if entry["success"])
        failed_executions = total_executions - successful_executions
        
        execution_times = [entry["execution_time"] for entry in self.execution_history]
        avg_execution_time = sum(execution_times) / len(execution_times)
        
        tool_usage = {}
        for entry in self.execution_history:
            tool_name = entry["tool_name"]
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "average_execution_time": avg_execution_time,
            "tool_usage": tool_usage
        }


# Convenience function for creating a tool execution engine
def create_tool_engine(execution_mode: ExecutionMode = ExecutionMode.SAFE,
                      timeout: int = 30) -> ToolExecutionEngine:
    """Create a configured tool execution engine"""
    return ToolExecutionEngine(execution_mode, timeout)