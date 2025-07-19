"""
Cognitive Language Model Architecture with Tool Calling
Optimized for RTX 3090 Ti (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
try:
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm,
        rotate_half,
        apply_rotary_pos_emb
    )
except ImportError:
    # Fallback implementations
    LlamaRMSNorm = nn.LayerNorm


class CognitiveAttention(nn.Module):
    """Custom attention mechanism for cognitive model"""
    
    def __init__(self, config: 'CognitiveConfig', layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, tuple):
                kv_seq_len += past_key_value[0].shape[-2]
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class CognitiveMLP(nn.Module):
    """Custom MLP for cognitive model"""
    
    def __init__(self, config: 'CognitiveConfig'):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish activation
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class CognitiveConfig(PretrainedConfig):
    """Configuration for Cognitive Language Model with Tool Calling"""
    
    model_type = "cognitive_transformer"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=4096,
        
        # Cognitive module settings
        working_memory_size=512,
        working_memory_update_freq=4,
        num_reasoning_heads=4,
        reasoning_dim=256,
        reasoning_layers=None,
        num_states=32,
        state_dim=128,
        planning_horizon=8,
        action_space_size=64,
        
        # Tool calling settings
        max_tools=128,
        tool_embedding_dim=256,
        tool_hidden_dim=512,
        
        # Standard settings
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        

        
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        
        # Cognitive settings
        self.working_memory_size = working_memory_size
        self.working_memory_update_freq = working_memory_update_freq
        self.num_reasoning_heads = num_reasoning_heads
        self.reasoning_dim = reasoning_dim
        self.reasoning_layers = reasoning_layers or [12, 16, 20, 23]
        self.num_states = num_states
        self.state_dim = state_dim
        self.planning_horizon = planning_horizon
        self.action_space_size = action_space_size
        
        # Tool calling settings
        self.max_tools = max_tools
        self.tool_embedding_dim = tool_embedding_dim
        self.tool_hidden_dim = tool_hidden_dim
        
        # Standard settings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        

        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ToolCallingModule(nn.Module):
    """Tool calling system with function discovery and execution"""
    
    def __init__(self, config: CognitiveConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_tools = config.max_tools
        self.tool_embedding_dim = config.tool_embedding_dim
        self.tool_hidden_dim = config.tool_hidden_dim
        
        # Tool registry and embeddings
        self.tool_embeddings = nn.Embedding(self.max_tools, self.tool_embedding_dim)
        self.tool_name_encoder = nn.Linear(self.hidden_size, self.tool_embedding_dim)
        
        # Function selection network
        self.function_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.tool_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.tool_hidden_dim, self.max_tools)
        )
        
        # Parameter generation network
        self.parameter_generator = nn.Sequential(
            nn.Linear(self.hidden_size + self.tool_embedding_dim, self.tool_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.tool_hidden_dim, self.hidden_size)
        )
        
        # Result integration network
        self.result_integrator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.tool_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.tool_hidden_dim, self.hidden_size)
        )
        
        # Tool call gate
        self.tool_gate = nn.Linear(self.hidden_size, 1)
        
        # Special tokens for tool calling
        self.tool_call_token = nn.Parameter(torch.randn(self.hidden_size))
        self.tool_result_token = nn.Parameter(torch.randn(self.hidden_size))
        
    def forward(self, 
                hidden_states: torch.Tensor,
                tool_registry: Optional[Dict[str, Any]] = None,
                execute_tools: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            tool_registry: Dictionary of available tools
            execute_tools: Whether to actually execute tools
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Determine if tool calling is needed
        tool_gate_scores = torch.sigmoid(self.tool_gate(hidden_states))
        should_call_tools = tool_gate_scores > 0.5
        
        tool_outputs = None
        enhanced_states = hidden_states
        
        if should_call_tools.any() and tool_registry is not None:
            # Select tools to call
            function_logits = self.function_selector(hidden_states)
            tool_probs = F.softmax(function_logits, dim=-1)
            
            # Get top-k tools for each position
            top_k = 3
            top_tool_indices = torch.topk(tool_probs, top_k, dim=-1).indices
            
            tool_call_results = []
            
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    if should_call_tools[batch_idx, seq_idx]:
                        # Get context for this position
                        context_vector = hidden_states[batch_idx, seq_idx]
                        
                        # Generate tool calls
                        for tool_idx in top_tool_indices[batch_idx, seq_idx]:
                            tool_id = tool_idx.item()
                            if tool_id < len(tool_registry):
                                tool_embedding = self.tool_embeddings(tool_idx)
                                
                                # Generate parameters
                                param_input = torch.cat([context_vector, tool_embedding], dim=0)
                                generated_params = self.parameter_generator(param_input.unsqueeze(0))
                                
                                if execute_tools:
                                    # Execute tool (placeholder for actual execution)
                                    tool_result = self._execute_tool(
                                        tool_registry, tool_id, generated_params
                                    )
                                    tool_call_results.append({
                                        'batch_idx': batch_idx,
                                        'seq_idx': seq_idx,
                                        'tool_id': tool_id,
                                        'result': tool_result
                                    })
            
            # Integrate tool results back into hidden states
            if tool_call_results:
                for result in tool_call_results:
                    batch_idx = result['batch_idx']
                    seq_idx = result['seq_idx']
                    tool_result_embedding = result['result']  # Assume it's already embedded
                    
                    # Integrate result
                    original_state = hidden_states[batch_idx, seq_idx]
                    integration_input = torch.cat([original_state, tool_result_embedding], dim=0)
                    integrated_state = self.result_integrator(integration_input.unsqueeze(0))
                    
                    enhanced_states[batch_idx, seq_idx] = integrated_state.squeeze(0)
            
            tool_outputs = {
                'tool_probs': tool_probs,
                'should_call_tools': should_call_tools,
                'tool_call_results': tool_call_results
            }
        
        return enhanced_states, tool_outputs
    
    def _execute_tool(self, tool_registry: Dict, tool_id: int, parameters: torch.Tensor) -> torch.Tensor:
        """Execute a tool (placeholder implementation)"""
        # In a real implementation, this would:
        # 1. Decode parameters to actual function arguments
        # 2. Call the tool function safely in a sandbox
        # 3. Encode the result back to tensor format
        
        # For now, return a dummy result
        return torch.randn_like(parameters)


class WorkingMemoryModule(nn.Module):
    """Working memory module for cognitive processing"""
    
    def __init__(self, config: CognitiveConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.working_memory_size
        self.hidden_size = config.hidden_size
        
        # Memory storage
        self.memory_bank = nn.Parameter(
            torch.randn(self.memory_size, self.hidden_size) * 0.02
        )
        
        # Attention mechanisms for memory access
        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Memory update mechanism (GRU-style)
        self.update_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.reset_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.new_memory = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Memory importance scoring
        self.importance_scorer = nn.Linear(self.hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor, update_memory: bool = False):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Expand memory for batch
        memory = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute attention between hidden states and memory
        queries = self.query_proj(hidden_states)
        keys = self.key_proj(memory)
        values = self.value_proj(memory)
        
        # Attention computation
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(hidden_size)
        attention_weights = F.softmax(scores, dim=-1)
        memory_output = torch.matmul(attention_weights, values)
        
        # Project output
        memory_enhanced = self.output_proj(memory_output)
        
        # Update memory if required
        if update_memory:
            # Compute importance scores for current input
            importance_scores = self.importance_scorer(hidden_states)
            
            # Weighted average of hidden states
            importance_weights = F.softmax(importance_scores, dim=1)
            context_vector = torch.sum(
                hidden_states * importance_weights, dim=1, keepdim=True
            )  # [batch, 1, hidden]
            
            context_expanded = context_vector.expand(-1, self.memory_size, -1)
            
            # Compute gates for memory update
            combined = torch.cat([memory, context_expanded], dim=-1)
            update_gate = torch.sigmoid(self.update_gate(combined))
            reset_gate = torch.sigmoid(self.reset_gate(combined))
            
            # Compute new memory content
            reset_memory = reset_gate * memory
            new_content_input = torch.cat([reset_memory, context_expanded], dim=-1)
            new_content = torch.tanh(self.new_memory(new_content_input))
            
            # Update memory
            updated_memory = (1 - update_gate) * memory + update_gate * new_content
            
            # Update the parameter (use batch average)
            avg_updated_memory = updated_memory.mean(dim=0)
            self.memory_bank.data = avg_updated_memory.detach()
        
        return memory_enhanced


class ReasoningModule(nn.Module):
    """Specialized reasoning attention heads with step prediction"""
    
    def __init__(self, config: CognitiveConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_reasoning_heads
        self.head_dim = config.reasoning_dim
        self.hidden_size = config.hidden_size
        
        # Multi-head reasoning attention
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Reasoning-specific components
        self.reasoning_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.step_predictor = nn.Linear(self.hidden_size, 8)  # Predict next reasoning step type
        self.confidence_estimator = nn.Linear(self.hidden_size, 1)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer norm
        normed_states = self.layer_norm(hidden_states)
        
        # Multi-head attention computation
        queries = self.q_proj(normed_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        keys = self.k_proj(normed_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        values = self.v_proj(normed_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        
        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # [batch, heads, seq, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attention_output)
        
        # Apply reasoning gate
        gate = torch.sigmoid(self.reasoning_gate(hidden_states))
        reasoning_output = gate * output + (1 - gate) * hidden_states
        
        # Predict next reasoning step (only for last token)
        step_logits = self.step_predictor(reasoning_output[:, -1:, :])
        
        # Estimate confidence
        confidence_scores = torch.sigmoid(self.confidence_estimator(reasoning_output))
        
        return reasoning_output, step_logits, confidence_scores


class StateTrackingModule(nn.Module):
    """State tracking for planning and decision making"""
    
    def __init__(self, config: CognitiveConfig):
        super().__init__()
        self.config = config
        self.num_states = config.num_states
        self.state_dim = config.state_dim
        self.hidden_size = config.hidden_size
        
        # State representation
        self.state_embeddings = nn.Embedding(self.num_states, self.state_dim)
        self.state_encoder = nn.Linear(self.hidden_size, self.state_dim)
        self.state_decoder = nn.Linear(self.state_dim, self.hidden_size)
        
        # State transition model
        self.transition_net = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.state_dim, self.num_states)
        )
        
        # Planning network
        self.planning_net = nn.Sequential(
            nn.Linear(self.state_dim + self.hidden_size, self.state_dim),
            nn.ReLU(),
            nn.Linear(self.state_dim, config.action_space_size)
        )
        
        # Current state tracker (learned parameter)
        self.current_state = nn.Parameter(torch.zeros(self.state_dim))
        
    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode hidden states to state space
        encoded_states = self.state_encoder(hidden_states)
        
        # Compute state transitions
        current_state_expanded = self.current_state.unsqueeze(0).unsqueeze(0)
        current_state_expanded = current_state_expanded.expand(batch_size, seq_len, -1)
        
        # Predict next states
        state_input = torch.cat([current_state_expanded, encoded_states], dim=-1)
        next_state_logits = self.transition_net(state_input)
        
        # Update current state using last token's prediction
        with torch.no_grad():
            last_token_prediction = next_state_logits[:, -1, :].mean(dim=0)
            predicted_state_idx = torch.argmax(last_token_prediction)
            new_state = self.state_embeddings(predicted_state_idx)
            
            # Exponential moving average update
            alpha = 0.1
            self.current_state.data = (1 - alpha) * self.current_state.data + alpha * new_state
        
        # Generate action predictions for planning
        planning_input = torch.cat([current_state_expanded, hidden_states], dim=-1)
        action_logits = self.planning_net(planning_input)
        
        # Decode back to hidden space
        state_enhanced = self.state_decoder(encoded_states)
        
        return state_enhanced, next_state_logits, action_logits


class CognitiveTransformerLayer(nn.Module):
    """Single transformer layer with cognitive modules"""
    
    def __init__(self, config: CognitiveConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Standard transformer components
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = CognitiveAttention(config, layer_idx)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CognitiveMLP(config)
        
        # Cognitive modules (conditionally added)
        self.working_memory = None
        self.reasoning_module = None
        self.state_tracking = None
        self.tool_calling = None
        
        # Add working memory every few layers
        if layer_idx % config.working_memory_update_freq == 0:
            self.working_memory = WorkingMemoryModule(config)
        
        # Add reasoning in specified layers
        if layer_idx in config.reasoning_layers:
            self.reasoning_module = ReasoningModule(config)
            
        # Add state tracking in final layer
        if layer_idx == config.num_hidden_layers - 1:
            self.state_tracking = StateTrackingModule(config)
            
        # Add tool calling in later layers
        if layer_idx >= config.num_hidden_layers - 3:
            self.tool_calling = ToolCallingModule(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        tool_registry: Optional[Dict] = None,
        **kwargs,
    ):
        residual = hidden_states
        
        # Layer norm and self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # Add residual
        hidden_states = residual + hidden_states
        
        # Working memory processing
        if self.working_memory is not None:
            memory_enhanced = self.working_memory(hidden_states, update_memory=True)
            hidden_states = hidden_states + 0.1 * memory_enhanced  # Scale factor
        
        # MLP processing
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Cognitive processing results
        reasoning_outputs = None
        state_outputs = None
        tool_outputs = None
        
        # Reasoning processing
        if self.reasoning_module is not None:
            reasoning_enhanced, step_logits, confidence = self.reasoning_module(
                hidden_states, attention_mask
            )
            hidden_states = reasoning_enhanced
            reasoning_outputs = {
                'step_logits': step_logits,
                'confidence': confidence
            }
        
        # State tracking processing
        if self.state_tracking is not None:
            state_enhanced, state_logits, action_logits = self.state_tracking(hidden_states)
            hidden_states = state_enhanced
            state_outputs = {
                'state_logits': state_logits,
                'action_logits': action_logits
            }
        
        # Tool calling processing
        if self.tool_calling is not None:
            tool_enhanced, tool_call_info = self.tool_calling(
                hidden_states, tool_registry, execute_tools=False
            )
            hidden_states = tool_enhanced
            tool_outputs = tool_call_info
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
        
        if use_cache:
            outputs += (present_key_value,)
            
        # Add cognitive outputs
        if reasoning_outputs is not None:
            outputs += (reasoning_outputs,)
        if state_outputs is not None:
            outputs += (state_outputs,)
        if tool_outputs is not None:
            outputs += (tool_outputs,)
            
        return outputs


class CognitiveLanguageModel(PreTrainedModel, GenerationMixin):
    """Complete cognitive language model with tool calling capabilities"""
    
    config_class = CognitiveConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CognitiveTransformerLayer"]
    
    def __init__(self, config: CognitiveConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CognitiveTransformerLayer(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tool registry (to be populated at runtime)
        self.tool_registry = {}
        
        # Initialize weights
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def register_tool(self, name: str, function: callable, description: str, parameters: Dict):
        """Register a new tool for the model to use"""
        tool_id = len(self.tool_registry)
        if tool_id < self.config.max_tools:
            self.tool_registry[name] = {
                'id': tool_id,
                'function': function,
                'description': description,
                'parameters': parameters
            }
            return tool_id
        else:
            raise ValueError(f"Maximum number of tools ({self.config.max_tools}) exceeded")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Initialize caches
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Process through layers
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                tool_registry=self.tool_registry,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        # Language modeling head
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + (next_cache,) if use_cache else (logits,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


# For compatibility with transformers library
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache

logger = logging.get_logger(__name__)

# Register the model for auto loading
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("cognitive_transformer", CognitiveConfig)
AutoModel.register(CognitiveConfig, CognitiveLanguageModel)
AutoModelForCausalLM.register(CognitiveConfig, CognitiveLanguageModel)