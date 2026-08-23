"""PyTorch implementations for BDH-CQ"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class MemoryBuffer(nn.Module):
    def __init__(self, hidden_dim: int, memory_dim: Optional[int] = None, update_type: str = "cross_attention", num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim or hidden_dim
        self.memory = nn.Parameter(torch.zeros(1, 1, self.memory_dim))
        self.proj_in = nn.Linear(hidden_dim, self.memory_dim)
        self.proj_out = nn.Linear(self.memory_dim, hidden_dim)
        if update_type == "cross_attention":
            self.update_attn = nn.MultiheadAttention(self.memory_dim, num_heads, dropout=dropout, batch_first=True)
            self.update_norm = nn.LayerNorm(self.memory_dim)
        elif update_type == "mlp":
            self.update_mlp = nn.Sequential(nn.Linear(self.memory_dim + hidden_dim, 4 * self.memory_dim), nn.GELU(), nn.Linear(4 * self.memory_dim, self.memory_dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.drop = nn.Dropout(dropout)
    
    def reset_memory(self):
        nn.init.zeros_(self.memory)
    
    def update(self, demo_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = demo_embeddings.size(0)
        demo_proj = self.proj_in(demo_embeddings)
        memory_expanded = self.memory.expand(batch_size, 1, -1)
        if hasattr(self, 'update_attn'):
            attn_output, _ = self.update_attn(memory_expanded, demo_proj, demo_proj)
            memory_update = self.update_norm(attn_output + memory_expanded)
        else:
            combined = torch.cat([memory_expanded, demo_proj.mean(dim=1, keepdim=True)], dim=-1)
            memory_update = self.update_mlp(combined).unsqueeze(1)
        memory_update = self.alpha * memory_update
        new_memory = memory_expanded + memory_update
        new_memory = self.drop(new_memory)
        self.memory.data = new_memory.detach().mean(dim=0, keepdim=True)
        return new_memory
    
    def get_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        memory_expanded = self.memory.expand(hidden_states.size(0), -1, -1)
        memory_context = self.proj_out(memory_expanded)
        return hidden_states + memory_context.expand(-1, hidden_states.size(1), -1)

class LatentReasoningStep(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff_proj1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        self.ff_proj2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        attn_output = self.drop(attn_output)
        hidden_states = self.norm1(hidden_states + attn_output)
        if memory is not None:
            hidden_states = hidden_states + memory.expand(-1, hidden_states.size(1), -1)
        ff_output = self.ff_proj2(F.gelu(self.ff_proj1(hidden_states)))
        ff_output = self.drop(ff_output)
        return self.norm2(hidden_states + ff_output)

class RecurrentReasoningModule(nn.Module):
    def __init__(self, hidden_dim: int, num_steps: int = 8, num_heads: int = 8, use_residual: bool = True):
        super().__init__()
        self.steps = nn.ModuleList([LatentReasoningStep(hidden_dim, num_heads) for _ in range(num_steps)])
        self.use_residual = use_residual
        if use_residual:
            self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor, memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = hidden_states
        for step in self.steps:
            next_states = step(current, memory)
            current = self.norm(current + next_states) if self.use_residual else next_states
        return current

class BDHCQModel(nn.Module):
    def __init__(self, base_model: nn.Module, memory_type: str = "single", num_reasoning_steps: int = 8, hidden_dim: Optional[int] = None):
        super().__init__()
        self.base_model = base_model
        self.num_reasoning_steps = num_reasoning_steps
        if hidden_dim is None:
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                hidden_dim = base_model.config.hidden_size
            else:
                raise ValueError("Provide hidden_dim or ensure base_model has config.hidden_size")
        self.hidden_dim = hidden_dim
        self.memory_buffer = MemoryBuffer(hidden_dim=hidden_dim)
        self.reasoning_module = RecurrentReasoningModule(hidden_dim=hidden_dim, num_steps=num_reasoning_steps)
        self.current_reasoning_steps = num_reasoning_steps
    
    def reset_memory(self):
        self.memory_buffer.reset_memory()
    
    def process_demonstration(self, demo_input: torch.Tensor, demo_output: torch.Tensor):
        with torch.no_grad():
            if hasattr(self.base_model, 'get_input_embeddings'):
                input_embeds = self.base_model.get_input_embeddings()(demo_input)
                output_embeds = self.base_model.get_input_embeddings()(demo_output)
            else:
                input_embeds, output_embeds = demo_input, demo_output
        demo_embedding = torch.cat([input_embeds, output_embeds], dim=1)
        self.memory_buffer.update(demo_embedding)
    
    def process_demonstrations(self, demonstrations: List[Tuple[torch.Tensor, torch.Tensor]]):
        for demo_input, demo_output in demonstrations:
            self.process_demonstration(demo_input, demo_output)
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, 'get_input_embeddings'):
            hidden_states = self.base_model.get_input_embeddings()(query)
        else:
            hidden_states = query
        return self.memory_buffer.get_context(hidden_states)
    
    def latent_reasoning(self, hidden_states: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.current_reasoning_steps
        batch_size = hidden_states.size(0)
        memory_expanded = self.memory_buffer.memory.expand(batch_size, -1, -1)
        memory_context = self.memory_buffer.proj_out(memory_expanded)
        return self.reasoning_module(hidden_states, memory_context)
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hasattr(self.base_model, 'lm_head'):
            return self.base_model.lm_head(hidden_states)
        return hidden_states
    
    def forward(self, input_ids=None, demonstrations=None, query=None, reasoning_steps=None, labels=None, **kwargs):
        if demonstrations is None and query is None:
            return self.base_model(input_ids=input_ids, labels=labels, **kwargs)
        self.reset_memory()
        if demonstrations is not None:
            self.process_demonstrations(demonstrations)
        hidden_states = self.encode_query(query)
        reasoned_states = self.latent_reasoning(hidden_states, reasoning_steps)
        logits = self.decode(reasoned_states)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits
