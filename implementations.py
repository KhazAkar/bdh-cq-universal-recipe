"""PyTorch implementations for BDH-CQ"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class ModelCapabilities:
    """Detects and stores model reasoning capabilities"""
    has_input_embeddings: bool = False
    has_lm_head: bool = False
    has_generate: bool = False
    has_config: bool = False
    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    is_api_model: bool = False
    
    @classmethod
    def detect(cls, model: nn.Module) -> 'ModelCapabilities':
        """Detect capabilities of a model"""
        caps = cls()
        caps.has_input_embeddings = hasattr(model, 'get_input_embeddings')
        caps.has_lm_head = hasattr(model, 'lm_head')
        caps.has_generate = hasattr(model, 'generate')
        caps.has_config = hasattr(model, 'config')
        
        if caps.has_config:
            if hasattr(model.config, 'hidden_size'):
                caps.hidden_size = model.config.hidden_size
            if hasattr(model.config, 'vocab_size'):
                caps.vocab_size = model.config.vocab_size
        
        # Check for API model indicators
        if hasattr(model, '__class__'):
            class_name = model.__class__.__name__.lower()
            caps.is_api_model = any(x in class_name for x in ['api', 'client', 'remote'])
        
        return caps


class ReasoningAdapter(nn.Module):
    """
    Universal reasoning adapter that provides reasoning capabilities
    for models that don't support it natively (like API-based models).
    
    This adapter wraps any model and provides:
    - Input embedding extraction/projection
    - Output logit projection
    - Reasoning step execution
    """
    
    def __init__(self, model: nn.Module, hidden_dim: Optional[int] = None, 
                 embedding_dim: Optional[int] = None, vocab_size: Optional[int] = None):
        super().__init__()
        self.model = model
        self.capabilities = ModelCapabilities.detect(model)
        
        # Determine dimensions
        if hidden_dim is None:
            hidden_dim = self.capabilities.hidden_size or 4096  # Default for mistral-large
        
        if embedding_dim is None:
            embedding_dim = hidden_dim
        
        if vocab_size is None:
            vocab_size = self.capabilities.vocab_size or 32000  # Default for mistral
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Create projection layers if model doesn't have native embeddings
        if not self.capabilities.has_input_embeddings:
            # For models without embeddings, we create a learnable embedding layer
            self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
        else:
            self.input_embedding = None
        
        if not self.capabilities.has_lm_head:
            # For models without LM head, create output projection
            self.output_proj = nn.Linear(hidden_dim, vocab_size)
        else:
            self.output_proj = None
        
        # Store reference to model's methods if available
        self._get_input_embeddings = None
        self._lm_head = None
        
        if self.capabilities.has_input_embeddings:
            self._get_input_embeddings = model.get_input_embeddings
        if self.capabilities.has_lm_head:
            self._lm_head = model.lm_head
    
    def get_input_embeddings(self) -> Optional[nn.Module]:
        """Get or create input embedding layer"""
        if self._get_input_embeddings is not None:
            return self._get_input_embeddings()
        return self.input_embedding
    
    def get_lm_head(self) -> Optional[nn.Module]:
        """Get or create output projection layer"""
        if self._lm_head is not None:
            return self._lm_head
        return self.output_proj
    
    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input tokens to embeddings"""
        if self._get_input_embeddings is not None:
            embeddings = self._get_input_embeddings()(input_ids)
        elif self.input_embedding is not None:
            embeddings = self.input_embedding(input_ids)
        else:
            raise ValueError("Cannot embed input - no embedding method available")
        return embeddings
    
    def project_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits"""
        if self._lm_head is not None:
            return self._lm_head(hidden_states)
        elif self.output_proj is not None:
            return self.output_proj(hidden_states)
        else:
            raise ValueError("Cannot project output - no LM head available")


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
    """
    BDH-CQ Model with universal reasoning support.
    
    This model wraps any base model (local or API-based) and provides:
    - Latent reasoning capabilities
    - Memory buffer for in-context learning
    - Automatic adaptation for models without native reasoning support
    
    Args:
        base_model: The base model to wrap (can be any nn.Module)
        num_reasoning_steps: Number of reasoning steps to perform
        hidden_dim: Hidden dimension size (auto-detected if not provided)
        vocab_size: Vocabulary size (auto-detected if not provided)
        use_adapter: Whether to use ReasoningAdapter for models without native support
    """
    
    def __init__(self, base_model: nn.Module, memory_type: str = "single", 
                 num_reasoning_steps: int = 8, hidden_dim: Optional[int] = None,
                 vocab_size: Optional[int] = None, use_adapter: bool = True):
        super().__init__()
        self.base_model = base_model
        self.num_reasoning_steps = num_reasoning_steps
        self.use_adapter = use_adapter
        
        # Detect model capabilities
        self.capabilities = ModelCapabilities.detect(base_model)
        
        # Determine hidden dimension
        if hidden_dim is None:
            hidden_dim = self.capabilities.hidden_size
            if hidden_dim is None:
                # Default for mistral-large and similar models
                hidden_dim = 4096
        
        # Determine vocab size
        if vocab_size is None:
            vocab_size = self.capabilities.vocab_size
            if vocab_size is None:
                vocab_size = 32000  # Default for mistral
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Create reasoning adapter if needed
        if use_adapter and not (self.capabilities.has_input_embeddings and self.capabilities.has_lm_head):
            self.reasoning_adapter = ReasoningAdapter(
                base_model, 
                hidden_dim=hidden_dim,
                embedding_dim=hidden_dim,
                vocab_size=vocab_size
            )
        else:
            self.reasoning_adapter = None
        
        # Initialize memory buffer and reasoning module
        self.memory_buffer = MemoryBuffer(hidden_dim=hidden_dim)
        self.reasoning_module = RecurrentReasoningModule(hidden_dim=hidden_dim, num_steps=num_reasoning_steps)
        self.current_reasoning_steps = num_reasoning_steps
    
    def reset_memory(self):
        """Reset the memory buffer"""
        self.memory_buffer.reset_memory()
    
    def _get_embedding_method(self):
        """Get the appropriate embedding method"""
        if self.reasoning_adapter is not None:
            return self.reasoning_adapter.embed_input
        elif hasattr(self.base_model, 'get_input_embeddings'):
            return lambda x: self.base_model.get_input_embeddings()(x)
        else:
            raise ValueError("No embedding method available")
    
    def _get_lm_head_method(self):
        """Get the appropriate LM head method"""
        if self.reasoning_adapter is not None:
            return self.reasoning_adapter.project_output
        elif hasattr(self.base_model, 'lm_head'):
            return self.base_model.lm_head
        else:
            raise ValueError("No LM head method available")
    
    def process_demonstration(self, demo_input: torch.Tensor, demo_output: torch.Tensor):
        """Process a single demonstration pair"""
        with torch.no_grad():
            try:
                embed_method = self._get_embedding_method()
                input_embeds = embed_method(demo_input)
                output_embeds = embed_method(demo_output)
            except ValueError:
                # Fallback: use inputs directly
                input_embeds, output_embeds = demo_input, demo_output
        demo_embedding = torch.cat([input_embeds, output_embeds], dim=1)
        self.memory_buffer.update(demo_embedding)
    
    def process_demonstrations(self, demonstrations: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Process multiple demonstration pairs"""
        for demo_input, demo_output in demonstrations:
            self.process_demonstration(demo_input, demo_output)
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode a query into hidden states with memory context"""
        try:
            embed_method = self._get_embedding_method()
            hidden_states = embed_method(query)
        except ValueError:
            hidden_states = query
        return self.memory_buffer.get_context(hidden_states)
    
    def latent_reasoning(self, hidden_states: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Perform latent reasoning on hidden states"""
        if num_steps is None:
            num_steps = self.current_reasoning_steps
        batch_size = hidden_states.size(0)
        memory_expanded = self.memory_buffer.memory.expand(batch_size, -1, -1)
        memory_context = self.memory_buffer.proj_out(memory_expanded)
        return self.reasoning_module(hidden_states, memory_context)
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode hidden states to logits"""
        try:
            lm_head = self._get_lm_head_method()
            return lm_head(hidden_states)
        except ValueError:
            return hidden_states
    
    def forward(self, input_ids=None, demonstrations=None, query=None, 
                reasoning_steps=None, labels=None, **kwargs):
        """
        Forward pass with reasoning support.
        
        Args:
            input_ids: Input token IDs (for standard model forward)
            demonstrations: List of (input, output) demonstration pairs
            query: Query input for reasoning
            reasoning_steps: Number of reasoning steps to use
            labels: Target labels for loss computation
            **kwargs: Additional arguments passed to base model
        
        Returns:
            Logits or loss depending on whether labels are provided
        """
        # Standard model forward (no reasoning)
        if demonstrations is None and query is None:
            return self.base_model(input_ids=input_ids, labels=labels, **kwargs)
        
        # Reasoning path
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
    
    def generate_with_reasoning(self, input_ids: torch.Tensor, 
                                 demonstrations: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                                 num_reasoning_steps: Optional[int] = None, 
                                 max_new_tokens: int = 50, **kwargs) -> torch.Tensor:
        """
        Generate output with reasoning for models that support generation.
        
        For models without native generate support, this provides a basic
        implementation using the reasoning path.
        
        Args:
            input_ids: Input token IDs
            demonstrations: Optional demonstration pairs for in-context learning
            num_reasoning_steps: Number of reasoning steps
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        if num_reasoning_steps is None:
            num_reasoning_steps = self.current_reasoning_steps
        
        # Use native generate if available and no reasoning requested
        if (hasattr(self.base_model, 'generate') and 
            demonstrations is None and 
            num_reasoning_steps == 0):
            return self.base_model.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        
        # Reasoning-based generation
        if demonstrations is not None:
            self.reset_memory()
            self.process_demonstrations(demonstrations)
        
        # For now, use a simple greedy approach with reasoning
        # This can be enhanced with proper beam search or sampling
        current_input = input_ids
        generated_ids = []
        
        for _ in range(max_new_tokens):
            hidden_states = self.encode_query(current_input.unsqueeze(0))
            reasoned_states = self.latent_reasoning(hidden_states, num_reasoning_steps)
            logits = self.decode(reasoned_states)
            
            # Get next token (greedy)
            next_token = logits.argmax(dim=-1)
            generated_ids.append(next_token)
            
            # Update input for next step
            current_input = torch.cat([current_input, next_token], dim=-1)
        
        return torch.cat(generated_ids, dim=-1)


class UniversalReasoningWrapper:
    """
    Standalone wrapper that adds reasoning capability to any model.
    
    This is useful for API-based models like mistral-large-3 where you
    can't modify the model architecture directly.
    
    Usage:
        from implementations import UniversalReasoningWrapper
        
        # Wrap an API model
        model = SomeAPIModel()
        reasoning_model = UniversalReasoningWrapper(model, hidden_dim=4096)
        
        # Use with reasoning
        output = reasoning_model.generate_with_reasoning(prompt, demonstrations)
    """
    
    def __init__(self, base_model: Any, hidden_dim: int = 4096, 
                 num_reasoning_steps: int = 8, vocab_size: int = 32000):
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_reasoning_steps = num_reasoning_steps
        
        # Create standalone reasoning components
        self.memory_buffer = MemoryBuffer(hidden_dim=hidden_dim)
        self.reasoning_module = RecurrentReasoningModule(
            hidden_dim=hidden_dim, 
            num_steps=num_reasoning_steps
        )
        
        # Create embedding and output projection for API models
        self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        self.current_reasoning_steps = num_reasoning_steps
    
    def reset_memory(self):
        self.memory_buffer.reset_memory()
    
    def process_demonstrations(self, demonstrations: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Process demonstrations to build memory context"""
        for demo_input, demo_output in demonstrations:
            with torch.no_grad():
                input_embeds = self.input_embedding(demo_input)
                output_embeds = self.input_embedding(demo_output)
            demo_embedding = torch.cat([input_embeds, output_embeds], dim=1)
            self.memory_buffer.update(demo_embedding)
    
    def encode_query(self, query: torch.Tensor) -> torch.Tensor:
        """Encode query with memory context"""
        hidden_states = self.input_embedding(query)
        return self.memory_buffer.get_context(hidden_states)
    
    def latent_reasoning(self, hidden_states: torch.Tensor, num_steps: Optional[int] = None):
        """Perform latent reasoning"""
        if num_steps is None:
            num_steps = self.current_reasoning_steps
        batch_size = hidden_states.size(0)
        memory_expanded = self.memory_buffer.memory.expand(batch_size, -1, -1)
        memory_context = self.memory_buffer.proj_out(memory_expanded)
        return self.reasoning_module(hidden_states, memory_context)
    
    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Decode to logits"""
        return self.output_proj(hidden_states)
    
    def generate_with_reasoning(self, input_ids: torch.Tensor,
                                 demonstrations: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                                 num_reasoning_steps: Optional[int] = None,
                                 max_new_tokens: int = 50) -> torch.Tensor:
        """Generate with reasoning for API models"""
        if num_reasoning_steps is None:
            num_reasoning_steps = self.current_reasoning_steps
        
        self.reset_memory()
        
        if demonstrations is not None:
            self.process_demonstrations(demonstrations)
        
        # Simple greedy generation with reasoning
        current_input = input_ids
        generated_ids = []
        
        for _ in range(max_new_tokens):
            hidden_states = self.encode_query(current_input.unsqueeze(0))
            reasoned_states = self.latent_reasoning(hidden_states, num_reasoning_steps)
            logits = self.decode(reasoned_states)
            
            next_token = logits.argmax(dim=-1)
            generated_ids.append(next_token)
            current_input = torch.cat([current_input, next_token], dim=-1)
        
        return torch.cat(generated_ids, dim=-1)
