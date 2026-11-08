# BDH-CQ Universal Fine-Tuning Recipe Library

Implementation of universal fine-tuning recipe from **[BDH-CQ: In-Context Learning with Recurrent Latent Reasoning](https://arxiv.org/abs/2608.09888)** (arXiv:2608.09888).

## Key Achievement
- **29.5% pass@2** on ARC-AGI-1
- **$0.0007 per task** (57x cheaper than GPT-5.6 Luna)
- **150M parameter** model

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from bdh_cq_lib import BDHCQModel, load_arc_dataset, create_dataloaders
from transformers import MistralForCausalLM

# Load data
tasks = load_arc_dataset(split='training')
train_loader, eval_loader = create_dataloaders(tasks, batch_size=32)

# Create model
model = BDHCQModel(MistralForCausalLM.from_pretrained("mistralai/mistral-7b"), num_reasoning_steps=8)

# Train
from bdh_cq_lib import create_training_config, BDHCQTrainer
trainer = BDHCQTrainer(model, create_training_config(), train_loader.dataset, eval_loader.dataset)
trainer.train()
```

## Universal Reasoning Support

For models that don't support reasoning natively (like API-based models such as `mistral-large-3`), the library provides automatic reasoning capability through adapters:

### Option 1: UniversalReasoningWrapper (Simplest)

```python
from bdh_cq_lib import UniversalReasoningWrapper

# Wrap any model with reasoning capabilities
reasoning_model = UniversalReasoningWrapper(
    base_model=your_api_model,  # Can be any model object
    hidden_dim=4096,            # For mistral-large-3
    num_reasoning_steps=8,
    vocab_size=32000
)

# Use with demonstrations
demo_input = torch.tensor([[1, 2, 3]])  # Token IDs
demo_output = torch.tensor([[4, 5, 6]])
reasoning_model.process_demonstrations([(demo_input, demo_output)])

# Generate with reasoning
query = torch.tensor([[7, 8, 9]])
generated = reasoning_model.generate_with_reasoning(
    input_ids=query,
    max_new_tokens=50
)
```

### Option 2: BDHCQModel with Automatic Adapter

```python
from bdh_cq_lib import BDHCQModel

# For models without native reasoning support
model = BDHCQModel(
    base_model=your_model,
    num_reasoning_steps=8,
    use_adapter=True  # Automatically creates adapter when needed
)

# Works the same way for all models
output = model(demonstrations=demos, query=query)
```

### Option 3: Direct Component Usage

```python
from bdh_cq_lib import ReasoningAdapter, ModelCapabilities

# Detect model capabilities
caps = ModelCapabilities.detect(your_model)

# Create adapter only if needed
if not (caps.has_input_embeddings and caps.has_lm_head):
    adapter = ReasoningAdapter(your_model, hidden_dim=4096, vocab_size=32000)
```

## Components
- **MemoryBuffer**: Recurrent memory for in-context learning
- **LatentReasoningStep**: Single reasoning step
- **RecurrentReasoningModule**: Multi-step reasoning module
- **BDHCQModel**: Complete model wrapper with automatic adapter support
- **ReasoningAdapter**: Adapts models without native reasoning support
- **UniversalReasoningWrapper**: Standalone wrapper for any model (including API-based)
- **ModelCapabilities**: Auto-detection of model capabilities
- **BDHCQTrainer**: Custom trainer with curriculum
- **ARCTaskDataset**: Dataset for ARC tasks

## API-Based Model Support

For models accessed via API (like `mistral-large-3` through Mistral's API):

```python
from mistralai.client import MistralClient
from bdh_cq_lib import UniversalReasoningWrapper

# Create API client
client = MistralClient(api_key="your_api_key")

# Wrap with reasoning
reasoning_client = UniversalReasoningWrapper(
    base_model=client,
    hidden_dim=4096,
    num_reasoning_steps=8,
    vocab_size=32000
)

# Now use reasoning_client with demonstrations for in-context learning
```

## New Features for Universal Reasoning

### ModelCapabilities
Automatically detects what features a model supports:
- `has_input_embeddings`: Model has `get_input_embeddings()` method
- `has_lm_head`: Model has `lm_head` attribute
- `has_config`: Model has `config` attribute
- `hidden_size`: Detected hidden dimension
- `vocab_size`: Detected vocabulary size
- `is_api_model`: True for API-based models

### ReasoningAdapter
Provides reasoning support for models without native capabilities:
- Creates `nn.Embedding` layer for input token embeddings
- Creates `nn.Linear` layer for output logit projection
- Automatically uses native methods when available

### UniversalReasoningWrapper
Complete standalone wrapper for any model:
- Self-contained reasoning pipeline
- Memory buffer for in-context learning
- Latent reasoning module
- Generation with reasoning
- Works with or without a base model

## License
GPL-3.0-or-later; see [LICENSE](LICENSE).

Unless otherwise noted, the original material in this repository is licensed
under the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version (SPDX: `GPL-3.0-or-later`).
See [LICENSE](LICENSE) for the full license text.

Third-party material retains its own copyright notices and license terms.
