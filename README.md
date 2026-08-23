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

## Components
- **MemoryBuffer**: Recurrent memory for in-context learning
- **LatentReasoningStep**: Single reasoning step
- **BDHCQModel**: Complete model wrapper
- **BDHCQTrainer**: Custom trainer with curriculum
- **ARCTaskDataset**: Dataset for ARC tasks

## License
Research and educational use only.
