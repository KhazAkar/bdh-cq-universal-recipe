"""Training utilities for BDH-CQ"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any, Tuple
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

class ARCTaskDataset(Dataset):
    def __init__(self, tasks: List[Dict[str, Any]], max_demos: int = 3):
        self.tasks = [t for t in tasks if len(t.get('train', [])) > 0]
        self.max_demos = max_demos
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        demos = task.get('train', [])[:self.max_demos]
        test_ex = task.get('test', [task['train'][-1]])[0]
        return {'demonstrations': demos, 'query': test_ex['input'], 'target': test_ex['output']}

def arc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_demos = max(len(s['demonstrations']) for s in batch)
    demo_inputs, demo_outputs, queries, targets = [], [], [], []
    for sample in batch:
        demos = sample['demonstrations']
        while len(demos) < max_demos:
            blank = (torch.zeros_like(demos[0][0]), torch.zeros_like(demos[0][1]))
            demos.append(blank)
        demo_inputs.append(torch.stack([d[0] for d in demos]))
        demo_outputs.append(torch.stack([d[1] for d in demos]))
        queries.append(sample['query'])
        targets.append(sample['target'])
    return {'demo_inputs': torch.stack(demo_inputs), 'demo_outputs': torch.stack(demo_outputs), 'query': torch.stack(queries), 'target': torch.stack(targets)}

class CurriculumCallback(TrainerCallback):
    def __init__(self, schedule: List[Tuple[int, int]] = None):
        self.schedule = schedule or [(0, 1), (3, 2), (6, 4), (8, 8)]
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        reasoning_steps = 1
        for schedule_epoch, steps in self.schedule:
            if epoch >= schedule_epoch:
                reasoning_steps = steps
        if hasattr(state.model, 'current_reasoning_steps'):
            setattr(state.model, 'current_reasoning_steps', reasoning_steps)

class MemoryDropoutCallback(TrainerCallback):
    def __init__(self, dropout_prob: float = 0.1, dropout_rate: float = 0.5):
        self.dropout_prob = dropout_prob
        self.dropout_rate = dropout_rate
    
    def on_step_begin(self, args, state, control, **kwargs):
        import torch
        if torch.rand(1).item() < self.dropout_prob:
            if hasattr(state.model, 'memory_buffer'):
                import torch.nn.functional as F
                mask = torch.bernoulli(torch.ones_like(state.model.memory_buffer.memory) * (1 - self.dropout_rate))
                state.model.memory_buffer.memory.data = state.model.memory_buffer.memory.data * mask / (1 - self.dropout_rate)

class BDHCQTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        demo_inputs = inputs.get('demo_inputs')
        demo_outputs = inputs.get('demo_outputs')
        query = inputs.get('query')
        labels = inputs.get('target')
        if demo_inputs is not None and demo_outputs is not None:
            batch_size = demo_inputs.size(0)
            num_demos = demo_inputs.size(1)
            demonstrations = []
            for b in range(batch_size):
                batch_demos = []
                for d in range(num_demos):
                    if demo_inputs[b, d].sum() == 0:
                        continue
                    batch_demos.append((demo_inputs[b, d], demo_outputs[b, d]))
                demonstrations.append(batch_demos)
        else:
            demonstrations = None
        outputs = model(demonstrations=demonstrations, query=query, labels=labels)
        loss = outputs
        return (loss, outputs) if return_outputs else loss

def create_training_config(model_size: str = "7b", batch_size: int = 32, learning_rate: float = 1e-5, num_epochs: int = 10, **kwargs) -> TrainingArguments:
    batch_sizes = {'7b': 32, '13b': 16, '30b': 8, '70b': 4}
    batch_size = batch_sizes.get(model_size, batch_size)
    return TrainingArguments(
        output_dir="./bdh_cq_results", per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs,
        learning_rate=learning_rate, warmup_steps=1000, fp16=True, bf16=False,
        gradient_accumulation_steps=1, logging_steps=100, save_steps=1000,
        evaluation_strategy="steps", eval_steps=500, save_total_limit=2,
        remove_unused_columns=True, report_to="none", **kwargs)
