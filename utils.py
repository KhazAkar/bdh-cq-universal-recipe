"""Utilities for BDH-CQ"""
import json
import gzip
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import torch
from torch.utils.data import DataLoader

def download_file(url: str, destination: str, force: bool = False) -> str:
    destination = Path(destination)
    if destination.exists() and not force:
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return str(destination)

def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_arc_dataset(split: str = "training", data_dir: str = "./data/arc") -> List[Dict[str, Any]]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    urls = {'training': 'https://storage.googleapis.com/arc-agi/ARC-AGI-1/training.json.gz', 'evaluation': 'https://storage.googleapis.com/arc-agi/ARC-AGI-1/evaluation.json.gz'}
    tasks = []
    for split_name in ([split] if split != 'both' else ['training', 'evaluation']):
        file_path = data_dir / f"{split_name}.json.gz"
        if not file_path.exists():
            try:
                download_file(urls[split_name], file_path)
            except:
                continue
        with gzip.open(file_path, 'rt') as f:
            tasks.extend(json.load(f))
    return tasks

def create_dataloaders(tasks: List[Dict[str, Any]], batch_size: int = 32, max_demos: int = 3, test_size: float = 0.2, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    from .training import ARCTaskDataset, arc_collate_fn
    import random
    random.seed(42)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * (1 - test_size))
    train_dataset = ARCTaskDataset(tasks[:split_idx], max_demos)
    eval_dataset = ARCTaskDataset(tasks[split_idx:], max_demos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=arc_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=arc_collate_fn)
    return train_loader, eval_loader

def grid_to_tensor(grid: Union[List[List[int]], Any]) -> torch.Tensor:
    import numpy as np
    if isinstance(grid, np.ndarray):
        return torch.from_numpy(grid).long()
    return torch.tensor(grid, dtype=torch.long)

def tensor_to_grid(tensor: torch.Tensor) -> List[List[int]]:
    return tensor.tolist()
