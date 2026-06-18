import torch
import os
import numpy as np
from torch.utils.data import Dataset

PROPERTIES = ("vp", "vs", "density", "pm", "pr")

class velocity_models(Dataset):
    def __init__(self, data_dir: str, start_index: int, end_index: int):
        self.data_dir = data_dir
        self.start_index = start_index
        self.end_index = end_index
        self.samples_in_file = 500
        self.num_files = end_index - start_index + 1
        self.total_samples = self.num_files * self.samples_in_file
        self._cache: dict = {}
 
    def __len__(self) -> int:
        return self.total_samples
 
    def _file_and_sample(self, idx: int):
        file_offset = idx // self.samples_in_file
        sample_idx  = idx  % self.samples_in_file
        file_idx    = self.start_index + file_offset
        return file_idx, sample_idx
 
    def _get_mmaps(self, file_idx: int):
        if file_idx not in self._cache:
            self._cache[file_idx] = tuple(np.load(os.path.join(self.data_dir, f"{prop}_{file_idx}.npy"),mmap_mode="r")for prop in PROPERTIES)
        return self._cache[file_idx]
 
    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, sample_idx = self._file_and_sample(idx)
        mmaps = self._get_mmaps(file_idx)
        concat = np.concatenate([np.array(m[sample_idx]) for m in mmaps],axis=0)
        return torch.tensor(concat, dtype=torch.float32)