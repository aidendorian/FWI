import torch
import os
import numpy as np

class source_info(torch.utils.data.Dataset):
    def __init__(self, dir, start_index, end_index):
        self.dir = dir
        self.start_index = start_index
        self.end_index = end_index
        self.samples_in_file = 500
        self.num_files = self.end_index - self.start_index + 1
        self.total_samples = self.num_files * self.samples_in_file
        
    def __len__(self):
        return self.total_samples
    
    def file_and_index(self, idx):
        file_offset = idx // self.samples_in_file
        sample_idx = idx % self.samples_in_file
        file_idx = self.start_index + file_offset
        return file_idx, sample_idx       

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_and_index(idx)
        
        x_path = os.path.join(self.dir, f"data_x_{file_idx}.npy")
        z_path = os.path.join(self.dir, f"data_z_{file_idx}.npy")
        
        data_x = np.load(x_path, mmap_mode="r")[sample_idx]
        data_z = np.load(z_path, mmap_mode="r")[sample_idx]
        
        data_concat = np.concatenate([data_x, data_z], axis=0)
        data = torch.tensor(data_concat, dtype=torch.float32)
        
        return data
    
def dataloader_source_waveforms(batch_size=16, num_workers=4, shuffle=True):
    dataset = source_info(dir="data/source_info",
                          start_index=41,
                          end_index=60)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=shuffle)
    return dataloader