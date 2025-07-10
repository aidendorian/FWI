import torch
import os
import numpy as np

class velocity_models(torch.utils.data.Dataset):
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
    
    def __getitem__(self, index):
        file_idx, sample_idx = self.file_and_index(index)
        
        vp_path = os.path.join(self.dir, f"vp_{file_idx}.npy")
        vs_path = os.path.join(self.dir, f"vs_{file_idx}.npy")
        density_path = os.path.join(self.dir, f"density_{file_idx}.npy")
        pm_path = os.path.join(self.dir, f"pm_{file_idx}.npy")
        pr_path = os.path.join(self.dir, f"pr_{file_idx}.npy")
        
        vp = np.load(vp_path, mmap_mode="r")[sample_idx]
        vs = np.load(vs_path, mmap_mode="r")[sample_idx]
        density = np.load(density_path, mmap_mode="r")[sample_idx]
        pm = np.load(pm_path, mmap_mode="r")[sample_idx]
        pr = np.load(pr_path, mmap_mode="r")[sample_idx]
        
        data_concat = np.concatenate([vp, vs, density, pm, pr], axis=0)
        data = torch.tensor(data_concat, dtype=torch.float32)
        
        return data
    
def dataloader_velocity_models(batch_size=16, num_workers=4, shuffle=True):
    dataset = velocity_models(dir="data/velocity_models",
                              start_index = 41,
                              end_index = 60)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=shuffle)
    return dataloader