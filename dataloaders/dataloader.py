from dataloaders.source_waveforms import source_info
from dataloaders.velocity_models import velocity_models
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    def __init__(self, src_dataset: source_info, vel_dataset: velocity_models):
        if len(src_dataset) != len(vel_dataset):
            raise ValueError(
                f"Dataset length mismatch: source_info has {len(src_dataset)} samples "
                f"but velocity_models has {len(vel_dataset)}. They must be equal."
            )
        self.src = src_dataset
        self.vel = vel_dataset
 
    def __len__(self) -> int:
        return len(self.src)
 
    def __getitem__(self, idx: int):
        return self.src[idx], self.vel[idx]

def make_train_loader(batch_size: int = 8, num_workers: int = 2) -> DataLoader:
    ds = CombinedDataset(src_dataset=source_info(data_dir="data/source_info", start_index=41, end_index=60),
                         vel_dataset=velocity_models(data_dir="data/velocity_models", start_index=41, end_index=60))
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=True)
 
 
def make_val_loader(batch_size: int = 8, num_workers: int = 2) -> DataLoader:
    ds = CombinedDataset(src_dataset=source_info(data_dir="data/test/source_info", start_index=101, end_index=108),
                         vel_dataset=velocity_models(data_dir="data/test/velocity_models", start_index=101, end_index=108))
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      pin_memory=True)