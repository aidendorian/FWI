import torch
import os

def save_checkpoint(path, epoch, model, optimizer, scheduler, train_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_loss": train_loss,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,}, 
               path)

def load_checkpoint(path, model, optimizer, scheduler):
    if not os.path.exists(path):
        raise ValueError("Path NOT FOUND")
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"])
    if "cuda_rng_state" in ckpt and ckpt["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    
    return ckpt["epoch"]