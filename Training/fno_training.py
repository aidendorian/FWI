import torch
import os
from tqdm.auto import tqdm

from dataloaders.source_waveforms import source_info
from dataloaders.velocity_models import velocity_models
from FNO.losses import FNO_loss
from FNO.fno import ElasticWaveSolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 8
LR = 3e-4
GRAD_CLIP = 1.0
LAMBDA_DATA = 1.0
LAMBDA_PDE  = 0.1
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 5

train_vel = velocity_models(dir="data/velocity_models",
                            start_index=41,
                            end_index=60)
train_src = source_info(dir="data/source_info",
                        start_index=41,
                        end_index=60)

val_vel = velocity_models(dir="data/test/velocity_models",
                          start_index=101,
                          end_index=108)
val_src = source_info(dir="data/test/source_info",
                      start_index=101,
                      end_index=108)

model = ElasticWaveSolver().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(vel_loader, src_loader, train: bool):

    model.train() if train else model.eval()
    total_loss   = 0.
    n_batches    = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for vel, src in tqdm(zip(vel_loader, src_loader), total=min(len(vel_loader), len(src_loader)), desc="train" if train else "val", leave=False,):
            vel = vel.to(DEVICE)
            src = src.to(DEVICE)

            Vp = vel[:, 0:1, :, :]
            Vs = vel[:, 1:2, :, :]
            density = vel[:, 2:3, :, :]

            pred = model(vel)
            loss = FNO_loss(Vp, Vs, density, pred, src, lambda_data=LAMBDA_DATA, lambda_pde=LAMBDA_PDE)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


best_val_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_vel, train_src, train=True)
    val_loss   = run_epoch(val_vel,   val_src,   train=False)

    scheduler.step(val_loss)

    if epoch % CHECKPOINT_EVERY == 0 or val_loss < best_val_loss:
        tag = "best" if val_loss < best_val_loss else f"epoch{epoch:03d}"
        path = os.path.join(CHECKPOINT_DIR, f"fno_{tag}.pt")
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,}, path)

    if val_loss < best_val_loss:
        best_val_loss = val_loss