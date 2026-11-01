import torch
import os
from tqdm.auto import tqdm
from torch.amp.autocast_mode import autocast
from dataloaders.dataloader import make_train_loader, make_val_loader
from FNO.losses import FNO_loss
from FNO.fno import ElasticWaveSolver
from Training.checkpoint import load_checkpoint, save_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4
LR = 3e-4
GRAD_CLIP = 1.0
LAMBDA_DATA = 1.0
LAMBDA_PDE = 1e-4
CHECKPOINT_DIR = "Training/checkpoints"
RESUME_PATH = "Training/checkpoints/fno_epoch12.pt"
CHECKPOINT_EVERY = 1
VAL_EVERY = 5
RESUME = True

VP_MIN, VP_MAX  = 1500.0, 4500.0
VS_MIN, VS_MAX  = 612.0, 3000.0
RHO_MIN, RHO_MAX = 1.0, 3.0
SRC_MIN, SRC_MAX = -3.5,   3.5

def normalize_vel(vel):
    vel = vel.clone()
    vel[:, 0:1] = 2.0*(vel[:, 0:1]- VP_MIN)/ (VP_MAX-VP_MIN)-1.0
    vel[:, 1:2] = 2.0*(vel[:, 1:2]- VS_MIN)/(VS_MAX-VS_MIN)-1.0
    vel[:, 2:3] = 2.0*(vel[:, 2:3]- RHO_MIN) / (RHO_MAX-RHO_MIN)-1.0
    return vel

def normalize_src(src):
    return 2.0 * (src - SRC_MIN) / (SRC_MAX - SRC_MIN) - 1.0

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_loader = make_train_loader(batch_size=BATCH_SIZE)
val_loader   = make_val_loader(batch_size=BATCH_SIZE)

model     = ElasticWaveSolver().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{n_params/1e6:.1f}M parameters")

start_epoch = 0
if RESUME:
    start_epoch = load_checkpoint(RESUME_PATH, model, optimizer, scheduler)

for epoch in range(start_epoch + 1, EPOCHS + 1):
    total_batches = len(train_loader)
    pbar = tqdm(total=total_batches, desc=f"Epoch {epoch:02d}/{EPOCHS}", dynamic_ncols=True)

    model.train()
    train_loss_sum = 0.
    for i, (src, vel) in enumerate(train_loader):
        src = src.to(DEVICE)
        vel = vel.to(DEVICE)
        vel_norm = normalize_vel(vel)

        Vp = vel[:, 0:1]
        Vs = vel[:, 1:2]
        density = vel[:, 2:3]

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(vel_norm)
            loss = FNO_loss(Vp, Vs, density, pred, src,
                            lambda_data=LAMBDA_DATA, lambda_pde=LAMBDA_PDE)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        train_loss_sum += loss.item()
        pbar.set_postfix(phase="train", loss=f"{loss.item():.4f}", avg=f"{train_loss_sum/(i+1):.4f}")
        pbar.update(1)

    train_loss = train_loss_sum / len(train_loader)

    if epoch % VAL_EVERY == 0:
        model.eval()
        val_loss_sum = 0.
        with torch.no_grad():
            for i, (src, vel) in enumerate(val_loader):
                src = src.to(DEVICE)
                vel = vel.to(DEVICE)
                vel_norm = normalize_vel(vel)

                Vp = vel[:, 0:1]
                Vs = vel[:, 1:2]
                density = vel[:, 2:3]

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(vel_norm)
                    loss = FNO_loss(Vp, Vs, density, pred, src,
                                    lambda_data=LAMBDA_DATA, lambda_pde=LAMBDA_PDE)

                val_loss_sum += loss.item()
                pbar.set_postfix(phase="val", loss=f"{loss.item():.4f}", avg=f"{val_loss_sum/(i+1):.4f}")
                pbar.update(1)

        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)
        pbar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
    else:
        pbar.set_postfix(train=f"{train_loss:.4f}")

    pbar.close()

    if epoch % CHECKPOINT_EVERY == 0:
        save_checkpoint(os.path.join(CHECKPOINT_DIR, f"fno_epoch{epoch}.pt"),
                        epoch, model, optimizer, scheduler, train_loss)