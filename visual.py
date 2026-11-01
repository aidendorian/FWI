import torch
import matplotlib.pyplot as plt
import numpy as np
from FNO.fno import ElasticWaveSolver
from dataloaders.dataloader import make_val_loader

VP_MIN, VP_MAX   = 1500.0, 4500.0
VS_MIN, VS_MAX   = 612.0,  3000.0
RHO_MIN, RHO_MAX = 1.0,    3.0

def normalize_vel(vel):
    vel = vel.clone()
    vel[:, 0:1] = 2.0 * (vel[:, 0:1] - VP_MIN)  / (VP_MAX  - VP_MIN) - 1.0
    vel[:, 1:2] = 2.0 * (vel[:, 1:2] - VS_MIN)  / (VS_MAX  - VS_MIN) - 1.0
    vel[:, 2:3] = 2.0 * (vel[:, 2:3] - RHO_MIN) / (RHO_MAX - RHO_MIN) - 1.0
    return vel

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "Training/checkpoints/fno_epoch13.pt"
N_SAMPLES  = 4
T_SLICES   = [0, 250, 500, 750]

model = ElasticWaveSolver().to(DEVICE)
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

val_loader = make_val_loader(batch_size=N_SAMPLES)
src, vel = next(iter(val_loader))
src = src.to(DEVICE)
vel = vel.to(DEVICE)

with torch.no_grad():
    pred = model(normalize_vel(vel))

pred_np = pred.cpu().float().numpy()
src_np  = src.cpu().float().numpy()
vel_np  = vel.cpu().float().numpy()

print("GT   min/max:", src_np.min(), src_np.max())
print("Pred min/max:", pred_np.min(), pred_np.max())
print("GT   std:", src_np.std())
print("Pred std:", pred_np.std())

# one figure per sample, rows = time slices, cols = GT | Pred | Diff | Vp
T_SLICES  = [0, 250, 500, 750]
CHANNEL   = 0  # ux channel

for i in range(N_SAMPLES):
    fig, axes = plt.subplots(len(T_SLICES), 4, figsize=(16, 4 * len(T_SLICES)))
    fig.suptitle(f"Sample {i}", fontsize=14)

    for row, t in enumerate(T_SLICES):
        gt   = src_np[i, CHANNEL, t, :]     # (W,)  — one time row
        pr   = pred_np[i, CHANNEL, t, :]
        
        # for 2D imshow show the full (time, space) seismogram instead
        gt2d   = src_np[i, CHANNEL]          # (1000, 70)
        pr2d   = pred_np[i, CHANNEL]
        diff2d = gt2d - pr2d
        vmax   = max(np.abs(gt2d).max(), np.abs(pr2d).max()) + 1e-8

        axes[row, 0].imshow(gt2d,   aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[row, 0].axhline(t, color="yellow", linewidth=0.8)
        axes[row, 0].set_title(f"GT ux (t={t} marked)")

        axes[row, 1].imshow(pr2d,   aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[row, 1].axhline(t, color="yellow", linewidth=0.8)
        axes[row, 1].set_title(f"Pred ux (t={t} marked)")

        axes[row, 2].imshow(diff2d, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[row, 2].axhline(t, color="yellow", linewidth=0.8)
        axes[row, 2].set_title(f"Diff (t={t} marked)")

        if row == 0:
            im = axes[row, 3].imshow(vel_np[i, 0], aspect="auto", cmap="jet")
            axes[row, 3].set_title("GT Vp")
            plt.colorbar(im, ax=axes[row, 3])
        else:
            axes[row, 3].axis("off")

    plt.tight_layout()
    plt.savefig(f"eval_sample{i}.png", dpi=150)
    plt.close()
    print(f"saved eval_sample{i}.png")