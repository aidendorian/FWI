# Elastic Full Waveform Inversion (FWI) using Physics-Informed GAN & FNO

This repository implements a **Physics-Informed Generative Adversarial Network (GAN)** for **Elastic Full Waveform Inversion (FWI)**. It combines a **U-Net Generator**, a **Wasserstein Discriminator**, and a **Fourier Neural Operator (FNO)** based **Elastic Wave Solver**. The pipeline is designed for reconstructing sub-surface Earth properties (e.g., Vp, Vs, ρ) from seismic waveforms (u_x, u_z).

---

## Key Features

- **Physics-Informed Training** with elastic wave equation residual loss  
- **FNO-based Elastic Wave Solver** for waveform simulation  
- **GAN Training**: Generator learns Earth model from seismic data; Discriminator ensures realism  
- **Multi-parameter Inversion**: Includes P-wave (Vp), S-wave (Vs), density (ρ), Poisson’s ratio (pr), and Young’s modulus (pm)  
- **Fully differentiable** end-to-end pipeline using PyTorch 

---

## GAN Architecture Overview

```
Seismic Waveforms (uₓ, u_z)
│
▼
┌───────────────┐
│ Generator │───> (Vp, Vs, ρ, pr, pm)
└───────────────┘
│
▼
┌──────────────────────┐
│ Elastic Wave Solver │ (FNO)
└──────────────────────┘
│
▼
┌──────────────────────┐
│ Predicted Waveform û │
└──────────────────────┘
│
▼
┌──────────────────────┐
│ Discriminator (WGAN) │
└──────────────────────┘
```

### Loss Functions

- **Adversarial Loss (WGAN):**  
  `L_adv = -D(G(z)).mean()`

- **Data Misfit Loss (MSE):**  
  `L_data = MSE(FNO(V_pred), waveform_obs)`

- **Physics Loss (PDE Residual):**  
  `L_pde = || ρ ∂²u/∂t² - ∇·σ ||²`

- **Total Generator Loss:**  
  `L_total = L_adv + λ_data * L_data + λ_pde * L_pde`

---

## Dataset Format

dataset link
Each sample consists of:

| File            | Shape             | Description              |
|-----------------|------------------|--------------------------|
| data_x_i.npy    | [B, 5, 1000, 70]  | uₓ waveform              |
| data_z_i.npy    | [B, 5, 1000, 70]  | u_z waveform             |
| vp_i.npy        | [B, 1, 70, 70]    | P-wave velocity          |
| vs_i.npy        | [B, 1, 70, 70]    | S-wave velocity          |
| rho_i.npy       | [B, 1, 70, 70]    | Density                  |
| pr_i.npy        | [B, 1, 70, 70]    | Poisson's ratio          |
| pm_i.npy        | [B, 1, 70, 70]    | Young’s modulus          |

Inputs to the Generator: uₓ + u_z → [B, 10, 1000, 70]  
Outputs from Generator: [B, 5, 70, 70] → input to FNO for waveform reconstruction

---


