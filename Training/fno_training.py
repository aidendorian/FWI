import torch
from dataloaders.source_waveforms import dataloader_source_waveforms
from dataloaders.velocity_models import dataloader_velocity_models
from FNO.losses import FNO_loss
from FNO.fno import ElasticWaveSolver
from tqdm.auto import tqdm

true_vel = dataloader_velocity_models()
true_src = dataloader_source_waveforms()

epochs = 5
batch_size = 16
model = ElasticWaveSolver().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for (vel, src) in tqdm(zip(true_vel, true_src)):
        x, y = vel.to("cuda"), src.to("cuda")
        pred_vel = model(x)
        
        Vp = x[:, 0:1, :, :]
        Vs = x[:, 1:2, :, :]
        rho = x[:, 2:3, :, :]
        
        loss = FNO_loss(Vp, Vs, rho, pred_vel, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = total_loss / len(true_src)
    print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.6f}")