import torch
import logging
from dataloaders.source_waveforms import dataloader_source_waveforms
from dataloaders.velocity_models import dataloader_velocity_models
from FNO.losses import FNO_loss
from FNO.fno import ElasticWaveSolver
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initial setup validation
print("=== TRAINING SETUP ===")
true_vel = dataloader_velocity_models()
true_src = dataloader_source_waveforms()
print(f"Dataloaders loaded - Vel: {len(true_vel) if hasattr(true_vel, '__len__') else 'Unknown'}, Src: {len(true_src) if hasattr(true_src, '__len__') else 'Unknown'}")

epochs = 5
batch_size = 16
model = ElasticWaveSolver().to("cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
print(f"Model: {sum(p.numel() for p in model.parameters())} parameters, LR: {optimizer.param_groups[0]['lr']}")

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (vel, src) in enumerate(tqdm(zip(true_vel, true_src), desc=f"Epoch {epoch+1}")):
        x, y = vel.to("cuda"), src.to("cuda")
        
        # Debug first batch of first epoch only
        if epoch == 0 and batch_idx == 0:
            print(f"First batch shapes - x: {x.shape}, y: {y.shape}")
            print(f"Input ranges - x: [{torch.min(x):.3f}, {torch.max(x):.3f}], y: [{torch.min(y):.3f}, {torch.max(y):.3f}]")
        
        pred_vel = model(x)
        
        Vp = x[:, 0:1, :, :]
        Vs = x[:, 1:2, :, :]
        rho = x[:, 2:3, :, :]
        
        loss = FNO_loss(Vp, Vs, rho, pred_vel, y)
        
        # Check for problematic losses
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss at epoch {epoch+1}, batch {batch_idx+1}: {loss.item()}")
            logger.warning(f"Invalid loss detected at epoch {epoch+1}, batch {batch_idx+1}")
        
        optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradients for first few batches
        if batch_idx < 3:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if grad_norm > 10.0:
                print(f"Large gradient norm at epoch {epoch+1}, batch {batch_idx+1}: {grad_norm:.3f}")
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(true_src)
    print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.6f}")
    logger.info(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.6f}")