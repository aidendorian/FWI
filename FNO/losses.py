import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def pde_res(u, Vp, Vs, density, dx=1., dt=1.):
    """Computes Elastic PDE Residual
    
    Args:
        u : Earth source data
        Vp : P waves
        Vs : S waves
        density : Density of substructure
        dx and dt: Derivatives
    """
    print(f"=== PDE_RES DEBUG ===")
    print(f"Input u shape: {u.shape}")
    print(f"Vp shape: {Vp.shape}, Vs shape: {Vs.shape}, density shape: {density.shape}")
    print(f"dx: {dx}, dt: {dt}")
    
    ux = u[:, :5, :, :]
    uz = u[:, 5:, :, :]
    print(f"After splitting - ux shape: {ux.shape}, uz shape: {uz.shape}")
    
    u = torch.stack([ux, uz], dim=1)
    print(f"After stacking u shape: {u.shape}")
    
    u = u.permute(0, 1, 3, 2, 4)
    print(f"After permute u shape: {u.shape}")
    
    utt_x = (u[:, 0, 2:] - 2*u[:, 0, 1:-1] + u[:, 0, :-2]) / dt**2
    utt_z = (u[:, 1, 2:] - 2*u[:, 1, 1:-1] + u[:, 1, :-2]) / dt**2
    print(f"Time derivatives - utt_x shape: {utt_x.shape}, utt_z shape: {utt_z.shape}")
    
    utt = torch.stack([utt_x, utt_z], dim=1)
    print(f"Combined utt shape: {utt.shape}")
    
    density_exp = density.unsqueeze(2).expand_as(utt[:, 0])
    print(f"Expanded density shape: {density_exp.shape}")
    
    mu = density*Vs**2
    lam = density*(Vp**2-2*Vs**2)
    print(f"Material parameters - mu range: [{torch.min(mu):.6f}, {torch.max(mu):.6f}]")
    print(f"Material parameters - lam range: [{torch.min(lam):.6f}, {torch.max(lam):.6f}]")
    
    mu = mu.unsqueeze(2).expand_as(utt[:, 0])
    lam = lam.unsqueeze(2).expand_as(utt[:, 0])
    print(f"Expanded mu shape: {mu.shape}, lam shape: {lam.shape}")
    
    def spatial_deriv(f, dim):
        result = (f[..., 2:] - f[..., :-2]) / (2*dx)
        print(f"  Spatial derivative along dim {dim}: input {f.shape} -> output {result.shape}")
        return result
    
    u_x = u[:, 0, 1:-1]
    u_z = u[:, 1, 1:-1]
    print(f"Cropped u_x shape: {u_x.shape}, u_z shape: {u_z.shape}")
    
    print("Computing spatial derivatives...")
    dux_dx = spatial_deriv(u_x, dim=-1)
    duz_dz = spatial_deriv(u_z, dim=-2)
    dux_dz = spatial_deriv(u_x, dim=-2)
    duz_dx = spatial_deriv(u_z, dim=-1)
    
    fx = (lam+2*mu)*dux_dx+lam*duz_dz+mu*(dux_dz+duz_dx)
    fz = (lam+2*mu)*duz_dz+lam*dux_dx+mu*(dux_dz+duz_dx)
    print(f"Force components - fx shape: {fx.shape}, fz shape: {fz.shape}")
    print(f"fx range: [{torch.min(fx):.6f}, {torch.max(fx):.6f}]")
    print(f"fz range: [{torch.min(fz):.6f}, {torch.max(fz):.6f}]")
    
    f = torch.stack([fx, fz], dim=1)
    print(f"Combined force f shape: {f.shape}")
    
    utt_cropped = utt[..., 1:-1, 1:-1]
    density_cropped = density_exp[..., 1:-1, 1:-1]
    print(f"Final cropped shapes - utt: {utt_cropped.shape}, density: {density_cropped.shape}")
    
    residual = density_cropped.unsqueeze(1)*utt_cropped-f
    print(f"Residual shape: {residual.shape}")
    print(f"Residual range: [{torch.min(residual):.6f}, {torch.max(residual):.6f}]")
    
    final_loss = torch.mean(residual**2)
    print(f"PDE residual loss: {final_loss.item():.6f}")
    logger.info(f"PDE residual computed: {final_loss.item():.6f}")
    
    return final_loss

def loss(pred_value,obs_value):
     """Args:
        pred_vale:predicted value by FNO
        obs_value: observed value from the data
        """
     print(f"=== DATA LOSS DEBUG ===")
     print(f"Pred shape: {pred_value.shape}, Obs shape: {obs_value.shape}")
     print(f"Pred range: [{torch.min(pred_value):.6f}, {torch.max(pred_value):.6f}]")
     print(f"Obs range: [{torch.min(obs_value):.6f}, {torch.max(obs_value):.6f}]")
     
     mse_loss = torch.nn.functional.mse_loss(pred_value,obs_value)
     print(f"MSE loss: {mse_loss.item():.6f}")
     logger.info(f"Data loss (MSE): {mse_loss.item():.6f}")
     
     return mse_loss

def FNO_loss(Vp,Vs,density,value_pred,value_obs,lambda_data=1.0,lambda_pde=1.0):
    """Args:
    Discriminator:Discriminator model
    Vp: P waves
    Vs: S waves
    Density: DENSITY
    value_pred: pred_value in passive voice
    value_obs: pred_obs in passive voice
    lambda_pde : PDE weight. Defaults to 1.0
    LET THE RECORDS SHOW THAT I KNOW MY COMMENTS ARE UNPROFESSIONAL.
    """
    print(f"\n=== FNO LOSS DEBUG ===")
    print(f"Lambda values - data: {lambda_data}, pde: {lambda_pde}")
    print(f"Computing data loss...")
    
    L_data=loss(value_pred,value_obs)
    print(f"Computing PDE loss...")
    
    L_pde=pde_res(value_pred,Vp,Vs,density)
    
    weighted_data = L_data*lambda_data
    weighted_pde = L_pde*lambda_pde
    total_loss = weighted_data + weighted_pde
    
    print(f"Loss breakdown:")
    print(f"  Data loss (weighted): {weighted_data.item():.6f}")
    print(f"  PDE loss (weighted): {weighted_pde.item():.6f}")
    print(f"  Total loss: {total_loss.item():.6f}")
    
    logger.info(f"FNO Loss - Data: {weighted_data.item():.6f}, PDE: {weighted_pde.item():.6f}, Total: {total_loss.item():.6f}")
    
    return total_loss