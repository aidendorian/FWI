import torch

def pde_res(u, Vp, Vs, density, dx=1., dt=1.):
    """Computes Elastic PDE Residual
    
    Args:
        u : Earth source data
        Vp : P waves
        Vs : S waves
        density : Density of substructure
        dx and dt: Derivatives
    """
    ux = u[:, :5, :, :]
    uz = u[:, 5:, :, :]
    
    u = torch.stack([ux, uz], dim=1)
    u = u.permute(0, 1, 3, 2, 4)
    
    utt_x = (u[:, 0, 2:] - 2*u[:, 0, 1:-1] + u[:, 0, :-2]) / dt**2
    utt_z = (u[:, 1, 2:] - 2*u[:, 1, 1:-1] + u[:, 1, :-2]) / dt**2
    utt = torch.stack([utt_x, utt_z], dim=1)
    
    density_exp = density.unsqueeze(2).expand_as(utt[:, 0])
    mu = density*Vs**2
    lam = density*(Vp**2-2*Vs**2)
    mu = mu.unsqueeze(2).expand_as(utt[:, 0])
    lam = lam.unsqueeze(2).expand_as(utt[:, 0])
    
    def spatial_deriv(f, dim):
        return (f[..., 2:] - f[..., :-2]) / (2*dx)
    
    u_x = u[:, 0, 1:-1]
    u_z = u[:, 1, 1:-1]
    
    dux_dx = spatial_deriv(u_x, dim=-1)
    duz_dz = spatial_deriv(u_z, dim=-2)
    dux_dz = spatial_deriv(u_x, dim=-2)
    duz_dx = spatial_deriv(u_z, dim=-1)
    
    fx = (lam+2*mu)*dux_dx+lam*duz_dz+mu*(dux_dz+duz_dx)
    fz = (lam+2*mu)*duz_dz+lam*dux_dx+mu*(dux_dz+duz_dx)
    f = torch.stack([fx, fz], dim=1)
    
    utt_cropped = utt[..., 1:-1, 1:-1]
    density_cropped = density_exp[..., 1:-1, 1:-1]
    
    residual = density_cropped.unsqueeze(1)*utt_cropped-f
    return torch.mean(residual**2)

def loss(pred_value,obs_value):
     """Args:
        pred_vale:predicted value by FNO
        obs_value: observed value from the data
        """
     return torch.nn.functional.mse_loss(pred_value,obs_value)

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
    L_data=loss(value_pred,value_obs)
    L_pde=pde_res(value_pred,Vp,Vs,density)
    return L_data*lambda_data+L_pde*lambda_pde
