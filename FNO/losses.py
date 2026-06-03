import torch
from torch.nn import functional as f

def pde_res(u, Vp, Vs, density, dx=1., dt=1.):
    ux = u[:, :5, :, :]
    uz = u[:, 5:, :, :]

    utt_x = (ux[:, :, 2:, :] - 2.*ux[:, :, 1:-1, :] + ux[:, :, :-2, :])/(dt**2)
    utt_z = (uz[:, :, 2:, :] - 2.*uz[:, :, 1:-1, :] + uz[:, :, :-2, :])/(dt**2)
    
    i_ux = ux[:, :, 1:-1, :]
    i_uz = uz[:, :, 1:-1, :]
    
    dux_dx = (i_ux[:, :, 2:, :] -i_ux[:, :, :-2, :])/(2.*dx)
    duz_dx = (i_uz[:, :, 2:, :] -i_uz[:, :, :-2, :])/(2.*dx)
    
    dux_dz = ((ux[:, :, 2:, :] - ux[:, :, :-2, :])/(2.*dx))[:, :, :, 1:-1]
    duz_dz = ((uz[:, :, 2:, :] - uz[:, :, :-2, :])/(2.*dx))[:, :, :, 1:-1]
    
    mu  = (density * Vs ** 2).squeeze(1)
    lam = (density * (Vp**2 - 2*Vs**2)).squeeze(1)
    
    mu_r  = mu[:, :1, 1:-1].unsqueeze(1)
    lam_r = lam[:, :1, 1:-1].unsqueeze(1)
    
    sigma_xx = (lam_r + 2 * mu_r) * dux_dx + lam_r * duz_dz
    sigma_zz = (lam_r + 2 * mu_r) * duz_dz + lam_r * dux_dx
    sigma_xz = mu_r * (dux_dz + duz_dx)
    
    div_sigma_x = ((sigma_xx[..., 2:] - sigma_xx[..., :-2])/(2.*dx))[:, :, 1:-1, :] + ((sigma_xz[:, :, 2:, :]-sigma_xz[:, :, :-2, :])/(2.*dx))[:, :, :, 1:-1]
    div_sigma_z = ((sigma_xz[..., 2:] - sigma_xz[..., :-2])/(2.*dx))[:, :, 1:-1, :] + ((sigma_zz[:, :, 2:, :]-sigma_zz[:, :, :-2, :])/(2.*dx))[:, :, :, 1:-1]
    
    utt_x_trim = utt_x[:, :, 1:-1, 1:-1]
    utt_z_trim = utt_z[:, :, 1:-1, 1:-1]
 
    rho_r = density.squeeze(1)[:, :1, 1:-1].unsqueeze(1)[:, :, :, 1:-1]
 
    res_x = rho_r * utt_x_trim - div_sigma_x
    res_z = rho_r * utt_z_trim - div_sigma_z
 
    residual_loss = (res_x ** 2 + res_z ** 2).mean()
 
    return residual_loss


def loss(pred_value,obs_value):
     mse_loss = f.mse_loss(pred_value,obs_value)
     return mse_loss

def FNO_loss(Vp,Vs,density,value_pred,value_obs,lambda_data=1.0,lambda_pde=0.1):

    L_data=loss(value_pred,value_obs)
    L_pde=pde_res(value_pred,Vp,Vs,density)
    weighted_data = L_data*lambda_data
    weighted_pde = L_pde*lambda_pde
    total_loss = weighted_data + weighted_pde
    return total_loss