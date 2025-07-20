import torch

def adversarial_loss(Discriminator, fake_samples):
    """Adversarial Loss for Generator

    Args:
        Discriminator : Discriminator model
        fake_samples : Generated data from the Generator
    """
    return -Discriminator(fake_samples).mean()

def data_misfit(predicted_waveform, observed_waveform):
    """Data Misfit - Comparing FNO outputs and observed waveform

    Args:
        predicted_waveform : Waveform from Elastic Wave Equation Solver
        observed_waveform : Waveforms as observed from Earth data  
    """
    return torch.nn.functional.mse_loss(predicted_waveform, observed_waveform)

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
    
def generator_loss(Discriminator, fake_samples, waveform_pred, waveform_obs, Vp, Vs, density, lambda_data=1., lambda_pde=1.):
    """Computes total Generator Loss

    Args:
        Discriminator : Discriminator Model
        fake_samples : Samples from Generator
        waveform_pred : Waveform Predicted with Elastic Wave Equation Solver
        waveform_obs : Observed Waveform from Earth Data
        Vp : P waves
        Vs : S waves
        density : Density of the substructure
        lambda_data : Data Misfit weight. Defaults to 1.0
        lambda_pde : PDE weight. Defaults to 1.0
    """
    L_adv = adversarial_loss(Discriminator, fake_samples)
    L_data = data_misfit(waveform_pred, waveform_obs)
    L_pde = pde_res(waveform_pred, Vp, Vs, density)
    
    return L_adv + lambda_data * L_data + lambda_pde * L_pde