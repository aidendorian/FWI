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
    
def generator_loss(Discriminator, fake_samples, waveform_pred, waveform_obs, density, lambda_data=1., lambda_pde=1.):
    """Computes total Generator Loss

    Args:
        Discriminator : Discriminator Model
        fake_samples : Samples from Generator
        waveform_pred : Waveform Predicted with Elastic Wave Equation Solver
        waveform_obs : Observed Waveform from Earth Data
        density : Density of the substructure
        lambda_data : Data Misfit weight. Defaults to 1.0
        lambda_pde : PDE weight. Defaults to 1.0
    """
    L_adv = adversarial_loss(Discriminator, fake_samples)
    L_data = data_misfit(waveform_pred, waveform_obs)
    L_pde = pde_res(waveform_pred, density)
    
    return L_adv + lambda_data * L_data + lambda_pde * L_pde