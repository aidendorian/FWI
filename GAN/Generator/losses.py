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
        predicted_waveform : Waveform from FNO
        observed_waveform : Waveforms as observed from Earth data  
    """
    return torch.nn.functional.mse_loss(predicted_waveform, observed_waveform)

