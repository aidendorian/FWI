import torch

class Discriminator(torch.nn.Module):
    def __init__(self, velocity_channels, wave_channels):
        super().__init__()
        channels = velocity_channels + wave_channels
        
        self.model = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.02, inplace=True),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.02, inplace=True),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.02, inplace=True),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)),
            torch.nn.LeakyReLU(0.02, inplace=True),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1))
        )
    
    def forward(self, x):
        velocity = x[:, :5, :, :]
        waveform = x[:, 5:, :, :]
        return self.model(torch.cat([velocity, waveform], dim=1))

def get_discriminator(velocity_channels=5, wave_channels=1):
    discriminator = Discriminator(velocity_channels=velocity_channels,
                                  wave_channels=wave_channels)
    return discriminator