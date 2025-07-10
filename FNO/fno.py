import torch

class SpectralConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.scale = 1/(in_channels*out_channels)
        self.weights = torch.nn.Parameter(
            self.scale*(
                torch.randn(in_channels,
                            out_channels,
                            modes_x,
                            modes_y) + 1j*torch.randn(in_channels,
                                                      out_channels,
                                                      modes_x,
                                                      modes_y)
            )
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = torch.einsum("bixy,ioxy->boxy",
                                                                  x_ft[:, :, :self.modes_x, :self.modes_y],
                                                                  self.weights)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x
    
class FNOBlock(torch.nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.spec_conv = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = torch.nn.Conv2d(width, width, kernel_size=1)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.15)
        
    def forward(self, x):
        res = x
        x1 = self.spec_conv(x)
        x2 = self.pointwise(x)
        x = self.activation(x1+x2)
        x = self.dropout(x)
        return x+res
    
class ElasticWaveSolver(torch.nn.Module):
    def __init__(self, in_channels=5, out_channels=10, width=64, x_modes=16, y_modes=16, depth=4):
        super().__init__()
        self.input_projection = torch.nn.Conv2d(in_channels, width, kernel_size=1)
        self.fno_blocks = torch.nn.Sequential(*[FNOBlock(width, x_modes, y_modes) for _ in range(depth)])
        self.output_projection = torch.nn.Sequential(
            torch.nn.Conv2d(width, 128, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.fno_blocks(x)
        x = self.output_projection(x)
        return x