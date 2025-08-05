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
        self.batch_norm=torch.nn.BatchNorm2d(width)
        self.dropout = torch.nn.Dropout(0.15)
        
    def forward(self, x):
        res = x
        x1 = self.spec_conv(x)
        x2 = self.pointwise(x)
        x = self.activation(x1+x2)
        x=self.batch_norm(x)
        x = self.dropout(x)
        return x+res
    
class ElasticWaveSolver(torch.nn.Module):
    def __init__(self, in_channels=5, out_channels=10, width=256, x_modes=32, y_modes=24, depth=6):
        super().__init__()
        self.padding = torch.nn.ReflectionPad2d((4, 4, 8, 8))
        self.input_projection = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, width//2, kernel_size = 1),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(width//2),
            torch.nn.Conv2d(width//2, width, kernel_size=1),
            torch.nn.GELU()
        )
        self.input_residual = torch.nn.Conv2d(in_channels, width, kernel_size=1)
        self.fno_blocks = torch.nn.Sequential(*[FNOBlock(width, x_modes, y_modes) for _ in range(depth)])
        self.upsample=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(width, 128, kernel_size=(4, 1),stride=(2, 1),padding=(1, 0)),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=(4, 1),stride=(2, 1),padding=(1, 0)),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(128, 128, kernel_size=(4, 1),stride=(4, 1),padding=(1, 0)),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(128)
        )
        self.output_projection = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, out_channels, kernel_size=1),
            torch.nn.Tanh()
        )   
    def forward(self, x):
        x = self.padding(x)
        x_res = self.input_residual(x)
        x_in = self.input_projection(x)
        x = self.fno_blocks(x_in + x_res)
        x = self.upsample(x)
        x = self.output_projection(x)
        x = self.crop_t(x,target_h=1000, target_w=70)
        return x
    def crop_t(self, x, target_h=1000, target_w=70):
        _, _, h, w = x.shape
        crop_top = (h - target_h) // 2
        crop_left = (w - target_w) // 2
        return x[:, :, crop_top:crop_top + target_h, crop_left:crop_left + target_w]