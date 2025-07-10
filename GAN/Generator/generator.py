import torch

class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(4, 1),
                      stride=(2, 1),
                      padding=(1, 0)),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=(0, 0)):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=(4, 1),
                               stride=(2, 1),
                               padding=(1, 0),
                               output_padding=output_padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

def crop(src, target):
    _, _, h, w = target.shape
    src_h, src_w = src.shape[2], src.shape[3]
    crop_top = (src_h - h) // 2
    crop_left = (src_w - w) // 2
    return src[:, :, crop_top:crop_top + h, crop_left:crop_left + w]

class Generator(torch.nn.Module):
    def __init__(self, in_channels=10, out_channels=5):
        super().__init__()
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        self.up1 = UpSample(512, 512, output_padding=(0, 0))
        self.up2 = UpSample(768, 256, output_padding=(0, 0))
        self.up3 = UpSample(384, 128, output_padding=(0, 0))
        self.up4 = UpSample(192, 64, output_padding=(0, 0))

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(64 + in_channels, out_channels, kernel_size=1),
            torch.nn.Tanh()
        )

        self.final_down = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels,
                      kernel_size=(923, 1), stride=(1, 1), padding=(0, 0)),
            torch.nn.Tanh()
        )

    def forward(self, u):
        d1 = self.down1(u)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4)
        u1 = torch.cat([u1, crop(d3, u1)], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, crop(d2, u2)], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, crop(d1, u3)], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, crop(u, u4)], dim=1)
        out = self.block(u4)
        
        return self.final_down(out)
    
def get_generator(in_channels=10, out_channels=5):
    generator = Generator(in_channels=in_channels, out_channels=out_channels)
    return generator