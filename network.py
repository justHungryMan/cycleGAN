import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.InstanceNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, block, layer_count):
        super(Generator, self).__init__()

        # 256 * 256
        self.in_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv256_128 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv128_64 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.residual = self.make_layer(block, layer_count, 256, 256)
            
        self.conv64_128 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv128_256 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1),
            nn.Tanh(),
        )

    def make_layer(self, block, layer_count, in_channels, out_channels):
        layers = []
        for i in range(layer_count):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.in_layer(x)
        out = self.conv256_128(out)
        out = self.conv128_64(out)
        out = self.residual(out)
        out = self.conv64_128(out)
        out = self.conv128_256(out)
        out = self.out_layer(out)

        return out
            
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv256_128 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv128_64 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv64_32 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv32_16 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_layer = nn.Conv2d(512, 1, kernel_size=4, padding=1)
    
    def forward(self, x):
        out = self.conv256_128(x)
        out = self.conv128_64(out)
        out = self.conv64_32(out)
        out = self.conv32_16(out)
        out = self.out_layer(out)
        
        # Patch GAN
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
        return out
