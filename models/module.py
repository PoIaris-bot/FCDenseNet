import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, residual=False):
        super(DenseBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        feature_maps = [x] if self.residual else []
        for layer in self.layers:
            output = layer(x)
            x = torch.cat([x, output], dim=1)
            feature_maps.append(output)

        return torch.cat(feature_maps, dim=1)


class TransitionDown(nn.Module):
    def __init__(self, channels):
        super(TransitionDown, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.layers(x)


class TransitionUp(nn.Module):
    def __init__(self, channels):
        super(TransitionUp, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                             bias=False)

    def forward(self, x, skip_connection):
        output = self.trans_conv(x)
        return torch.cat([output, skip_connection], dim=1)
