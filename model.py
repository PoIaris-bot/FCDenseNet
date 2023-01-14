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
        # TODO: check this
        self.trans_conv = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, output_padding=1,
                                             bias=False)

    def forward(self, x, skip_connection):
        output = self.trans_conv(x)
        return torch.cat([output, skip_connection], dim=1)


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=3, down_path_block_layers=(4, 4, 4, 4, 4), bottleneck_block_layers=4,
                 up_path_block_layers=(4, 4, 4, 4, 4), growth_rate=12, first_conv_out_channels=48):
        super(FCDenseNet, self).__init__()
        self.down_path_block_layers = down_path_block_layers
        self.up_path_block_layers = up_path_block_layers
        skip_connection_channels = []

        self.first_conv = nn.Conv2d(in_channels, first_conv_out_channels, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        cur_channels = first_conv_out_channels

        # Down sampling path
        self.down_dense_blocks = nn.ModuleList([])
        self.transition_downs = nn.ModuleList([])

        for i in range(len(down_path_block_layers)):
            self.down_dense_blocks.append(
                DenseBlock(cur_channels, growth_rate, down_path_block_layers[i], residual=True)
            )
            cur_channels += growth_rate * down_path_block_layers[i]
            skip_connection_channels.insert(0, cur_channels)
            self.transition_downs.append(TransitionDown(cur_channels))

        # Bottleneck
        self.bottleneck = DenseBlock(cur_channels, growth_rate, bottleneck_block_layers)
        prev_out_channels = growth_rate * bottleneck_block_layers

        # Up sampling path
        self.up_dense_blocks = nn.ModuleList([])
        self.transition_ups = nn.ModuleList([])

        for i in range(len(up_path_block_layers)):
            self.transition_ups.append(TransitionUp(prev_out_channels))
            cur_channels = prev_out_channels + skip_connection_channels[i]

            self.up_dense_blocks.append(
                DenseBlock(cur_channels, growth_rate, up_path_block_layers[i])
            )
            prev_out_channels = growth_rate * up_path_block_layers[i]

        # Final layer
        self.final_conv = nn.Conv2d(prev_out_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.first_conv(x)

        skip_connections = []
        for i in range(len(self.down_dense_blocks)):
            output = self.down_dense_blocks[i](output)
            skip_connections.append(output)
            output = self.transition_downs[i](output)

        output = self.bottleneck(output)
        for i in range(len(self.up_dense_blocks)):
            skip_connection = skip_connections.pop()
            output = self.transition_ups[i](output, skip_connection)
            output = self.up_dense_blocks[i](output)

        output = self.final_conv(output)
        output = self.sigmoid(output)
        return output


FCDenseNets = {
    'FCDenseNet56': FCDenseNet(
        in_channels=3, down_path_block_layers=(4, 4, 4, 4, 4), up_path_block_layers=(4, 4, 4, 4, 4),
        bottleneck_block_layers=4, growth_rate=12, first_conv_out_channels=48
    ),
    'FCDenseNet67': FCDenseNet(
        in_channels=3, down_path_block_layers=(5, 5, 5, 5, 5), up_path_block_layers=(5, 5, 5, 5, 5),
        bottleneck_block_layers=5, growth_rate=16, first_conv_out_channels=48
    ),
    'FCDenseNet103': FCDenseNet(
        in_channels=3, down_path_block_layers=(4, 5, 7, 10, 12), up_path_block_layers=(12, 10, 7, 5, 4),
        bottleneck_block_layers=15, growth_rate=16, first_conv_out_channels=48
    )
}
