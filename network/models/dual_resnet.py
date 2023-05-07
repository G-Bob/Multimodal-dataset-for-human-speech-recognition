import torch.nn as nn
import torch

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels* BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels* BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))

class BasicBlock3d(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))

class dualresnet(nn.Module):
    def __init__(self,block2d,block3d,num_block,num_classes=15):
        super(dualresnet, self).__init__()
        self.in_channels = 64
        # First input network
        self.input1_net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Second input network
        self.input2_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Third input network
        self.input3_net = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer
        self.conv2_2d = self._make_layer2d(block2d, 64, num_block[0], 1)
        self.conv3_2d = self._make_layer2d(block2d, 128, num_block[1], 2)
        self.conv4_2d= self._make_layer2d(block2d, 256, num_block[2], 2)
        self.conv5_2d= self._make_layer2d(block2d, 512, num_block[3], 2)

        self.conv6_2d = self._make_layer2d(block2d, 64, num_block[0], 1)
        self.conv7_2d = self._make_layer2d(block2d, 128, num_block[1], 2)
        self.conv8_2d = self._make_layer2d(block2d, 256, num_block[2], 2)
        self.conv9_2d = self._make_layer2d(block2d, 512, num_block[3], 2)

        self.conv2_3d = self._make_layer3d(block3d, 64, num_block[0], 1)
        self.conv3_3d = self._make_layer3d(block3d, 128, num_block[1], 2)
        self.conv4_3d = self._make_layer3d(block3d, 256, num_block[2], 2)
        self.conv5_3d = self._make_layer3d(block3d, 512, num_block[3], 2)
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(1024, num_classes)  # Concatenated output size is 4096

    def _make_layer2d(self, block, out_channels, num_blocks, stride):

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _make_layer3d(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, input1, input2):
        # First input forward pass
        out1 = self.input1_net(input1)
        out1 = self.conv2_2d(out1)
        out1 = self.conv3_2d(out1)
        out1 = self.conv4_2d(out1)
        out1 = self.conv5_2d(out1)
        out1 = self.avg_pool2d(out1)
        out1 = out1.view(out1.size(0), -1)

        # Second input forward pass
        out2 = self.input2_net(input2)
        #out2 = self.conv6_3d(out2)
        out2 = self.conv7_2d(out2)
        out2 = self.conv8_2d(out2)
        out2 = self.conv9_2d(out2)
        out2 = self.avg_pool2d(out2)
        out2 = out2.view(out2.size(0), -1)

        # Concatenate outputs of three sub-networks
        out = torch.cat((out1, out2), dim=1)

        # Fully connected layer
        out = self.fc(out)

        return out

def dualresnet18():
    """ return a ResNet 18 object
    """
    return dualresnet(BasicBlock,BasicBlock, [2, 2, 2, 2])