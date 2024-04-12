"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """Create a 3x3 convolution layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Default is 1.
        dilation (int, optional): Spacing between kernel elements. Default is 1.

    Returns:
        nn.Conv2d: 3x3 Conv2d layer.
    """

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """Create a 1x1 convolution layer.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Default is 1.

    Returns:
        nn.Conv2d: 1x1 Conv2d layer.
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic Block module for ResNet architectures.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.
        __constants__ (list): List of attributes that are considered constant.
        inplanes (int): Number of input channels.
        stride (int): Stride of the convolution.
        downsample (nn.Module): Optional downsample layer.
        groups (int): Number of blocked connections from input channels to output channels.
        base_width (int): Base width for layers.
        dilation (int): Spacing between kernel elements.
        norm_layer (nn.Module): Type of normalization layer.
    """

    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        """Initialize a BasicBlock.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride of the convolution. Default is 1.
            downsample (nn.Module, optional): Optional downsample layer.
            groups (int, optional): Number of blocked connections from input channels to output channels. Only supports 1.
            base_width (int, optional): Base width for layers. Only supports 64.
            dilation (int, optional): Spacing between kernel elements. Only supports 1.
            norm_layer (nn.Module, optional): Type of normalization layer.

        Raises:
            ValueError: If groups != 1 or base_width != 64.
            NotImplementedError: If dilation > 1.
        """

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        """

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for deep neural networks, often used in architectures like ResNet.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.
        __constants__ (list): List of attributes that are considered constant.
        inplanes (int): Number of input channels.
        stride (int): Stride of the convolution.
        downsample (nn.Module): Optional downsample layer.
        groups (int): Number of blocked connections from input channels to output channels.
        base_width (int): Base width for layers.
        dilation (int): Spacing between kernel elements.
        norm_layer (nn.Module): Type of normalization layer.
    """

    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        """Initialize a Bottleneck block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride of the convolution. Default is 1.
            downsample (nn.Module, optional): Optional downsample layer.
            groups (int, optional): Number of blocked connections from input channels to output channels.
            base_width (int, optional): Base width for the middle (conv2) layer.
            dilation (int, optional): Spacing between kernel elements.
            norm_layer (nn.Module, optional): Type of normalization layer.
        """

        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass through the block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the block.
        """

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model for image classification.

    Attributes:
        block (nn.Module): Type of block to use, usually BasicBlock or Bottleneck.
        layers (list): List of numbers specifying number of blocks in each layer.
        num_classes (int): Number of output classes.
        groups (int): Number of groups for grouped convolutions.
        width_per_group (int): Width for each group.
        replace_stride_with_dilation (list or None): Whether to replace stride with dilation.
        imagenet (bool): Whether to initialize for ImageNet dataset.
        _norm_layer (nn.Module): Type of normalization layer.
        inplanes (int): Number of input channels.
        dilation (int): Spacing between kernel elements.
        base_width (int): Base width for layers.
    """

    def __init__(
        self,
        block,
        layers,
        num_classes,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        imagenet=False,
        *args,
        **kwargs
    ):
        """Initialize the ResNet model.

        Args:
            block (nn.Module): Type of block to use, usually BasicBlock or Bottleneck.
            layers (list): List of numbers specifying number of blocks in each layer.
            num_classes (int): Number of output classes.
            groups (int, optional): Number of groups for grouped convolutions.
            width_per_group (int, optional): Width for each group.
            replace_stride_with_dilation (list or None, optional): Whether to replace stride with dilation.
            imagenet (bool, optional): Whether to initialize for ImageNet dataset.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        super(ResNet, self).__init__(*args, **kwargs)
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # self.normalization = NormalizeByChannelMeanStd(
        #     mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        # )

        if not imagenet:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """Construct a layer with multiple residual blocks.

        Args:
            block (nn.Module): Type of block to use for this layer.
            planes (int): Number of output channels.
            blocks (int): Number of blocks in this layer.
            stride (int, optional): Stride for the first block.
            dilate (bool, optional): Whether to use dilation instead of stride.

        Returns:
            nn.Module: A layer consisting of multiple residual blocks.
        """

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """Internal implementation of forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """

        # x = self.normalization(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after classification.
        """

        return self._forward_impl(x)


# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34"""

#     # BasicBlock and BottleNeck block
#     # have different output size
#     # we use class attribute expansion
#     # to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         # residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 out_channels,
#                 out_channels * BasicBlock.expansion,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#         )

#         # shortcut
#         self.shortcut = nn.Sequential()

#         # the shortcut output dimension is not the same with residual function
#         # use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels * BasicBlock.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers"""

#     expansion = 4

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 out_channels,
#                 out_channels,
#                 stride=stride,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 out_channels,
#                 out_channels * BottleNeck.expansion,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels * BottleNeck.expansion,
#                     stride=stride,
#                     kernel_size=1,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# class ResNet(nn.Module):

#     def __init__(self, block, num_block, num_classes=100):
#         super().__init__()

#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#         )
#         # we use a different inputsize than the original paper
#         # so conv2_x's stride is 1
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block

#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer

#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc(output)

#         return output


def resnet18(num_classes=100):
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=100):
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


# def resnet50(num_classes=100):
#     """return a ResNet 50 object"""
#     return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


# def resnet101(num_classes=100):
#     """return a ResNet 101 object"""
#     return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


# def resnet152(num_classes=100):
#     """return a ResNet 152 object"""
#     return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)
