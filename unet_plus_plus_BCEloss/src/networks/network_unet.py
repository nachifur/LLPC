import numpy as np
import torchvision.models as models
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .loss import RCFLoss, AWBCELoss


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class UNET(BaseNetwork):

    def __init__(self, config, in_channels=7, init_weights=True):
        super(UNET, self).__init__()
        self.edge_detect = UNetSOURCE(
            1, in_channels=in_channels, depth=5, merge_mode='concat')
        # loss (BCELoss, RCFLoss, AWBCELoss)
        if config.LOSS == "BCELoss":
            self.add_module('loss', nn.BCELoss())
        elif config.LOSS == "RCFLoss":
            self.add_module('loss', RCFLoss())
        elif config.LOSS == "AWBCELoss":
            self.add_module('loss', AWBCELoss())

        if init_weights:
            self.init_weights()

    def process(self, images, gradient, mask, edge_truth):
        inputs = torch.cat((images, gradient, mask), dim=1)
        edges = self(inputs)
        loss, logs = self.cal_loss(edges, edge_truth)
        return edges[-1], loss, logs

    def forward(self, x):
        edges = self.edge_detect(x)
        return [edges]

    def cal_loss(self, edges, edge_truth):
        loss = 0
        logs = []
        i = 0
        for edge in edges:
            matching_loss = self.loss(edge, edge_truth)
            loss += matching_loss
            logs.append(("l_"+str(i), matching_loss.item()))
            i += 1
        return [loss], logs


class UNetSOURCE(nn.Module):
    # https://github.com/RobotLocomotion/unet-pytorch
    """ 
    model = UNet(3, depth=5, merge_mode='concat')
    `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', use_spectral_norm=False, Sigmoid=True):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNetSOURCE, self).__init__()

        """edit"""
        if Sigmoid:
            self.Sigmoid = True
        else:
            self.Sigmoid = False
        """edit"""

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling,
                                 use_spectral_norm=use_spectral_norm)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode, use_spectral_norm=use_spectral_norm)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(
            outs, self.num_classes, use_spectral_norm=use_spectral_norm)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        if self.Sigmoid:
            x = torch.sigmoid(x)
        return x


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, use_spectral_norm=True):
    return spectral_norm(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups), use_spectral_norm)


def upconv2x2(in_channels, out_channels, mode='transpose', use_spectral_norm=True):
    if mode == 'transpose':
        return spectral_norm(nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2), use_spectral_norm)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            spectral_norm(conv1x1(in_channels, out_channels), use_spectral_norm))


def conv1x1(in_channels, out_channels, groups=1, use_spectral_norm=True):
    return spectral_norm(nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1), use_spectral_norm)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, use_spectral_norm=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels,
                             use_spectral_norm=use_spectral_norm)
        self.conv2 = conv3x3(self.out_channels, self.out_channels,
                             use_spectral_norm=use_spectral_norm)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', use_spectral_norm=True):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode, use_spectral_norm=use_spectral_norm)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels, use_spectral_norm=use_spectral_norm)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(
                self.out_channels, self.out_channels, use_spectral_norm=use_spectral_norm)
        self.conv2 = conv3x3(self.out_channels, self.out_channels,
                             use_spectral_norm=use_spectral_norm)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
