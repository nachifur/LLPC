import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
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

    def get_net(self, in_channels, ResnetBlockNum, Sigmoid=True, use_spectral_norm=False):
        encoder_param = [
            [in_channels, 64, 7, 1, 0],
            [64, 128, 4, 2, 1],
            [128, 256, 4, 2, 1]]
        encoder = nn.Sequential(
            *get_encoder(encoder_param, use_spectral_norm))
        middle_param = [ResnetBlockNum, 256]
        middle = nn.Sequential(
            *get_middle(middle_param, use_spectral_norm))
        decoder_param = [
            [256, 128, 4, 2, 1],
            [128, 64, 4, 2, 1],
            [64, 1, 7, 1, 0]]
        decoder = nn.Sequential(
            *get_decoder(decoder_param, use_spectral_norm, Sigmoid=Sigmoid))
        return nn.Sequential(*[encoder, middle, decoder])


class ENDE(BaseNetwork):

    def __init__(self, config, in_channels=7, init_weights=True):
        super(ENDE, self).__init__()
        self.config = config
        edge_detect = self.get_net(
            in_channels, config.MIDDLE_RES_NUM)
        self.edge_detect = edge_detect
        # loss (BCELoss, RCFLoss, AWBCELoss)
        if config.LOSS == "BCELoss":
            self.add_module('loss', nn.BCELoss())
        elif config.LOSS == "RCFLoss":
            self.add_module('loss', RCFLoss())
        elif config.LOSS == "AWBCELoss":
            self.add_module('loss', AWBCELoss())

        if init_weights:
            self.init_weights()

    def forward(self, inputs_img_grad):
        x = []
        x.append(self.edge_detect(inputs_img_grad))
        return x

    def process(self, images, gradient, mask, edge_truth):
        inputs = torch.cat((images, gradient, mask), dim=1)
        edges = self(inputs)
        loss, logs = self.cal_loss(edges, edge_truth)
        return edges[-1], loss, logs

    def cal_loss(self, edges, edge_truth):
        loss = 0
        i = 0
        for edge in edges:
            loss += self.loss(edge, edge_truth)
            # loss += torch.sum(gradient*(1-edge))/3
            i += 1
        mean_loss = loss/len(edges)
        logs = []
        logs.append(("mean_loss", mean_loss.item()))
        return [loss], logs


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_spectral_norm):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(2),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,
                                    padding=0, dilation=2, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3,
                                    padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def get_encoder(encoder_param, use_spectral_norm):
    encoder = []
    index = 0
    for param in encoder_param:
        if index == 0:
            encoder.append(nn.Sequential(
                nn.ReflectionPad2d(3),
                spectral_norm(nn.Conv2d(in_channels=param[0], out_channels=param[1],
                                        kernel_size=param[2], stride=param[3], padding=param[4]), use_spectral_norm),
                nn.InstanceNorm2d(param[1], track_running_stats=False),
                nn.ReLU(True),
            ))
        else:
            encoder.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=param[0], out_channels=param[1],
                                        kernel_size=param[2], stride=param[3], padding=param[4]), use_spectral_norm),
                nn.InstanceNorm2d(param[1], track_running_stats=False),
                nn.ReLU(True),
            ))
        index += 1
    return encoder


def get_middle(middle_param, use_spectral_norm):
    blocks = []
    for _ in range(middle_param[0]):
        block = ResnetBlock(
            middle_param[1], use_spectral_norm)
        blocks.append(block)
    return blocks


def get_decoder(decoder_param, use_spectral_norm, Sigmoid=True):
    decoder = []
    index = 0
    for param in decoder_param:
        if index == len(decoder_param)-1:
            if Sigmoid:
                decoder.append(nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels=param[0], out_channels=param[1],
                              kernel_size=param[2], stride=param[3], padding=param[4]),
                    nn.Sigmoid(),
                ))
            else:
                decoder.append(nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels=param[0], out_channels=param[1],
                              kernel_size=param[2], stride=param[3], padding=param[4]),
                ))
        else:
            decoder.append(nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(
                    in_channels=param[0], out_channels=param[1], kernel_size=param[2], stride=param[3], padding=param[4]), use_spectral_norm),
                nn.InstanceNorm2d(param[1], track_running_stats=False),
                nn.ReLU(True),
            ))
        index += 1
    return decoder


def get_features_merge(features_merge_param, use_spectral_norm=False, F='none', kernel_size=1):
    """features_merge_param = [
        [[256, 4, 2], [256, 2, 1]],
        [[128, 3, 2], [128, 2, 1]],
        [[64, 3, 2], [64, 2, 1]]]
    features_merge = nn.Sequential(
        *get_features_merge(features_merge_param))"""
    blocks = []
    for param in features_merge_param:
        block = []
        index = 0
        for channel in param:
            if index == len(param)-1:
                if F == 'sigmoid':
                    block.append(
                        nn.Sequential(
                            spectral_norm(nn.Conv2d(in_channels=channel[0]*channel[1], out_channels=channel[0]*channel[2],
                                                    kernel_size=kernel_size, stride=1), use_spectral_norm),
                            nn.Sigmoid(),
                        )
                    )
                elif F == 'relu':
                    block.append(
                        nn.Sequential(
                            spectral_norm(nn.Conv2d(in_channels=channel[0]*channel[1], out_channels=channel[0]*channel[2],
                                                    kernel_size=kernel_size, stride=1), use_spectral_norm),
                            nn.InstanceNorm2d(
                                channel[0]*channel[2], track_running_stats=False),
                            nn.ReLU(True),
                        )
                    )
                else:
                    block.append(
                        nn.Sequential(
                            spectral_norm(nn.Conv2d(in_channels=channel[0]*channel[1], out_channels=channel[0]*channel[2],
                                                    kernel_size=kernel_size, stride=1), use_spectral_norm),
                        )
                    )
            else:
                block.append(
                    nn.Sequential(
                        spectral_norm(nn.Conv2d(in_channels=channel[0]*channel[1], out_channels=channel[0]*channel[2],
                                                kernel_size=kernel_size, stride=1), use_spectral_norm),
                        nn.InstanceNorm2d(
                            channel[0]*channel[2], track_running_stats=False),
                        nn.ReLU(True),
                    )
                )
            index += 1

        blocks.append(nn.Sequential(*block))
    return blocks
