# https://github.com/jannctu/FINED
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FINED(BaseNetwork):

    def __init__(self, config, in_channels=7, init_weights=True):
        super(FINED, self).__init__()
        self.edge_detect = FINEDSOURCE(in_channels=in_channels)
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
        return edges

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


def batchnorm(in_planes):
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes))


class CRPBlock(nn.Module):

    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes,
                            out_planes, stride=1,
                            bias=False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


stages_suffixes = {0: '_conv',
                   1: '_conv_relu_varout_dimred'}


class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, stride=1,
                                bias=(j == 0)))
        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x += residual
        return x


class MSBlock(nn.Module):
    def __init__(self, c_in, rate=2):
        super(MSBlock, self).__init__()
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        dilation = self.rate * 4 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1,
                               dilation=dilation, padding=dilation)
        self.relu4 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        o4 = self.relu4(self.conv4(o))
        out = o + o1 + o2 + o3 + o4
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class FINEDSOURCE(nn.Module):
    def __init__(self, in_channels=3):
        super(FINEDSOURCE, self).__init__()
        self.isTrain = True
        ############## STAGE 1##################
        self.conv1_1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)

        if self.isTrain:
            self.msblock1_1 = MSBlock(16, 4)
            self.msblock1_2 = MSBlock(16, 4)

            self.RCU1_1 = RCUBlock(32, 32, 2, 2)
            self.RCU1_2 = RCUBlock(32, 32, 2, 2)

            self.conv1_1_down = nn.Conv2d(32, 8, 1, padding=0)
            self.conv1_2_down = nn.Conv2d(32, 8, 1, padding=0)

            self.crp1 = CRPBlock(8, 8, 4)

            self.score_stage1 = nn.Conv2d(8, 1, 1)
            self.st1_BN = nn.BatchNorm2d(
                1, affine=True, eps=1e-5, momentum=0.1)

        ############## STAGE 2 ##################
        self.conv2_1 = nn.Conv2d(16, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        if self.isTrain:
            self.msblock2_1 = MSBlock(64, 4)
            self.msblock2_2 = MSBlock(64, 4)

            self.RCU2_1 = RCUBlock(32, 32, 2, 2)
            self.RCU2_2 = RCUBlock(32, 32, 2, 2)

            self.conv2_1_down = nn.Conv2d(32, 8, 1, padding=0)
            self.conv2_2_down = nn.Conv2d(32, 8, 1, padding=0)

            self.crp2 = CRPBlock(8, 8, 4)

            self.score_stage2 = nn.Conv2d(8, 1, 1)
            self.st2_BN = nn.BatchNorm2d(
                1, affine=True, eps=1e-5, momentum=0.1)

        ############## STAGE 3 ##################
        self.conv3_1 = nn.Conv2d(64, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)

        self.msblock3_1 = MSBlock(256, 4)
        self.msblock3_2 = MSBlock(256, 4)

        self.RCU3_1 = RCUBlock(32, 32, 2, 2)
        self.RCU3_2 = RCUBlock(32, 32, 2, 2)

        # CONV DOWN
        self.conv3_1_down = nn.Conv2d(32, 8, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(32, 8, 1, padding=0)

        self.crp3 = CRPBlock(8, 8, 4)

        # SCORE
        self.score_stage3 = nn.Conv2d(8, 1, 1)
        self.st3_BN = nn.BatchNorm2d(1, affine=True, eps=1e-5, momentum=0.1)

        # POOL
        self.maxpool = nn.MaxPool2d(
            2, stride=2, ceil_mode=True)  # pooling biasa

        # RELU
        self.relu = nn.ReLU()
        if self.isTrain:
            self.score_final = nn.Conv2d(3, 1, 1)
            self.final_BN = nn.BatchNorm2d(
                1, affine=True, eps=1e-5, momentum=0.1)

    def forward(self, x):
        img_H, img_W = x.shape[2], x.shape[3]
        ############## STAGE 1 ##################
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        if self.isTrain:
            er1_1 = self.msblock1_1(conv1_1)
            er1_2 = self.msblock1_2(conv1_2)

            rcu1_1 = self.relu(self.RCU1_1(er1_1))
            rcu1_2 = self.relu(self.RCU1_2(er1_2))

            conv1_1_down = self.conv1_1_down(rcu1_1)
            conv1_2_down = self.conv1_2_down(rcu1_2)

            crp1 = self.crp1(conv1_1_down + conv1_2_down)
            o1_out = self.score_stage1(crp1)
            so1 = crop(o1_out, img_H, img_W)
            so1 = self.st1_BN(so1)
        ############## END STAGE 1 ##################
        pool1 = self.maxpool(conv1_2)
        ############## STAGE 2 ##################
        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        if self.isTrain:
            er2_1 = self.msblock2_1(conv2_1)
            er2_2 = self.msblock2_2(conv2_2)

            rcu2_1 = self.relu(self.RCU2_1(er2_1))
            rcu2_2 = self.relu(self.RCU2_2(er2_2))

            conv2_1_down = self.conv2_1_down(rcu2_1)
            conv2_2_down = self.conv2_2_down(rcu2_2)

            crp2 = self.crp2(conv2_1_down + conv2_2_down)
            o2_out = self.score_stage2(crp2)
            upsample2 = nn.UpsamplingBilinear2d(size=(img_H, img_W))(o2_out)
            so2 = crop(upsample2, img_H, img_W)
            so2 = self.st2_BN(so2)
        ############## END STAGE 2 ##################
        pool2 = self.maxpool(conv2_2)
        ############## STAGE 3 ##################
        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))

        er3_1 = self.msblock3_1(conv3_1)
        er3_2 = self.msblock3_2(conv3_2)

        rcu3_1 = self.relu(self.RCU3_1(er3_1))
        rcu3_2 = self.relu(self.RCU3_2(er3_2))

        conv3_1_down = self.conv3_1_down(rcu3_1)
        conv3_2_down = self.conv3_2_down(rcu3_2)

        crp3 = self.crp3(conv3_1_down + conv3_2_down)
        o3_out = self.score_stage3(crp3)
        upsample3 = nn.UpsamplingBilinear2d(size=(img_H, img_W))(o3_out)
        so3 = crop(upsample3, img_H, img_W)
        so3 = self.st3_BN(so3)
        ############## END STAGE 3 ##################

        ############## FUSION ##################
        if self.isTrain:
            fusecat = torch.cat((so1, so2, so3), dim=1)
            fuse = self.score_final(fusecat)
            fuse = self.final_BN(fuse)
            results = [so1, so2, so3, fuse]
        else:
            results = [so3]
        results = [torch.sigmoid(r) for r in results]

        return results


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]
