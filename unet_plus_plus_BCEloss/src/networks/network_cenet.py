# https://github.com/pindi-krishna/Lung-Segmentation/blob/master/Segmentation.ipynb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import RCFLoss, AWBCELoss
from torchvision import models
from torch.autograd import Variable
import numpy as np

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


class CENet(BaseNetwork):

    def __init__(self, config, in_channels=7, init_weights=True):
        super(CENet, self).__init__()
        self.edge_detect = CENetSOURCE()
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
        x = x[:,0:3,:,:]
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


class DAC(nn.Module):
    
    def __init__(self,channels):
        
        super(DAC, self).__init__()
        self.conv11 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 1, padding = 1)
        
        self.conv21 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 3, padding = 3)
        self.conv22 = nn.Conv2d(channels, channels, kernel_size = 1, dilation = 1, padding = 0)
        
        self.conv31 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 1, padding = 1)
        self.conv32 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 3, padding = 3)
        self.conv33 = nn.Conv2d(channels, channels, kernel_size = 1, dilation = 1, padding = 0)
        
        self.conv41 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 1, padding = 1)
        self.conv42 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 3, padding = 3)
        self.conv43 = nn.Conv2d(channels, channels, kernel_size = 3, dilation = 5, padding = 5)
        self.conv44 = nn.Conv2d(channels, channels, kernel_size = 1, dilation = 1, padding = 0)
        
    def forward(self, x):
        
        c1 = F.relu(self.conv11(x))
        
        c2 = self.conv21(x)
        c2 = F.relu(self.conv22(c2))
        
        c3 = self.conv31(x)
        c3 = self.conv32(c3)
        c3 = F.relu(self.conv33(c3))
        
        c4 = self.conv41(x)
        c4 = self.conv42(c4)
        c4 = self.conv43(c4)
        c4 = F.relu(self.conv44(c4))
        
        c = x + c1 + c2 + c3 + c4 
        
        return c

# Residual Multi Kernel Pooling

class RMP(nn.Module):
    
    def __init__(self,channels):
        super(RMP, self).__init__()

        self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(channels, 1, kernel_size = 1)
        
        self.max2 = nn.MaxPool2d(kernel_size = 3, stride = 3)
        self.conv2 = nn.Conv2d(channels, 1, kernel_size = 1)
        
        self.max3 = nn.MaxPool2d(kernel_size = 5, stride = 5)
        self.conv3 = nn.Conv2d(channels, 1, kernel_size = 1)
       
        self.max4 = nn.MaxPool2d(kernel_size = 6)
        self.conv4 = nn.Conv2d(channels, 1, kernel_size = 1)
        
    def forward(self, x):
        
        m1 = self.max1(x)
        m1 = F.interpolate(self.conv1(m1), size = x.size()[2:], mode = 'bilinear' )
        
        m2 = self.max2(x)
        m2 = F.interpolate(self.conv2(m2), size = x.size()[2:], mode = 'bilinear' )
        
        m3 = self.max3(x)
        m3 = F.interpolate(self.conv3(m3), size = x.size()[2:], mode = 'bilinear' )
        
        m4 = self.max4(x)
        m4 = F.interpolate(self.conv4(m4), size = x.size()[2:], mode = 'bilinear' )
        
        m = torch.cat([m1,m2,m3,m4,x], axis = 1)
        
        return m
        
# Decoder Architecture

class Decoder(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.bn3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x

# Main Architecture   

class CENetSOURCE(nn.Module):
    def __init__(self, num_classes = 1):
        super(CENetSOURCE, self).__init__()

        filters = [64, 128, 256, 512]
        self.resnet = models.resnet34(pretrained=True)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.maxpool1 = self.resnet.maxpool

        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        self.dac = DAC(512)
        self.rmp = RMP(512)

        self.decoder4 = Decoder(516, filters[2])
        self.decoder3 = Decoder(filters[2], filters[1])
        self.decoder2 = Decoder(filters[1], filters[0])
        self.decoder1 = Decoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalconv2 = nn.Conv2d(32, num_classes, 3, padding=1)

        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = Variable(torch.from_numpy(mean)).cuda()
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = Variable(torch.from_numpy(std)).cuda()

    def re_init(self, num_classes = 1):
        filters = [64, 128, 256, 512]
        self.resnet = models.resnet34(pretrained=True).cuda()
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.maxpool1 = self.resnet.maxpool

        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = Variable(torch.from_numpy(mean)).cuda()
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = Variable(torch.from_numpy(std)).cuda()

    def forward(self, x):
        # Encoder
        x = self.conv1((x-self.mean)/self.std)
        x = F.relu(self.bn1(x))
        x = self.maxpool1(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dac(e4)
        e4 = self.rmp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = F.relu(self.finaldeconv1(d1))
        out = self.finalconv2(out)

        return [torch.sigmoid(out)]