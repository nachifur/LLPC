import os

import numpy
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .network_rcf import RCF
from .network_unet import UNET
from .network_simple import ENDE


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.weights_path = os.path.join(config.PATH, name + '.pth')

    def load(self):
        if os.path.exists(self.weights_path):
            print('Loading %s model...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.weights_path)
                print(self.weights_path)
            else:
                data = torch.load(self.weights_path,
                                  map_location=lambda storage, loc: storage)

            self.edge_detect.load_state_dict(data['model'])
            self.iteration = data['iteration']

    def save(self, Max_end=False):
        print('\nsaving %s...\n' % self.weights_path)
        torch.save({
            'iteration': self.iteration,
            'model': self.edge_detect.state_dict()
        }, self.weights_path)

        if self.config.BACKUP:
            INTERVAL_ = 4
            if self.config.SAVE_INTERVAL and self.iteration % (self.config.SAVE_INTERVAL*INTERVAL_) == 0 or Max_end:
                print('\nsaving %s...\n' % self.name+'_backup')
                torch.save({
                    'iteration': self.iteration,
                    'model': self.edge_detect.state_dict()
                }, os.path.join(self.config.PATH, 'backups/' + self.name + '_' + str(self.iteration // (self.config.SAVE_INTERVAL*INTERVAL_)) + '.pth'))


class EdgeModel(BaseModel):
    def __init__(self, config):
        super(EdgeModel, self).__init__(config.MODEL_NAME, config)
        # networks choose
        if config.NETWORK == "RCF":
            edge_detect = RCF(config, in_channels=7)
        elif config.NETWORK == "UNET":
            edge_detect = UNET(config, in_channels=7)
        elif config.NETWORK == "ENDE":
            edge_detect = ENDE(config, in_channels=7)
        # muti GPU
        if len(config.GPU) > 1:
            edge_detect = nn.DataParallel(edge_detect, config.GPU)
        self.add_module('edge_detect', edge_detect)
        # edge_detect
        self.optimizer = optim.Adam(
            params=edge_detect.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, images, gradient, mask, edge_truth, eval_mode=False):
        if not eval_mode:
            self.iteration += 1
        # zero optimizers
        self.optimizer.zero_grad()
        # process outputs
        edge, loss, logs = self.edge_detect.process(
            images, gradient, mask, edge_truth)

        return edge, loss, logs

    def forward(self, images, gradient, mask):
        inputs = torch.cat((images, gradient, mask), dim=1)
        edges = self.edge_detect(inputs)
        return edges

    def backward(self, loss=None):
        if len(loss) == 2:
            gen_loss = loss[0]
            dis_loss = loss[1]
            if dis_loss is not None:
                dis_loss.backward()
                self.dis_optimizer.step()
            if gen_loss is not None:
                gen_loss.backward()
                self.optimizer.step()
        else:
            gen_loss = loss[0]
            if gen_loss is not None:
                gen_loss.backward()
                self.optimizer.step()
