import torch
import torch.nn as nn


class RCFLoss(nn.Module):
    # https://github.com/meteorshowers/RCF-pytorch

    def forward(self, prediction, label):
        mask = label.clone()
        num_positive = torch.sum((mask == 1)).float()
        num_negative = torch.sum((mask == 0)).float()
        num_all = num_positive + num_negative

        mask[mask == 1] = 1.0 * num_negative / num_all
        mask[mask == 0] = 1.1 * num_positive / num_all
        loss = torch.nn.functional.binary_cross_entropy(
            prediction, label, weight=mask)
        return loss


class AWBCELoss(nn.Module):
    # Adaptive weighted binary cross entropy
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    def __init__(self):
        super(AWBCELoss, self).__init__()

    def forward(self, prediction, label):

        mask = (label - prediction).abs().detach()+0.01
        loss = torch.nn.functional.binary_cross_entropy(
            prediction, label, weight=mask)
        # mask = (mask-mask.min())/(mask.max()-mask.min())
        # imshow(TF.to_pil_image((mask)[0,:,:,:].cpu()))
        return loss
