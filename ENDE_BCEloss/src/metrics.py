import torch
import torch.nn as nn
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as F
from .utils import imshow
from scipy import ndimage
import math
import multiprocessing
from skimage import morphology


class EdgeEvaluation(nn.Module):
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    def __init__(self, threshold=0.5):
        super(EdgeEvaluation, self).__init__()
        self.threshold = threshold
        nThreads = multiprocessing.cpu_count()
        # self.edge_utils = EdgeUtils(nThreads)

    def eval_accuracy(self, inputs, outputs, mask=None):
        """Measures the accuracy of the edge map"""
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)
        if mask is not None:
            mask = (mask > self.threshold)
            delete_num = torch.sum((1-mask).float())
        else:
            delete_num = 0
        # imshow(F.to_pil_image(labels[1,:,:,:].cpu()))
        relevant = torch.sum(labels.float()) - delete_num
        selected = torch.sum(outputs.float()) - delete_num

        if relevant == 0 and selected == 0:
            return torch.tensor(1), torch.tensor(1)

        true_positive = torch.sum(
            ((outputs == labels) * labels).float()) - delete_num
        recall = true_positive / (relevant + 1e-8)
        precision = true_positive / (selected + 1e-8)

        return precision, recall

    def eval_bd(self, imgs, img_gt, BATCH_SIZE=4, n_thresh=99, threshold=0.5, MODE=0):
        """count for  calculate precision/recall curve"""
        if img_gt.shape[0] != BATCH_SIZE:
            BATCH_SIZE = img_gt.shape[0]
        cntR = torch.Tensor(BATCH_SIZE, 0).cuda()
        sumR = torch.Tensor(BATCH_SIZE, 0).cuda()
        cntP = torch.Tensor(BATCH_SIZE, 0).cuda()
        sumP = torch.Tensor(BATCH_SIZE, 0).cuda()
        thresh = torch.arange(1/(n_thresh+1), 1, 1/(n_thresh+1)).cuda()

        # if MODE == 4 or MODE == 3:
        #     imgs = self.edge_utils.edges_nms(imgs)
        labels = (img_gt > threshold)
        sumR_ = labels.sum(dim=(2, 3)).float()
        for i in range(n_thresh):
            img_bw = imgs >= thresh[i]
            # if MODE == 4 or MODE == 3:
            #     img_bw = self.edge_utils.edges_thin(img_bw)
            sumR = torch.cat((sumR, sumR_), dim=1)
            sumP_ = img_bw.sum(dim=(2, 3)).float()
            sumP = torch.cat((sumP, sumP_), dim=1)
            TP = ((img_bw == labels) * labels).float().sum(dim=(2, 3))
            cntR = torch.cat((cntR, TP), dim=1)
            cntP = torch.cat((cntP, TP), dim=1)

        return thresh, cntR, sumR, cntP, sumP

    def collect_eval_bd(self, thresh, cntR, sumR, cntP, sumP):
        # ODS
        bestT, bestR, bestP, bestF, R, P, F = self.get_ODS(
            thresh, cntR, sumR, cntP, sumP)
        # scores
        scores = self.get_scores(thresh, cntR, sumR, cntP, sumP)
        # OIS
        R_max_sum, P_max_sum, F_max_sum = self.get_OIS(cntR, sumR, cntP, sumP)
        # AP
        AP = self.get_AP(R, P)
        eval_bdry = torch.Tensor(
            [bestT, bestR, bestP, bestF, R_max_sum, P_max_sum, F_max_sum, AP]).cpu().numpy()
        eval_bdry_img = scores.cpu().numpy()
        eval_bdry_thr = torch.cat((thresh.unsqueeze(1), R.unsqueeze(
            1), P.unsqueeze(1), F.unsqueeze(1)), dim=1).cpu().numpy()
        # self.edge_utils.close()
        return eval_bdry, eval_bdry_img, eval_bdry_thr

    def get_ODS(self, thresh, cntR, sumR, cntP, sumP):
        """get ODS"""
        cntR_total = cntR.sum(dim=0, keepdim=True)
        sumR_total = sumR.sum(dim=0, keepdim=True)
        cntP_total = cntP.sum(dim=0, keepdim=True)
        sumP_total = sumP.sum(dim=0, keepdim=True)

        R = cntR_total/(sumR_total+(sumR_total == 0.0).float())
        P = cntP_total/(sumP_total+(sumP_total == 0.0).float())
        F = self.fmeasure(R, P)

        bestT, bestR, bestP, bestF = self.maxF(thresh, R, P)

        return bestT[0, 0], bestR[0, 0], bestP[0, 0], bestF[0, 0], R.squeeze(0), P.squeeze(0), F.squeeze(0)

    def get_scores(self, thresh, cntR, sumR, cntP, sumP):
        """get each image scores"""
        scores = torch.arange(
            1, cntR.shape[0]+1, 1).cuda().float().unsqueeze(1)
        R_I = cntR/(sumR+(sumR == 0.0).float())
        P_I = cntP/(sumP+(sumP == 0.0).float())
        bestT_Img, bestR_Img, bestP_Img, bestF_Img = self.maxF(
            thresh, R_I, P_I)

        scores = torch.cat((scores, bestT_Img), dim=1)
        scores = torch.cat((scores, bestR_Img), dim=1)
        scores = torch.cat((scores, bestP_Img), dim=1)
        scores = torch.cat((scores, bestF_Img), dim=1)

        return scores

    def get_OIS(self, cntR, sumR, cntP, sumP):
        """get OIS"""
        R_I = cntR/(sumR+(sumR == 0.0).float())
        P_I = cntP/(sumP+(sumP == 0.0).float())
        _, id_max = (self.fmeasure(R_I, P_I)).max(dim=1)
        cntR_max_sum = cntR.gather(
            1, id_max.unsqueeze(1)).sum(dim=0, keepdim=True)
        sumR_max_sum = sumR.gather(
            1, id_max.unsqueeze(1)).sum(dim=0, keepdim=True)
        cntP_max_sum = cntP.gather(
            1, id_max.unsqueeze(1)).sum(dim=0, keepdim=True)
        sumP_max_sum = sumP.gather(
            1, id_max.unsqueeze(1)).sum(dim=0, keepdim=True)

        R_max_sum = cntR_max_sum/sumR_max_sum
        P_max_sum = cntP_max_sum/sumP_max_sum
        F_max_sum = self.fmeasure(R_max_sum, P_max_sum)

        return R_max_sum[0, 0], P_max_sum[0, 0], F_max_sum[0, 0]

    def get_AP(self, R, P):
        """get Average Precision"""
        Ru, indR = np.unique(R.cpu().numpy(), return_index=True)
        Pu = P.cpu().numpy()[indR]
        Ri = np.arange(min(Ru)//0.01*0.01+0.01, max(Ru)//0.01*0.01+0.01, 0.01)

        if len(Ru) > 1:
            P_int1 = np.interp(Ri, Ru, Pu)
            Area_PR = sum(P_int1)*0.01
        else:
            Area_PR = 0

        return torch.Tensor([Area_PR]).cuda()

    def fmeasure(self, R, P, beta2=1**2):
        """compute f-measure"""
        return ((1+beta2)*P*R)/((beta2*P)+R+(((beta2*P)+R) == 0.0).float())

    def maxF(self, thresh, R, P):
        """interpolate to find best F"""
        d = torch.arange(0, 1.01, 0.01).cuda().unsqueeze(0)
        t = torch.Tensor([]).cuda()
        r = torch.Tensor([]).cuda()
        p = torch.Tensor([]).cuda()
        f = torch.Tensor([]).cuda()

        for i in range(1, len(thresh)):
            t = torch.cat((t, thresh[i]*d + thresh[i-1]*(1-d)), dim=1)
            r_ = R[:, i].unsqueeze(1).mm(d) + R[:, i-1].unsqueeze(1).mm(1-d)
            r = torch.cat((r, r_), dim=1)
            p_ = P[:, i].unsqueeze(1).mm(d) + P[:, i-1].unsqueeze(1).mm(1-d)
            p = torch.cat((p, p_), dim=1)
            f = torch.cat((f, self.fmeasure(r_, p_)), dim=1)

        bestF, id_max = f.max(dim=1, keepdim=True)
        bestT = t.squeeze(0)[id_max]
        bestR = r.gather(1, id_max)
        bestP = p.gather(1, id_max)

        return bestT, bestR, bestP, bestF

    def PR_curve(self, precision, recall, AP, path):
        """PR curve"""
        colors = cycle(['navy', 'turquoise', 'darkorange',
                        'cornflowerblue', 'teal'])
        fig = plt.figure(figsize=(7, 8))
        # iso-f1 curves
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')
        # PR curve
        for i, color in zip(range(len(recall)), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append(
                'Precision-recall for class {} (average area = {:.3f})'.format(i, AP[i]))
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        path = os.path.join(path, 'PR.png')
        plt.savefig(path)
        plt.close(fig)


def edge_nms(edge, r=1, s=5, m=1.01):
    # https://github.com/pdollar/edges/blob/master/private/edgesNmsMex.cpp
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    edge = edge[0, :, :]
    h = edge.shape[0]
    w = edge.shape[1]

    # compute approximate orientation O from edges edge
    G, theta = sobel_filters(edge)

    # # v1:suppress edges where edge is stronger in orthogonal direction
    # edge_o = edge.copy()
    # for x in range(w):
    #     for y in range(h):
    #         e = edge[y, x]
    #         if e != 0:
    #             e *= m
    #             coso = math.cos(theta[y, x])
    #             sino = math.sin(theta[y, x])
    #             for d in range(-r, r+1):
    #                 if d != 0:
    #                     e0 = interp(
    #                         edge_o, h, w, x+d*coso, y+d*sino)
    #                     if e < e0:
    #                         edge[y, x] = 0
    #                         break

    # # v2:acceleration
    edge_o = edge.copy()
    coso = np.cos(theta)
    sino = np.sin(theta)
    xx, yy = np.meshgrid(np.arange(0, w), np.arange(0, h))
    r_list = list(range(-r, r+1))
    r_list.remove(0)
    edge_o_m = edge_o*m
    id_matrix = np.zeros_like(edge_o_m) > np.ones_like(edge_o_m)
    for d in r_list:
        x_news = xx+d*coso
        y_news = yy+d*sino
        edge_ = interp2d(edge_o, h, w, x_news, y_news)
        id_matrix = id_matrix | (edge_o_m < edge_)
    edge[id_matrix] = 0

    # # v1:suppress noisy edge estimates near boundaries
    # if s > w/2:
    #     s = w/2
    # if s > h/2:
    #     s = h/2
    # for x in range(s):
    #     for y in range(h):
    #         edge[y, x] *= x/s
    #         edge[y, w-1-x] *= x/s
    # for x in range(w):
    #     for y in range(s):
    #         edge[y, x] *= y/float(s)
    #         edge[h-1-y, x] *= y/float(s)
    # v2:acceleration
    if s > w/2:
        s = w/2
    if s > h/2:
        s = h/2
    y = list(range(h))
    for x in range(s):
        edge[y, x] *= x/s
        edge[y, w-1-x] *= x/s
    x = list(range(w))
    for y in range(s):
        edge[y, x] *= y/s
        edge[h-1-y, x] *= y/s

    return edge[np.newaxis, :, :]


def sobel_filters(img):
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta


def interp(I, h, w, x, y):
    # return I[x,y] via bilinear interpolation
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    if x < 0:
        x = 0
    else:
        if x > w-1.001:
            x = w-1.001
    if y < 0:
        y = 0
    else:
        if y > h-1.001:
            y = h-1.001
    x0 = int(x)
    y0 = int(y)
    x1 = x0+1
    y1 = y0+1
    dx0 = x-x0
    dy0 = y-y0
    dx1 = 1-dx0
    dy1 = 1-dy0
    return I[y0, x0]*dx1*dy1 + I[y0, x1]*dx0*dy1 + I[y1, x0]*dx1*dy0 + I[y1, x1]*dx0*dy0


def interp2d(I, h, w, x, y):
    # return I[x,y] via bilinear interpolation
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    x[x < 0] = 0
    x[x > (w-1.001)] = w-1.001
    y[y < 0] = 0
    y[y > (h-1.001)] = h-1.001
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0+1
    y1 = y0+1
    dx0 = x-x0
    dy0 = y-y0
    dx1 = 1-dx0
    dy1 = 1-dy0

    output = I[y0.ravel(), x0.ravel()].reshape((h, w))*dx1*dy1 + I[y0.ravel(), x1.ravel()].reshape((h, w))*dx0 * \
        dy1 + I[y1.ravel(), x0.ravel()].reshape((h, w))*dx1*dy0 + \
        I[y1.ravel(), x1.ravel()].reshape((h, w))*dx0*dy0
    return output


def edge_thin(edge):
    edge = edge[0, :, :]
    edge = (morphology.skeletonize(edge)).astype(np.uint8)
    return edge[np.newaxis, :, :]


class EdgeUtils():
    def __init__(self, nThreads=4):
        self.nThreads = nThreads
        self.pool = multiprocessing.Pool(processes=self.nThreads)

    def edges_nms(self, edges, r=1, s=5, m=1.01):
        edges = edges.cpu().numpy()
        edges_ = self.pool.map(edge_nms, edges)
        i = 0
        for edge in edges_:
            edges[i, :, :, :] = edge
            i += 1
        edges = torch.from_numpy(edges).cuda()
        return edges

    def edges_thin(self, edges):
        edges = edges.cpu().numpy()
        edges_ = self.pool.map(edge_thin, edges)
        i = 0
        for edge in edges_:
            edges[i, :, :, :] = edge
            i += 1
        edges = torch.from_numpy(edges).cuda()
        return edges

    def close(self):
        self.pool.close()
        self.pool.join()
