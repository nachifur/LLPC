import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import EdgeModel
from .utils import Progbar, create_dir, stitch_images, imsave, imshow, save_config
from .metrics import EdgeEvaluation
import torchvision.transforms.functional as F
from tensorboardX import SummaryWriter
from shutil import copyfile
import yaml


class EdgeDetect():
    # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
    def __init__(self, config):
        # config
        self.config = config
        if config.DEBUG == 1:
            self.debug = True
        else:
            self.debug = False
        self.model_name = config.MODEL_NAME
        self.RESULTS_SAMPLE = self.config.RESULTS_SAMPLE
        # model
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        # eval
        self.edgeeva = EdgeEvaluation(threshold=0.5).to(config.DEVICE)
        # dataset
        if config.MODE == 2:  # test
            self.test_dataset = Dataset(
                config, config.TEST_FLIST, config.TEST_EDGE_FLIST, augment=False)
        elif config.MODE == 3:  # eval
            self.val_dataset = Dataset(
                config, config.VAL_FLIST, config.VAL_EDGE_FLIST, augment=False)
        elif config.MODE == 4:  # eval
            self.val_dataset = Dataset(
                config, config.TEST_FLIST, config.TEST_EDGE_FLIST, augment=False)
        else:
            if config.MODE == 0:
                self.train_dataset = Dataset(
                    config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, augment=True)
                self.val_dataset = Dataset(
                    config, config.VAL_FLIST, config.VAL_EDGE_FLIST, augment=True)
            elif config.MODE == 1:
                self.train_dataset = Dataset(
                    config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, augment=False)
                self.val_dataset = Dataset(
                    config, config.VAL_FLIST, config.VAL_EDGE_FLIST, augment=False)
            self.sample_iterator = self.val_dataset.create_iterator(
                config.SAMPLE_SIZE)
        # path
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.backups_path = os.path.join(config.PATH, 'backups')
        self.results_samples_path = os.path.join(self.results_path, 'samples')
        if self.config.BACKUP:
            create_dir(self.backups_path)
        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)
        # load file
        self.log_file = os.path.join(
            config.PATH, 'log_' + self.model_name + '.dat')

        data_save_path = os.path.join(self.config.PATH, 'log_eval_val_ap.txt')
        if os.path.exists(data_save_path):
            self.eval_val_ap = np.genfromtxt(
                data_save_path, dtype=np.str, encoding='utf-8').astype(np.float)
        else:
            self.eval_val_ap = np.zeros((3, 2))

        data_save_path = os.path.join(
            self.config.PATH, 'log_eval_val_ap_id.txt')
        if os.path.exists(data_save_path):
            self.eval_val_ap_id = np.genfromtxt(
                data_save_path, dtype=np.str, encoding='utf-8').astype(np.float)[0]
        else:
            self.eval_val_ap_id = 0.0

        data_save_path = os.path.join(
            self.config.PATH, 'final_model_epoch.txt')
        if os.path.exists(data_save_path):
            self.epoch = np.genfromtxt(
                data_save_path, dtype=np.str, encoding='utf-8').astype(np.float).astype(np.int)
            i = 0
            for epoch in self.epoch:
                if epoch > 0 and config.MODE == 0:
                    self.epoch[i] = epoch-1
                i += 1
        else:
            self.epoch = np.array([0, 0])

        if config.MODE == 1:
            self.eval_val_ap = np.zeros((10, 2))
            self.eval_val_ap_id = 0.0
            self.epoch[1] = 0

        # lr scheduler(not contain discriminator optimizer)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.edge_model.optimizer, 'max', factor=0.5, patience=0, min_lr=1e-5)

    def load(self):
        self.edge_model.load()

    def save(self, Max_end=False):
        self.edge_model.save(Max_end)

    def train(self):
        # initial
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        keep_training = True
        mode = self.config.MODE
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        self.TRAIN_DATA_NUM = total
        if total == 0:
            print(
                'No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        # tensorboardX
        writer = SummaryWriter('runs/edge_detect')
        # train
        while(keep_training):
            # epoch
            if self.config.MODE == 0:
                epoch = self.epoch[0]
            elif self.config.MODE == 1:
                epoch = self.epoch[1]
            epoch += 1
            if self.config.MODE == 0:
                self.epoch[0] = epoch
            elif self.config.MODE == 1:
                self.epoch[1] = epoch
            print('\n\nTraining epoch: %d' % epoch)
            # progbar
            progbar = Progbar(
                total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                # initial
                self.edge_model.train()

                # get data
                images, gradient, edge_truth, mask = self.cuda(
                    *items)
                # imshow(F.to_pil_image((outputs)[0,:,:,:].cpu()))
                # train
                outputs, loss, logs = self.edge_model.process(
                    images, gradient, mask, edge_truth)
                # metrics
                precision, recall = self.edgeeva.eval_accuracy(
                    (1-edge_truth)*mask, (1-outputs)*mask)
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                # backward
                self.edge_model.backward(loss)
                iteration = self.edge_model.iteration

                # tensorboardX
                if (iteration*self.config.BATCH_SIZE) % (self.config.SAVE_INTERVAL//10) == 0:
                    writer.add_scalar(
                        'precision', precision, global_step=iteration*self.config.BATCH_SIZE)
                    writer.add_scalar(
                        'recall', recall, global_step=iteration*self.config.BATCH_SIZE)
                    writer.add_scalar(
                        'loss', loss[0], global_step=iteration*self.config.BATCH_SIZE)
                # log-epoch, iteration
                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs
                # progbar
                progbar.add(len(images), values=logs if self.config.VERBOSE else [
                            x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)
                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    pre_exit, ap = self.eval()
                    self.scheduler.step(ap)

                    with open(self.config.CONFIG_PATH, 'r') as f_obj:
                        config = yaml.load(f_obj, Loader=yaml.FullLoader)
                    config['LR'] = self.scheduler.optimizer.param_groups[0]['lr']
                    save_config(config, self.config.CONFIG_PATH)
                else:
                    pre_exit = False

                # debug
                if self.config.MODE == 0:
                    if self.debug:
                        if iteration >= 40:
                            pre_exit = True
                            copyfile('checkpoints/cell/EdgeDetect.pth',
                                     'checkpoints/cell/EdgeDetect_pre.pth')

                # end condition
                if iteration >= max_iteration or pre_exit:
                    if not pre_exit:
                        self.save(True)
                    keep_training = False
                    writer.close()
                    break

        print('\nEnd training....')

    def eval(self):
        # Jiawei Liu <liujiawei18@mails.ucas.ac.cn>
        # torch.cuda.empty_cache()
        
        # if self.config.MODE == 3 or self.config.MODE == 4:
        #     BATCH_SIZE = self.config.BATCH_SIZE*20
        #     num_workers = 10
        # else:
        #     BATCH_SIZE = self.config.BATCH_SIZE
        #     num_workers = 4

        BATCH_SIZE = self.config.BATCH_SIZE
        num_workers = 4

        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False
        )
        total = len(self.val_dataset)

        self.edge_model.eval()
        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        log_eval_PR = [[0], [0]]
        n_thresh = 99
        # zero all counts
        cntR = torch.Tensor(0, n_thresh).cuda()
        sumR = torch.Tensor(0, n_thresh).cuda()
        cntP = torch.Tensor(0, n_thresh).cuda()
        sumP = torch.Tensor(0, n_thresh).cuda()
        # eval each image
        with torch.no_grad():
            for items in val_loader:
                iteration += 1
                images, gradient, edge_truth, mask = self.cuda(
                    *items)
                # eval
                edges = self.edge_model(images, gradient, mask)
                outputs = 1-edges[-1]
                edge_truth = 1-edge_truth
                thresh, cntR_, sumR_, cntP_, sumP_ = self.edgeeva.eval_bd(
                    outputs*mask, edge_truth*mask, BATCH_SIZE=BATCH_SIZE, n_thresh=99, MODE=self.config.MODE)
                cntR = torch.cat((cntR, cntR_), dim=0)
                sumR = torch.cat((sumR, sumR_), dim=0)
                cntP = torch.cat((cntP, cntP_), dim=0)
                sumP = torch.cat((sumP, sumP_), dim=0)
                # metrics-P,R
                precision, recall = self.edgeeva.eval_accuracy(
                    edge_truth*mask, outputs*mask)
                log_eval_PR[0].append(precision.item())
                log_eval_PR[1].append(recall.item())
                # log
                logs = []
                logs.append(('precision', precision.item()))
                logs.append(('recall', recall.item()))
                logs = [("it", iteration), ] + logs
                progbar.add(len(images), values=logs)

                if self.debug:
                    if iteration == 10:
                        break
        # collect, metrics-ODS,OIS,AP
        eval_bdry, eval_bdry_img, eval_bdry_thr = self.edgeeva.collect_eval_bd(
            thresh, cntR, sumR, cntP, sumP)
        # [bestT, bestR, bestP, bestF, R_max_sum, P_max_sum, F_max_sum, AP], [scores], [thresh, R, P, F]
        # PR curve
        self.edgeeva.PR_curve([eval_bdry_thr[:, 2]], [eval_bdry_thr[:, 1]], [
                              eval_bdry[-1]], self.config.PATH)
        # log
        log_eval_PR = np.array(log_eval_PR)
        log_eval_PR[0][0] = np.mean(log_eval_PR[0][1:])
        log_eval_PR[1][0] = np.mean(log_eval_PR[1][1:])
        data_save_path = os.path.join(self.config.PATH, 'log_eval_PR.txt')
        np.savetxt(data_save_path, log_eval_PR, fmt='%s')
        data_save_path = os.path.join(self.config.PATH, 'eval_bdry.txt')
        np.savetxt(data_save_path, eval_bdry.T, fmt='%s')
        data_save_path = os.path.join(self.config.PATH, 'eval_bdry_img.txt')
        np.savetxt(data_save_path, eval_bdry_img, fmt='%s')
        data_save_path = os.path.join(self.config.PATH, 'eval_bdry_thr.txt')
        np.savetxt(data_save_path, eval_bdry_thr, fmt='%s')
        self.edge_model.edge_detect.best_threshold = eval_bdry[0]
        # print
        print('\naverage precision mean:    {:.3f}'.format(log_eval_PR[0][0]))
        print('average recall mean:    {:.3f}'.format(log_eval_PR[1][0]))
        print('ODS:    F({:.3f},{:.3f}) = {:.3f}    [th={:.3f}]'.format(
            eval_bdry[1], eval_bdry[2], eval_bdry[3], eval_bdry[0]))
        print('OIS:    F({:.3f},{:.3f}) = {:.3f}'.format(
            eval_bdry[4], eval_bdry[5], eval_bdry[6]))
        print('AP:    AP = {:.3f}'.format(eval_bdry[7]))
        # avoid overfitting (pre_train and train)
        if self.config.MODE == 0 or self.config.MODE == 1:
            if eval_bdry[7] > min(self.eval_val_ap[:, 0]):
                pre_exit = False
            else:
                pre_exit = True

            if pre_exit:
                idmax = np.array(self.eval_val_ap[:, 0]).argmax()
                if os.path.exists(os.path.join(self.config.PATH, str(self.eval_val_ap[idmax, 1]) + '.pth')):
                    copyfile(os.path.join(self.config.PATH, str(
                        self.eval_val_ap[idmax, 1]) + '.pth'), os.path.join(self.config.PATH, self.model_name + '.pth'))

                data_save_path = os.path.join(
                    self.config.PATH, 'final_model_epoch.txt')
                if self.config.MODE == 0:
                    self.epoch[0] = self.epoch[0]-idmax-1
                    print('final model epoch:'+str(self.epoch[0]))
                elif self.config.MODE == 1:
                    self.epoch[1] = self.eval_val_ap[idmax, 1]
                    print('final model id:'+str(self.epoch[1]))
                np.savetxt(data_save_path, self.epoch, fmt='%s')

                if self.config.MODE == 0:
                    if os.path.exists(os.path.join(self.config.PATH, str(self.eval_val_ap[idmax, 1]) + '.pth')):
                        copyfile(os.path.join(self.config.PATH, str(
                            self.eval_val_ap[idmax, 1]) + '.pth'), os.path.join(self.config.PATH, self.model_name + '_pre.pth'))
            else:
                self.eval_val_ap = np.delete(self.eval_val_ap, -1, axis=0)
                self.eval_val_ap = np.append(
                    [[eval_bdry[7], self.eval_val_ap_id]], self.eval_val_ap, axis=0)

                self.edge_model.weights_path = os.path.join(
                    self.config.PATH, str(self.eval_val_ap_id) + '.pth')
                self.save()
                self.edge_model.weights_path = os.path.join(
                    self.config.PATH, self.model_name + '.pth')

                data_save_path = os.path.join(
                    self.config.PATH, 'final_model_epoch.txt')
                np.savetxt(data_save_path, self.epoch, fmt='%s')
                if self.eval_val_ap_id == (len(self.eval_val_ap)-1):
                    self.eval_val_ap_id = 0.0
                else:
                    self.eval_val_ap_id += 1

                data_save_path = os.path.join(
                    self.config.PATH, 'log_eval_val_ap.txt')
                np.savetxt(data_save_path, self.eval_val_ap, fmt='%s')
                data_save_path = os.path.join(
                    self.config.PATH, 'log_eval_val_ap_id.txt')
                np.savetxt(data_save_path, [self.eval_val_ap_id, 0], fmt='%s')

            return pre_exit, eval_bdry[7]

    def test(self):
        # initial
        self.edge_model.eval()
        if self.RESULTS_SAMPLE:
            save_path = os.path.join(
                self.results_samples_path, self.model_name)
            create_dir(save_path)
        else:
            save_path = os.path.join(self.results_path, self.model_name)
            create_dir(save_path)
        if self.debug:
            debug_path = os.path.join(save_path, "debug")
            create_dir(debug_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        # test
        index = 0
        with torch.no_grad():
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                index += 1
                images, gradient, edge_truth, mask = self.cuda(
                    *items)
                if self.RESULTS_SAMPLE:
                    image_per_row = 2
                    if self.config.SAMPLE_SIZE <= 6:
                        image_per_row = 1
                    edges = self.edge_model(
                        images, gradient, mask)
                    i = 0
                    for edge in edges:
                        edge = 1 - edge
                        edges[i] = self.postprocess(edge)
                        i += 1
                    images = stitch_images(
                        self.postprocess(images),
                        self.postprocess(gradient),
                        self.postprocess(mask),
                        edges,
                        self.postprocess(1-edge_truth),
                        img_per_row=image_per_row
                    )
                    path = os.path.join(save_path, name)
                    images.save(path)
                else:
                    edges = self.edge_model(images, gradient, mask)
                    outputs = self.postprocess(edges[-1])[0]
                    # outputs = 255-outputs
                    path = os.path.join(save_path, name)
                    imsave(outputs, path)
                # debug
                if self.debug:
                    gradient = self.postprocess(gradient)[0]
                    fname, fext = name.split('.')
                    imsave(gradient, os.path.join(
                        debug_path, fname + '_gradient.' + fext))
                    if index == 10:
                        break
        print('\nEnd test....')

    def sample(self, it=None):
        # initial, do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return
        self.edge_model.eval()
        iteration = self.edge_model.iteration

        items = next(self.sample_iterator)
        images, gradient, edge_truth, mask = self.cuda(
            *items)
        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        edges = self.edge_model(
            images, gradient, mask)
        i = 0
        for edge in edges:
            edge = 1 - edge
            edges[i] = self.postprocess(edge)
            i += 1
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(gradient),
            self.postprocess(mask),
            edges,
            self.postprocess(1-edge_truth),
            img_per_row=image_per_row
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
