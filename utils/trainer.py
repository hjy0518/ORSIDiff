import glob
import os
from collections import defaultdict
from pathlib import Path
from validate_metrics import metrics_v1, metrics_dict_to_float
import math
import numpy as np
import torch
from tqdm import tqdm
from utils.logger_utils import create_url_shortcut_of_wandb, create_logger_of_wandb
from utils.train_utils import SmoothedValue, set_random_seed
from utils.import_utils import fill_args_from_dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from model.train_val_forward import simple_train_val_forward
import sys
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def exists(x):
    return x is not None


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cal_mae(gt, res,  save_to=None, n=None):
    res = F.interpolate(res.unsqueeze(0), size=gt.shape, mode='bilinear', align_corners=False)
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = (res > 0.5).float()
    res = res.cpu().numpy().squeeze()
    if save_to is not None:
        plt.imsave(os.path.join(save_to, n), res, cmap='gray')
    return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


def run_on_seed(func):
    def wrapper(*args, **kwargs):
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        set_random_seed(0)
        res = func(*args, **kwargs)
        set_random_seed(seed)
        return res

    return wrapper


class Trainer(object):
    def __init__(
            self,
            model,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader = None,
            train_val_forward_fn=simple_train_val_forward,
            gradient_accumulate_every=1,
            optimizer=None, scheduler=None,
            train_num_epoch=100,
            results_folder='./cpts/',
            amp=False,
            fp16=False,
            split_batches=True,
            log_with='wandb',
            cfg=None,
    ):
        super().__init__()
        self.best_mae = 1e10
        self.model = model
        self.train_val_forward_fn = train_val_forward_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_epoch = train_num_epoch
        # optimizer
        self.opt = optimizer
        self.scheduler = scheduler
        self.results_folder = results_folder
        self.cfg = cfg
        """
            Initialize the data loader.
        """
        self.cur_epoch = 0
        self.total_step = len(train_loader)
        self.best_max_performance = 0
        self.best_epoch = 999
        self.best_dict = dict()
        # prepare model, dataloader, optimizer with accelerator


    def save(self, epoch):
        """
        Delete the old checkpoints to save disk space.
        """

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        torch.save(self.model.state_dict(), self.results_folder + 'MyNet_epoch_{}.pth'.format(epoch))

    def load(self, resume_path: str = None, pretrained_path: str = None):

        data = torch.load(pretrained_path)
        model = self.model
        model.load_state_dict(data)


    def val(self, model, test_data_loader, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = model
        device = model.device
        maes = []
        for i in range(test_data_loader.size):
            image, gt, name, image_for_post = test_data_loader.load_data()
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            out = self.train_val_forward_fn(model, image=image, verbose=False)
            res = out["pred"].detach().cpu()
            maes += [cal_mae(g, r, save_to, n) for g, r, n in zip(gt, res, name)]
        # gather all the results from different processes
        mae = torch.tensor(maes).mean().to(device)
        mae = mae.mean().item()
        # mae = mae_sum / test_data_loader.dataset.size
        _best_mae = min(_best_mae, mae)
        return mae, _best_mae

    def adjust_lr(self,optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=150):
        decay = decay_rate ** (epoch // decay_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = init_lr * decay
            print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr * decay))

    def val_time_ensemble(self, model, test_data_loader):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        def cal_mae(gt,res):
            res = res.data.cpu().numpy().squeeze()
            return np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        model.eval()
        model = model
        device = model.cuda()
        maes = []
        ensemble_maes = []
        for i in range(test_data_loader.size):
            image, gt, name, image_for_post = test_data_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)
            image = image.cuda()
            ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=gt.shape, verbose=False)
            ensem_res = ensem_out["pred"]
            # ensemble_maes += [cal_mae(gt,r) for r in ensem_res]
        # gather all the results from different processes
            ensemble_maes += [cal_mae(gt, r) for r in ensem_res]
        ensemble_maes = torch.tensor(ensemble_maes).mean().cuda().mean().item()
        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae


    def val_batch_ensemble(self, model, test_data_loader, thresholding=False, save_to=None):
        """
        validation function
        """
        global _best_mae
        if '_best_mae' not in globals():
            _best_mae = 1e10

        model.eval()
        model = model
        device = model.device
        ensemble_maes = []
        for data in tqdm(test_data_loader):
            image, gt, name, image_for_post = data['image'], data['gt'], data['name'], data['image_for_post']
            gt = [np.array(x, np.float32) for x in gt]
            gt = [x / x.max() + 1e-8 for x in gt]
            image = image.to(device).squeeze(1)
            batch_res = []
            for i in range(5):
                ensem_out = self.train_val_forward_fn(model, image=image, time_ensemble=True, verbose=False)
                ensem_res = ensem_out["pred"].detach().cpu()
                batch_res.append(ensem_res)
            batch_res = torch.mean(torch.concat(batch_res, dim=1), dim=1, keepdim=True)
            for g, r, n in zip(gt, batch_res, name):
                ensemble_maes.append(cal_mae(g, r, thresholding, save_to, n))

        # gather all the results from different processes
        ensemble_maes = torch.tensor(ensemble_maes).mean().to(device).mean().item()

        _best_mae = min(_best_mae, ensemble_maes)
        return ensemble_maes, _best_mae

    def train(self):
        for epoch in range(self.cur_epoch, self.train_num_epoch):
            # self.adjust_lr(self.opt, self.cfg.lr, epoch)
            self.cur_epoch = epoch
            # Train
            self.model.train()
            loss_sm = SmoothedValue(window_size=10)

            for i, (images, gts) in enumerate(self.train_loader, start=1):
                images = images.cuda()
                gts = gts.cuda()
                self.model = self.model.cuda()
                loss = self.train_val_forward_fn(self.model,gts,images,gts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.opt.step()
                self.opt.zero_grad()
                if i % 100 == 0 or i == self.total_step:
                    loss_sm.update(loss.item())
                    # print(f'Epoch:{epoch}/{self.train_num_epoch} loss: {loss_sm.avg:.4f}({loss_sm.global_avg:.4f})')
                    msg = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],  Loss: {:.4f}'.format(
                        datetime.now(), epoch, self.train_num_epoch, i, self.total_step, loss_sm.avg)
                    # print(f"\r{msg}", end="")
                    print(msg)
            if self.scheduler is not None:
                self.scheduler.step()

            loss_sm_gather = torch.tensor([loss_sm.global_avg])
            loss_sm_avg = loss_sm_gather.mean().item()
            # print(f'Epoch:{epoch}/{self.train_num_epoch} loss_sm_avg: {loss_sm_avg:.4f}')

            # Val
            self.model.eval()
            if (epoch + 1) % 1 == 0 or (epoch >= self.train_num_epoch * 0.7):
                mae, best_mae = self.val_time_ensemble(self.model, self.test_loader)
                print(f'Epoch:{epoch}/{self.train_num_epoch} mae: {mae:.4f}({best_mae:.4f})')
                self.validate(self.model, self.test_loader,self.cur_epoch)

            if best_mae < self.best_mae:
                self.best_mae = best_mae
                self.save("best"+str(epoch))
                self.test(self.model, self.test_loader,'./result/ORS-4199/'+str(epoch))
            self.save(self.cur_epoch)

        print('training complete')


    def test(self, model, test_data_loader, save_to='./result/ORSSD/'):

        def cal_mae(gt,res, save_to=None, name=None):
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print('save img to: ', save_to + name)
            cv2.imwrite(save_to+name,res*255)

        model.eval()
        model = model.cuda()
        for i in range(test_data_loader.size):
            image, gt, name, image_for_post = test_data_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)

            image = image.cuda()

            pre = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=gt.shape, verbose=False)
            pre = pre["pred"][0]
            cal_mae(gt,pre, save_to, name)

    def test1(self, model, test_data_loader, save_to='./result/ORSSD1/'):

        def cal_mae(gt,res, save_to=None, name=None):
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # print('save img to: ', save_to + name)
            cv2.imwrite(save_to+name,res*255)

        model.eval()
        model = model.cuda()
        for i in range(test_data_loader.size):
            image, gt, name, image_for_post = test_data_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)

            image = image.cuda()

            pre = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=gt.shape, verbose=False)
            pre = pre["pred"][0]
            cal_mae(gt,pre, save_to, name)

    def validate(self, model, test_loader, epoch):
        validate_tool = metrics_v1()
        save_path = './cpts/'
        model.eval().cuda()
        with torch.no_grad():
            for i in tqdm(range(test_loader.size), desc="Validating", file=sys.stdout):
                image, gt, name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()

                res = self.train_val_forward_fn(model, image=image, time_ensemble=True,
                                                  gt_sizes=gt.shape, verbose=False)
                res = res["pred"][0]
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()

                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                validate_tool.step((res * 255).astype(np.uint8), (gt * 255).astype(np.uint8))
        curr_metrics_dict = validate_tool.show()
        curr_max_performance, curr_avg_performance = metrics_dict_to_float(curr_metrics_dict)

        if curr_max_performance >= self.best_max_performance:
            self.best_max_performance = curr_max_performance
            self.best_epoch = epoch
            self.best_dict.update(curr_metrics_dict)
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_performance_best.pth'),
                       _use_new_zipfile_serialization=False)

        msg = 'Epoch: {:03d} curr_max_performance: {:.4f} ####  best_max_performance: {:.4f} bestEpoch: {:03d}'.format(
            epoch,
            curr_max_performance,
            self.best_max_performance,
            self.best_epoch)
        print(f"Epoch     {epoch:03d} {curr_metrics_dict} {curr_max_performance:.4f} {curr_avg_performance:.4f}")
        print(f"bestEpoch {self.best_epoch:03d} {self.best_dict} {self.best_max_performance:.4f}")
        # logging.info(msg)
        # logging.info(f"Epoch     {epoch:03d} {curr_metrics_dict} {curr_max_performance:.4f}")
        # logging.info(f"bestEpoch {best_epoch:03d} {best_dict} {best_max_performance:.4f}")

        return curr_metrics_dict, curr_max_performance