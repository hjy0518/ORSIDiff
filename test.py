from utils import init_env

import argparse

import torch

from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from utils.trainer import Trainer
import os
from model.SimpleDiffSef import CondGaussianDiffusion
from model.Net import net
from model.train_val_forward import modification_train_val_forward
set_random_seed(42)

from data_cod import get_loader, test_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print('USE GPU 1')
def getloader(cfg):

    # image_root = './Trian/EORSSD/Images/'
    # gt_root = './Trian/EORSSD/gt/'
    # test_image_root = './Test/EORSSD/Images/'
    # test_gt_root = './Test/EORSSD/gt//'

    # image_root = './Trian/ORSSD/Images/'
    # gt_root = './Trian/ORSSD/gt/'
    # test_image_root = './Test/ORSSD/Images/'
    # test_gt_root = './Test/ORSSD/gt//'

    image_root = './Trian/ORS-4199/Images/'
    gt_root = './Trian/ORS-4199/gt/'
    test_image_root = './Test/ORS-4199/Images/'
    test_gt_root = './Test/ORS-4199/gt//'


    train_loader = get_loader(image_root, gt_root, batchsize=cfg.batch_size, trainsize=cfg.trainsize)
    test_loader = test_dataset(test_image_root, test_gt_root, testsize=cfg.trainsize)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default='./cpts/MyNet_epoch_best.pth')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./cpts/', help='None for saving in wandb folder.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--train_dataset', type=str, default='Train')
    parser.add_argument('--test_dataset', type=str, default='CAMO')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr_min', type=float, default=1e-6)

    cfg = parser.parse_args()
    model = net()
    diffusion_model = CondGaussianDiffusion(model=model,image_size=384,num_sample_steps=10)

    train_loader, test_loader = getloader(cfg)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)

    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=modification_train_val_forward,
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,  # debug
        cfg=cfg,
    )
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)
    # trainer.train()
    trainer.test1(diffusion_model,test_loader,save_to='./result/ORS-41991/')