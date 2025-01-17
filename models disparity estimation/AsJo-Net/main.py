
from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test
from utils import *
from torch.utils.data import DataLoader
import gc
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from IPython.display import clear_output
torch.__version__
# def get_torch_device():
#     global directml_enabled
#     global cpu_state
#     if directml_enabled:
#         global directml_device
#         return directml_device
#     if cpu_state == CPUState.MPS:
#         return torch.device("mps")
#     if cpu_state == CPUState.CPU:
#         return torch.device("cpu")
#     else:
#         if is_intel_xpu():
#             return torch.device("xpu")
#         else:
#             return torch.device(torch.cuda.current_device())

# def get_torch_device():
#     return torch.device("cpu")

# device = get_torch_device()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
parser = argparse.ArgumentParser(description='Accurate and Real-Time Stereo Matching via Context and Geometry Interaction (CGI-Stereo)')

parser.add_argument('--model', default='CGI_Stereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')

# parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
# parser.add_argument('--datapath', default="/home/xgw/data/KITTI_2015/", help='data path')
# parser.add_argument('--trainlist', default='./filenames/kitti12_15_all.txt', help='training list')
# parser.add_argument('--testlist',default='./filenames/kitti15_all.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=20, help='testing batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="10,14,16,18:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
# @dataclass
# class Args:
#     model: str           = 'CGI_Stereo'
#     maxdisp: int         = 192
#     dataset: str         = 'sceneflow'
#     datapath: str        = "/Users/Documents/"
#     trainlist: str       = 'sceneflow_train_flying_things.txt'
#     testlist: str        = 'sceneflow_val_flying_things.txt'
#     lr: float            = 0.001
#     batch_size: int      = 4
#     test_batch_size: int = 20
#     epochs: int          = 20
#     lrepochs: str        = "10,14,16,18:2"
#     logdir: str          = ''
#     loadckpt: str        = ''
#     resume: bool         = False
#     seed: int            = 1
#     summary_freq: int    = 20
#     save_freq: int       = 1

# args = Args()
# @dataclass
# class Args:
#     model: str           = 'CGI_Stereo'
#     maxdisp: int         = 192*7
#     dataset: str         = 'scape_pipes'
#     datapath: str        = ""
#     trainlist: str       = "scape_pipes_jpg.txt"
#     testlist: str        = "scape_pipes_jpg.txt"
#     lr: float            = 0.001
#     batch_size: int      = 4
#     test_batch_size: int = 20
#     epochs: int          = 20
#     lrepochs: str        = "10,14,16,18:2"
#     logdir: str          = ''
#     loadckpt: str        = ''
#     resume: bool         = False
#     seed: int            = 1
#     summary_freq: int    = 20
#     save_freq: int       = 1

# args = Args()
from pathlib import Path
@dataclass
class Args:
    model: str           = 'CGI_Stereo'
    maxdisp: int         = 192*7
    dataset: str         = 'scape_pipes'
    datapath: str        = r"C:\Users"
    trainlist: str       = "scape_pipes_train_jpg.txt"
    testlist: str        = "scape_pipes_test_jpg.txt"
    lr: float            = 0.001
    batch_size: int      = 4
    test_batch_size: int = 1
    epochs: int          = 20
    lrepochs: str        = "10,14,16,18:2"
    logdir: str          = r'.\logs'
    loadckpt: str        = r""
    resume: bool         = False
    seed: int            = 1
    summary_freq: int    = 20
    save_freq: int       = 1
args = Args()
print(os.path.abspath(args.loadckpt))
# parse arguments, set seeds
# args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)
# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)
# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=False)
# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
# model.cuda()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# load parameters
start_epoch = 0
if args.resume:

    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(os.path.abspath(args.logdir), all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict) 
    # model.load_state_dict(state_dict['model'])
    model.load_state_dict(model_dict)
print("start at epoch {}".format(start_epoch))

# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']
    # imgL = imgL.cuda()
    # imgR = imgR.cuda()
    # disp_gt = disp_gt.cuda()
    # disp_gt_low = disp_gt_low.cuda()
    disp_gt = torch.abs(disp_gt)
    disp_gt_low = torch.abs(disp_gt_low)
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    # print(f"disp_gt is type {type(disp_gt)}")
    disp_gt = disp_gt.to(device)
    disp_gt_low = disp_gt_low.to(device)
    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = (disp_gt_low < args.maxdisp) & (disp_gt_low > 0)
    masks = [mask, mask_low]
    disp_gts = [disp_gt, disp_gt_low] 
    loss = model_loss_train(disp_ests, disp_gts, masks)
    disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    # image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            # image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests_final]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            # scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests_final]
            # scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests_final]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)
# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    # imgL = imgL.cuda()
    # imgR = imgR.cuda()
    # disp_gt = disp_gt.cuda()
    disp_gt = torch.abs(disp_gt)
    imgL = imgL.to(device)
    imgR = imgR.to(device)
    disp_gt = disp_gt.to(device)
    print(f"disp_gt is type {type(disp_gt)}")
    plt.imshow(disp_gt[0].detach().cpu().numpy())
    plt.show()

    disp_ests = model(imgL, imgR)
    plt.imshow(disp_ests[0][0].detach().cpu().numpy())
    plt.show()
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)
def train():
    bestepoch = 0
    error = 100
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            # do_summary = global_step % args.summary_freq == 0
            do_summary = global_step == 20
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss, time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        #bestepoch = 0
        #error = 100
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                # save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs, batch_idx, len(TestImgLoader), loss, time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["D1"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["D1"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))

if __name__ == '__main__':
    train()