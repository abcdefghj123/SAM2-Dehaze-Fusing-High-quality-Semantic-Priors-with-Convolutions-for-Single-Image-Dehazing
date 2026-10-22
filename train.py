import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import AverageMeter
from datasets.loader import PairLoader
from models import *
import time
# from utils.CR import ContrastLoss
from utils.CR_res import ContrastLoss_res
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/city', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='/home/lisen/sam_depth/city/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='', type=str, help='dataset name')
parser.add_argument('--exp', default='city', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,', type=str, help='GPUs used for training')
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


last_train_stats = {'total': 0.0, 'l1': 0.0, 'contrast': 0.0}

# def setup_distributed():
#     dist.init_process_group(backend='nccl', init_method='env://')
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     return local_rank


def format_time(t):
    hrs = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def train(train_loader, network, criterion, optimizer, scaler):
    losses = AverageMeter()
    l1_meter = AverageMeter()
    contrast_meter = AverageMeter()
    total_meter = AverageMeter()

    torch.cuda.empty_cache()

    network.train()

    # def _peek_shapes(batch):
    #     report = {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')}
    #     print("[SHAPES]", report)

    for i, batch in enumerate(train_loader):
        # print("-------------train------------")
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        mask_real = batch['mask_real'].cuda()
        mask_dcp = batch['mask_dcp'].cuda()
        depth_real = batch['depth_real'].cuda()
        depth_dcp = batch['depth_dcp'].cuda()
        dehazy_dcp = batch['dehazy_dcp'].cuda()


        # if i == 0:
        #     _peek_shapes({
        #         'source': source_img, 'dehazy_dcp': dehazy_dcp,
        #         'depth_real': depth_real, 'depth_dcp': depth_dcp,
        #         'mask_real': mask_real, 'mask_dcp': mask_dcp
        #     })
        # print("source",source_img.shape)
        # print("target",target_img.shape)
        # print("mask",mask_img.shape)


        # with autocast(args.no_autocast):
        with autocast(enabled=False):
            output = network(source_img, dehazy_dcp,depth_real, depth_dcp,mask_real, mask_dcp)
            # print("output",output.shape)
            l1_loss = criterion[0](output, target_img)
            contrast_loss = criterion[1](output, target_img, source_img)
            loss = l1_loss + contrast_loss * 0.2  # 与你原本完全一致
            # loss = criterion[0](output, target_img) + criterion[1](output, target_img, source_img) * 0.2
        # ablation-base
        # loss = criterion[0](output, target_img)
        # print("-------------trainend------------")
        losses.update(loss.item())

        l1_meter.update(l1_loss.item())
        contrast_meter.update(contrast_loss.item())
        total_meter.update(loss.item())

        if i % 10 == 0:  # 每10个batch打印一次
            print(f"Train Epoch [{epoch}] Batch [{i}/{len(train_loader)}], Loss: {loss.item():.6f}")

        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        optimizer.zero_grad()

        # 防止梯度爆炸或反向传播 NaN
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # 解尺度以便裁剪梯度
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)  # 裁剪梯度
        scaler.step(optimizer)
        scaler.update()

    global last_train_stats
    last_train_stats = {
        'total': total_meter.avg,
        'l1': l1_meter.avg,
        'contrast': contrast_meter.avg
    }
    return losses.avg


def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        mask_real = batch['mask_real'].cuda()
        mask_dcp = batch['mask_dcp'].cuda()
        depth_real = batch['depth_real'].cuda()
        depth_dcp = batch['depth_dcp'].cuda()
        dehazy_dcp = batch['dehazy_dcp'].cuda()



        with torch.no_grad():  # torch.no_grad() may cause warning
            output = network(source_img, dehazy_dcp,depth_real, depth_dcp,mask_real, mask_dcp).clamp_(-1, 1)

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':
    # local_rank = setup_distributed()
    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    # pretrain weights loader
    checkpoint = torch.load('/home/lisen/dep_sam03/dep_01/saved_models/city/city/MixDehazeNet-s.pth')
    #checkpoint = None
    network = eval(args.model.replace('-', '_'))()
    # network = DDP(network, device_ids=[local_rank], output_device=local_rank)

    network = nn.DataParallel(network).cuda()
    if checkpoint is not None:
        network.load_state_dict(checkpoint['state_dict'])
    criterion = []
    criterion.append(nn.L1Loss())
    criterion.append(ContrastLoss_res(ablation=False).cuda())

    if setting['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        # optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
        optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], eps=1e-6)

    else:
        raise Exception("ERROR: unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],
                                                           eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        best_psnr = checkpoint['best_psnr']
        start_epoch = checkpoint['epoch'] + 1
    else:
        best_psnr = 0
        start_epoch = 0

    best_psnr = 0

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'],
                               setting['edge_decay'],
                               setting['only_h_flip'])
    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # train_loader = DataLoader(train_dataset, batch_size=setting['batch_size'],
    #                           sampler=train_sampler,
    #                           num_workers=args.num_workers, pin_memory=True,
    #                           drop_last=True, persistent_workers=True, prefetch_factor=2)


    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    # if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
    print('==> Start training, current model name: ' + args.model)
    # print(network)

    # writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

    train_ls, test_ls, idx = [], [], []
    start_time = time.time()
    for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):
        loss = train(train_loader, network, criterion, optimizer, scaler)
        # print(f"Epoch [{epoch}/{setting['epochs']}], Train Loss: {loss:.6f}")
        train_ls.append(loss)
        idx.append(epoch)

        # writer.add_scalar('train_loss', loss, epoch)

        scheduler.step()

        if epoch % setting['eval_freq'] == 0:
            avg_psnr = valid(val_loader, network)

            # writer.add_scalar('valid_psnr', avg_psnr, epoch)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                print(avg_psnr)

                torch.save({'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'epoch': epoch,
                            'best_psnr': best_psnr
                            },
                           os.path.join(save_dir, args.model + '.pth'))
                # >>> add: 每一轮结束统一打印
        avg_dict = last_train_stats  # 来自 train() 刚刚更新的全局统计
        avg_total = loss  # 这一轮训练的总损失均值（你的 train() 返回值）
        prefix = "Train"
        elapsed_time = time.time() - start_time  # 当前耗时
        # 避免除零：epoch 从 start_epoch 开始
        finished_epochs = max(1, (epoch - start_epoch + 1))
        avg_epoch_time = elapsed_time / finished_epochs
        total_est_time = avg_epoch_time * (setting['epochs'] - start_epoch + 1)

        cur_time_str = format_time(elapsed_time)
        total_time_str = format_time(total_est_time)

        print(
            f"{prefix} Epoch {epoch}/{setting['epochs']} | "
            f"Time [{cur_time_str} < {total_time_str}] | "
            f"Total {avg_dict['total']:.6f} "
            f"L1 {avg_dict['l1']:.6f} "
            f"Contrast {avg_dict['contrast']:.6f} "
            f"Avg {avg_total:.6f} "
            f"| BestPSNR {best_psnr:.6f} "

        )

        # writer.add_scalar('best_psnr', best_psnr, epoch)

    # if ((epoch + 1) % 10) == 0:
    # plt.plot(idx, train_ls)
    # plt.title('single image dehazy')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig('test2.png', bbox_inches='tight')
    # plt.show()

# else:
#     print('==> Existing trained model')
#     exit(1)
