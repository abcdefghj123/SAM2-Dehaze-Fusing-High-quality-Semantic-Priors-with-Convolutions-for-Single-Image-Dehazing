import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import cv2
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
import skimage
from skimage.color import deltaE_ciede2000 as compare_ciede
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
# from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='dense', type=str, help='dataset name')
parser.add_argument('--exp', default='densehaze', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def calc_ciede2000(im1, im2):
    """
    计算两张图片的CIEDE2000色差
    :param im1: 第一个输入图像 (torch.Tensor)
    :param im2: 第二个输入图像 (torch.Tensor)
    :return: 计算出的CIEDE2000色差值
    """
    im1 = im1[0].view(im1.shape[2], im1.shape[3], 3).detach().cpu().numpy()
    im2 = im2[0].view(im2.shape[2], im2.shape[3], 3).detach().cpu().numpy()

    # 转换为float32
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # 将RGB图像转换为Lab色彩空间
    in_lab = cv2.cvtColor(im1, cv2.COLOR_RGB2Lab)
    gt_lab = cv2.cvtColor(im2, cv2.COLOR_RGB2Lab)

    # 计算CIEDE2000色差
    ciede2000_value = np.average(compare_ciede(gt_lab, in_lab))
    return ciede2000_value

def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    ciede2000_value = AverageMeter()  # 确保 ciede2000_value 是 AverageMeter 实例

    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()
        mask_img = batch['mask'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input, mask_img).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        # 计算 CIEDE2000 色差
        ciede2000_val = calc_ciede2000(output, target)
        ciede2000_value.update(ciede2000_val)  # 更新 CIEDE2000 色差

        # 打印当前样本的 PSNR, SSIM 和 CIEDE2000 色差
        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})\t'
              'CIEDE2000: {ciede2000.val:.04f} ({ciede2000.avg:.04f})'.format(
                  idx, psnr=PSNR, ssim=SSIM, ciede2000=ciede2000_value))

        # 保存结果到CSV文件
        f_result.write('%s,%.02f,%.03f,%.04f\n' % (filename, psnr_val, ssim_val, ciede2000_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    # 测试结束后，打印平均 CIEDE2000 色差
    print('\nAverage CIEDE2000: {:.04f}'.format(ciede2000_value.avg))

    f_result.close()
    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)))



if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()
	saved_model_dir = os.path.join(args.save_dir, args.exp, args.model+'.pth')

	if os.path.exists(saved_model_dir):
		print('==> Start testing, current model name: ' + args.model)
		network.load_state_dict(single(saved_model_dir))
	else:
		print('==> No existing trained model!')
		exit(0)

	# macs, params = get_model_complexity_info(network, (3, 224, 224), as_strings=True,
	# 										 print_per_layer_stat=True, verbose=True)
	# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	test_dataset = PairLoader(dataset_dir, 'test', 'test')
	test_loader = DataLoader(test_dataset,
							 batch_size=1,
							 num_workers=args.num_workers,
							 pin_memory=True)

	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	test(test_loader, network, result_dir)
