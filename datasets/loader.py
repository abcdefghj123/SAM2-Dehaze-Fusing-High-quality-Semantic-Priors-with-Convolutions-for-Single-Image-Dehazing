import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img
from utils.common import read_mask_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]
    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs

class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'hazy')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        # read image, and scale [0, 1] to [-1, 1]
        hazy_img_name = self.img_names[idx]
        #print("hazy",hazy_img_name)
        # 训练时候用
        # clear_img_name = hazy_img_name.split('_')[0] + '.png'
        #print("GT",clear_img_name)
        # 测试时候用
        # clear_img_name = hazy_img_name.split('_')[0]
        #OTS、desen、haze4K时用
        #mask_img_name = 'segmented_' + clear_img_name
        #ITS\NH时用
        # mask_real_name = 'segmented_' + hazy_img_name
        # mask_dcp_name = 'segmented_' + hazy_img_name
        # depth_real_name = hazy_img_name
        # depth_dcp_name = hazy_img_name
        # dehazy_dcp_name = hazy_img_name
        # 修改 `clear_img_name`, `mask_real_name`, `mask_dcp_name`, `depth_real_name`, `depth_dcp_name`, 和 `dehazy_dcp_name` 获取相应的图像
        clear_img_name = hazy_img_name.split('_foggy')[0] + '.png'  # 获取与 hazy 图像对应的清晰图像名称
        mask_real_name = 'segmented_' + hazy_img_name # 通过 hazy 图像名称提取 mask_real 名称
        mask_dcp_name = 'segmented_' + hazy_img_name
        depth_real_name = hazy_img_name # 获取与 hazy 图像对应的深度图名称
        depth_dcp_name = hazy_img_name  # 同上
        dehazy_dcp_name = hazy_img_name # 获取去雾图像名称

        hazy_img_name = self.img_names[idx]  # 例如: "1_1_0.9.png"

        # 1. 提取基础 ID (比如 "1")
        # split('_')[0] 得到 "1"
        base_id = hazy_img_name.split('_')[0]

        # 2. 构造各个文件的完整名称
        # 对应原图 (GT): 1.png
        clear_img_name = base_id + '.png'

        # 对应真实掩码: segmented_1.png (假设是基于原图 ID)
        # 如果你的 mask 文件名是 segmented_1_1_0.9.png，则用 hazy_img_name
        mask_real_name = 'segmented_' + base_id + '.png'

        # 对应 DCP 掩码: 这里你定义了两个，通常 DCP 是针对雾图处理的
        mask_dcp_name = 'segmented_' + base_id + '.png'

        # 对应深度图: 1.png (根据你上传的文件名，深度图和原图名一致)
        depth_real_name = base_id + '.png'
        depth_dcp_name = base_id + '.png'

        # 对应去雾图: 1_1_0.9_dehazed.png
        # 注意：去掉原名的 .png 再加后缀
        dehazy_dcp_name = hazy_img_name.replace('.png', '') + '_dehazed.png'

        # print("depth",depth_img_name)
        source_img = read_img(os.path.join(self.root_dir, 'hazy', hazy_img_name)) * 2 - 1
        target_img = read_img(os.path.join(self.root_dir, 'GT', clear_img_name)) * 2 - 1
        mask_real = read_mask_img(os.path.join(self.root_dir, 'mask_real', mask_real_name)) * 2 - 1
        mask_dcp = read_mask_img(os.path.join(self.root_dir, 'mask_dcp', mask_dcp_name)) * 2 - 1
        depth_real= read_mask_img(os.path.join(self.root_dir, 'depth_real', depth_real_name)) * 2 - 1
        depth_dcp = read_mask_img(os.path.join(self.root_dir, 'depth_dcp', depth_dcp_name)) * 2 - 1
        dehazy_dcp = read_img(os.path.join(self.root_dir, 'dehazy_dcp', dehazy_dcp_name)) * 2 - 1

        mask_real = np.expand_dims(mask_real, axis=-1)
        mask_dcp = np.expand_dims(mask_dcp, axis=-1)
        depth_real = np.expand_dims(depth_real, axis=-1)
        depth_dcp = np.expand_dims(depth_dcp, axis=-1)
        # print(mask_img.shape)
        # print(source_img.shape)
        if self.mode == 'train':
            [source_img, target_img,dehazy_dcp,depth_real,depth_dcp, mask_real,mask_dcp] = augment([source_img, target_img,dehazy_dcp,depth_real,depth_dcp, mask_real, mask_dcp], self.size, self.edge_decay,
                                                         self.only_h_flip)
        if self.mode == 'valid':
            [source_img, target_img,dehazy_dcp,depth_real,depth_dcp, mask_real,mask_dcp] = align([source_img, target_img,dehazy_dcp,depth_real,depth_dcp, mask_real, mask_dcp], self.size)
        #
        # names = ["source_img", "target_img", "dehazy_dcp", "depth_real", "depth_dcp", "mask_real", "mask_dcp"]
        # for nm, arr in zip(names, [source_img, target_img, dehazy_dcp, depth_real, depth_dcp, mask_real, mask_dcp]):
        #     print(f"[DEBUG] {nm}: shape={arr.shape}, ndim={arr.ndim}")

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'dehazy_dcp': hwc_to_chw(dehazy_dcp),'depth_real': hwc_to_chw(depth_real),
                'depth_dcp': hwc_to_chw(depth_dcp), 'mask_real': hwc_to_chw(mask_real),'mask_dcp': hwc_to_chw(mask_dcp),'filename': clear_img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        mask_name = 'segmented_' + img_name
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1
        mask_img = read_mask_img(os.path.join(self.root_dir, 'mask', mask_name)) * 2 - 1

        return {'img': hwc_to_chw(img),'mask': hwc_to_chw(mask_img),'filename': img_name}
