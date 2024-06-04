
from typing import List
import torch.utils.data as data
from torchvision import transforms
from data.util.dem_transform import DEMNormalize, MaxPooling2DTransform, ToFloat32
from torchvision.transforms import functional as F
from PIL import Image
import os
import torch
import numpy as np
import random
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path, mode='L'):
    return Image.open(path).convert(mode)

def TIF_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

class DEMDataset(data.Dataset):
    def __init__(
        self, 
        data_root: str,
        mask_root: str = None,
        data_len: int = -1,
        data_aug: bool = False,
        image_size: List[int] = [256, 256],
        horizontal_flip: bool = True,
        loader: callable = TIF_loader
    ):
        gt_imgs = make_dataset(data_root)
        mask_imgs = make_dataset(mask_root) if mask_root is not None else None

        if data_len > 0:
            self.gt_imgs = gt_imgs[:int(data_len)]
            self.mask_imgs = mask_imgs[:int(data_len)] if mask_imgs is not None else None
        else:
            self.gt_imgs = gt_imgs
            self.mask_imgs = mask_imgs
        
        self.gt_tfs = transforms.Compose([
            transforms.ToTensor(),
            ToFloat32(),
            transforms.RandomCrop(256),
            DEMNormalize(),
            MaxPooling2DTransform(kernel_size=2, stride=2),
        ])
        
        self.mask_tfs = transforms.Compose([
            transforms.ToTensor(),
            ToFloat32(),
            MaxPooling2DTransform(kernel_size=2, stride=2)
        ])                           

        self.loader = loader
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.data_aug = data_aug
        self.rotate_angles = [0, 90, 180, 270]

    def __getitem__(self, aug_index):
        ret = {}

        index = int(aug_index / len(self.rotate_angles)) if self.data_aug else aug_index
        
        gt_img = self.gt_tfs(self.loader(self.gt_imgs[index]))

        if self.data_aug:
            # rotate
            gt_img = F.rotate(gt_img, self.rotate_angles[aug_index % len(self.rotate_angles)])

            if random.random() > 0.5:
                gt_img = F.hflip(gt_img)

        cond_img = gt_img.clone()
        if self.mask_imgs is None:
            y, x, ch, cw = self.get_crop_bbox(gt_img)
            cond_img[:,y:y+ch, x:x+cw] = -1
            mask = torch.zeros(1, self.image_size[0], self.image_size[1], dtype=torch.float32)
            mask[:,y:y+ch, x:x+cw] = 1
        else:
            mask = self.mask_tfs(pil_loader(self.mask_imgs[index]))
            mask[mask > 0] = 1
            cond_img[mask > 0] = -1

        ret['gt_image'] = gt_img
        ret['cond_image'] = cond_img
        ret['mask'] = mask
        ret['path'] = self.gt_imgs[index].rsplit("/")[-1].rsplit("\\")[-1]

        # from torchvision.utils import save_image
        # gt_path = './debug/gt_{}.png'.format(random.randint(0,1000))
        # cond_path = './debug/cond_{}.png'.format(index)
        # mask_path = './debug/mask_{}.png'.format(random.randint(0,1000))
        # # Image.fromarray(((gt_img + 1) / 2 * 256).squeeze(0).numpy().astype(np.uint32)).save(gt_path)
        # # Image.fromarray(((cond_img + 1) / 2 * 256).squeeze(0).numpy().astype(np.uint32)).save(cond_path)
        # save_image((gt_img + 1) / 2, gt_path)
        # save_image((cond_img + 1) / 2, cond_path)
        # save_image(mask, mask_path)

        return ret

    def __len__(self):
        k = len(self.rotate_angles) if self.data_aug else 1
        return len(self.gt_imgs) * k

    def get_crop_bbox(self, img):
        """
        Return a random bounding box within a 256x256 image with size ranging from 64 to 160 pixels.
        The bounding box does not need to be square.
        """
        h, w = img.shape[1], img.shape[2]

        bbox_width = np.random.randint(32, 81)
        bbox_height = np.random.randint(32, 81)

        # Calculate maximum possible x and y coordinates for the top-left corner
        x_max = w - bbox_width
        y_max = h - bbox_height

        # Randomly select x and y coordinates for the top-left corner
        x = np.random.randint(0, x_max + 1)
        y = np.random.randint(0, y_max + 1)

        return y, x, bbox_height, bbox_width
