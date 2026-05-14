'''
https://github.com/wzhouxiff/RestoreFormerPlusPlus/tree/main
'''

import os
import cv2
import math
import numpy as np
import random
import os.path as osp
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)

'''
If you meet an error like "ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'", 
feel free to follow 'https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985' to solve it.

Open ./stable-diffusion-webui/venv/lib/python3.10/site-packages/basicsr/data/degradations.py and on line 8, simply change:
from torchvision.transforms.functional_tensor import rgb_to_grayscale
to:
from torchvision.transforms.functional import rgb_to_grayscale
Which should at least get you past this step.

If you meet an error like "cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'imencode'",
feel free to follow 'https://github.com/XPixelGroup/BasicSR/issues/578' to solve it.

BasicSR/basicsr/data/degradations.py
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] 
to
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
'''

from basicsr.data import degradations
from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, usm_sharp
from basicsr.utils.registry import DATASET_REGISTRY
from typing import Sequence, Dict, Union, List, Mapping, Any, Optional

@DATASET_REGISTRY.register()
class FFHQDegradationDataset(data.Dataset):

    def __init__(
        self,
        dataroot_gt: List[str],
        io_backend: Mapping[str, Any],
        use_hflip: bool,
        mean: Sequence[float],
        std: Sequence[float],
        out_size: int,

        blur_kernel_size: Sequence[int],
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[int],
        jpeg_range: Sequence[int],

        color_jitter_prob: Optional[float] = None,
        color_jitter_pt_prob: Optional[float] = None,
        color_jitter_shift: Optional[int] = 20,
        gray_prob: Optional[float] = None,
        gt_gray: Optional[bool] = True,

        crop_components: Optional[bool] = False,
        eye_enlarge_ratio: Optional[int] = 1,
        component_path: Optional[str] = None,
        usm: Mapping[str, Any] = None,
        **kwargs
    ) -> "FFHQDegradationDataset":
        super(FFHQDegradationDataset, self).__init__()
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = io_backend
        self.use_hflip = use_hflip

        self.gt_folder = dataroot_gt
        self.mean = mean
        self.std = std
        self.out_size = out_size

        self.usm_opt = usm
        if self.usm_opt['enable']:
            print("Using USM")
        else:
            print("Not using USM")

        self.crop_components = crop_components  # facial components
        self.eye_enlarge_ratio = eye_enlarge_ratio

        if self.crop_components:
            self.components_list = torch.load(component_path)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = []
            for folder in self.gt_folder:
                # if not osp.exists(folder):
                #     raise ValueError(f"'dataroot_gt' is not a valid path: {folder}")
                paths_folder = paths_from_folder(folder)
                self.paths += paths_folder
                print(f"Load {len(paths_folder)} GT images from {folder}")
            print(f"Total {len(self.paths)} images are loaded.")
            

        # degradations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range

        # color jitter
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_pt_prob = color_jitter_pt_prob
        self.color_jitter_shift = color_jitter_shift
        # to gray
        self.gray_prob = gray_prob
        self.gt_gray = gt_gray

        logger = get_root_logger()
        logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, '
                    f'sigma: [{", ".join(map(str, self.blur_sigma))}]')
        logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
        logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
        logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, '
                        f'shift: {self.color_jitter_shift}')
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')

        self.color_jitter_shift /= 255.


    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def get_component_coordinates(self, index, status):
        components_bbox = self.components_list[f'{index:08d}']
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        if self.usm_opt['enable']:
            usm_args = {}
            if 'weight' in self.usm_opt:
                usm_args['weight'] = self.usm_opt['weight']
            if 'radius' in self.usm_opt:
                usm_args['radius'] = self.usm_opt['radius']
            if 'threshold' in self.usm_opt:
                usm_args['threshold'] = self.usm_opt['threshold']
            
            img_gt = usm_sharp(img_gt, **usm_args)

        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=True)
        h, w, _ = img_gt.shape

        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ------------------------ generate lq image ------------------------ #
        # blur
        assert self.blur_kernel_size[0] < self.blur_kernel_size[1], 'Wrong blur kernel size range'
        cur_kernel_size = random.randint(self.blur_kernel_size[0],self.blur_kernel_size[1]) * 2 + 1
        kernel = degradations.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            cur_kernel_size,
            self.blur_sigma,
            self.blur_sigma, [-math.pi, math.pi],
            noise_range=None)
        img_lq = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if self.noise_range is not None:
            img_lq = degradations.random_add_gaussian_noise(img_lq, self.noise_range)
        # jpeg compression
        if self.jpeg_range is not None:
            img_lq = degradations.random_add_jpg_compression(img_lq, self.jpeg_range)

        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_lq = self.color_jitter_pt(img_lq, brightness, contrast, saturation, hue)

        # round and clip
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        # unified with codeformer (512, 512, 3) as 'numpy.ndarray'
        img_lq = img_lq.permute(1, 2, 0).cpu().numpy()
        img_gt = img_gt.permute(1, 2, 0).cpu().numpy()

        img_lq = img_lq * 0.5 + 0.5
        img_gt = img_gt * 0.5 + 0.5

        return_dict = {
                'lq': img_lq,
                'gt': img_gt,
                'gt_path': gt_path
            }
        if self.crop_components:
            return_dict['loc_left_eye'] = loc_left_eye
            return_dict['loc_right_eye'] = loc_right_eye
            return_dict['loc_mouth'] = loc_mouth

        return return_dict #, return_dict['gt_path']
        # return return_dict

    def __len__(self):
        return len(self.paths)

import argparse
from omegaconf import OmegaConf
import pdb
from tqdm import tqdm
from basicsr.utils import img2tensor, imwrite, tensor2img

if __name__ == '__main__':
    # Run Command: python -m dataset.ffhq_degradation_dataset
    from torchvision.utils import make_grid


    def prepare_images_for_saving(image_tensor):
        grid_image = make_grid(image_tensor.cpu(), nrow=int(np.sqrt(image_tensor.size(0))), padding=2)
        image = grid_image.permute(1, 2, 0).mul(255).byte().numpy() 
        return image


    # pdb.set_trace()
    base = './options/1degra.yaml'

    opt = OmegaConf.load(base)
    params = opt['params']
    dataset = FFHQDegradationDataset(**params)
    # Sort paths based on gt_path
    sorted_indices = sorted(range(len(dataset.paths)), key=lambda i: dataset.paths[i])
    for i in sorted_indices[:100]:
        dict = dataset.__getitem__(i)  # (512, 512, 3) as 'numpy.ndarray'
        gt = dict["gt"]
        lq = dict["lq"]
        gt_path = dict["gt_path"]

        gt = torch.tensor(gt).permute(2, 0, 1)
        lq = torch.tensor(lq).permute(2, 0, 1)

        code_gt = prepare_images_for_saving(torch.tensor(gt))
        code_gt_tensor = torch.tensor(code_gt).permute(2, 0, 1)
        name = gt_path.split('/')[-1][:-4]
        gt = tensor2img(gt)
        imwrite(gt, 'exp_data/1degra/' + name + '_gt_usm.png')
        lq = tensor2img(lq)
        imwrite(lq, 'exp_data/1degra/' + name + '_lq_nojitter.png')
        code_gt_tensor = tensor2img(code_gt_tensor / 255.0)
        imwrite(code_gt_tensor, 'exp_data/1degra/' + name + '_gt_code.png')