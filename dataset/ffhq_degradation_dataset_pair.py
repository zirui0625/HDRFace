import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, usm_sharp
from basicsr.utils.registry import DATASET_REGISTRY
from typing import Sequence, Dict, Union, List, Mapping, Any, Optional


@DATASET_REGISTRY.register()
class FFHQPairedDataset(data.Dataset):
    
    def __init__(
            self,
            dataroot_gt: Union[str, List[str]],
            dataroot_lq: Union[str, List[str]],
            io_backend: Mapping[str, Any],
            use_hflip: bool,
            mean: Sequence[float],
            std: Sequence[float],
            out_size: int = 512,
            
            color_jitter_prob: Optional[float] = None,
            color_jitter_pt_prob: Optional[float] = None,
            color_jitter_shift: Optional[int] = 20,
            gray_prob: Optional[float] = None,
            gt_gray: Optional[bool] = True,
            
            crop_components: Optional[bool] = False,
            eye_enlarge_ratio: Optional[int] = 1,
            component_path: Optional[str] = None,
            usm: Optional[Mapping[str, Any]] = None,
            **kwargs
    ) -> "FFHQPairedDataset":
        super(FFHQPairedDataset, self).__init__()

        self.file_client = None
        self.io_backend_opt = io_backend
        self.use_hflip = use_hflip
        
        self.mean = mean
        self.std = std
        self.out_size = out_size
        
        self.usm_opt = usm if usm is not None else {'enable': False}
        if self.usm_opt.get('enable', False):
            print("Using USM for GT images")
        else:
            print("Not using USM")

        self.crop_components = crop_components
        self.eye_enlarge_ratio = eye_enlarge_ratio
        if self.crop_components:
            self.components_list = torch.load(component_path)

        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_pt_prob = color_jitter_pt_prob
        self.color_jitter_shift = color_jitter_shift / 255. if color_jitter_shift else None
        self.gray_prob = gray_prob
        self.gt_gray = gt_gray

        self.gt_paths = []
        self.lq_paths = []
        
        if isinstance(dataroot_gt, str):
            dataroot_gt = [dataroot_gt]
        if isinstance(dataroot_lq, str):
            dataroot_lq = [dataroot_lq]
            
        if len(dataroot_gt) != len(dataroot_lq):
            raise ValueError(
                f"GT and LQ folder counts must match. "
                f"Got {len(dataroot_gt)} GT folders and {len(dataroot_lq)} LQ folders."
            )

        logger = get_root_logger()
        for gt_folder, lq_folder in zip(dataroot_gt, dataroot_lq):
            if not os.path.exists(gt_folder):
                logger.warning(f"GT folder does not exist: {gt_folder}")
                continue
            if not os.path.exists(lq_folder):
                logger.warning(f"LQ folder does not exist: {lq_folder}")
                continue
                
            logger.info(f"Scanning GT folder: {gt_folder}")
            logger.info(f"Scanning LQ folder: {lq_folder}")
            
            gt_paths_temp = []
            for root, _, files in os.walk(gt_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, gt_folder)
                        gt_paths_temp.append((full_path, rel_path))
            
            for gt_path, rel_path in gt_paths_temp:

                lq_path = os.path.join(lq_folder, rel_path)
                
                if os.path.exists(lq_path):
                    self.gt_paths.append(gt_path)
                    self.lq_paths.append(lq_path)
                else:
                    base_name = os.path.splitext(rel_path)[0]
                    found = False
                    for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                        lq_path_alt = os.path.join(lq_folder, base_name + ext)
                        if os.path.exists(lq_path_alt):
                            self.gt_paths.append(gt_path)
                            self.lq_paths.append(lq_path_alt)
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"LQ image not found for GT: {gt_path}")

        if len(self.gt_paths) == 0:
            raise ValueError(
                f"No paired images found in GT folders: {dataroot_gt} "
                f"and LQ folders: {dataroot_lq}"
            )
        
        sorted_indices = sorted(range(len(self.gt_paths)), key=lambda i: self.gt_paths[i])
        self.gt_paths = [self.gt_paths[i] for i in sorted_indices]
        self.lq_paths = [self.lq_paths[i] for i in sorted_indices]
        
        logger.info(f"Total {len(self.gt_paths)} paired images loaded.")
        logger.info(f"Image size: {self.out_size}x{self.out_size}")
        if self.color_jitter_prob is not None:
            logger.info(f"Color jitter probability: {self.color_jitter_prob}")
        if self.gray_prob is not None:
            logger.info(f"Gray probability: {self.gray_prob}")

    @staticmethod
    def color_jitter(img, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    def get_component_coordinates(self, index, status):
        if index >= 70000:
            index -= 70000
        components_bbox = self.components_list[f'{index:08d}']
        
        if status[0]:  # hflip
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            components_bbox['left_eye'][0] = self.out_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.out_size - components_bbox['right_eye'][0]
            components_bbox['mouth'][0] = self.out_size - components_bbox['mouth'][0]

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

        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        
        lq_path = self.lq_paths[index]
        img_bytes = self.file_client.get(lq_path)
        img_lq = imfrombytes(img_bytes, float32=True)
        
        if img_gt is None:
            raise ValueError(f"Failed to load GT image: {gt_path}")
        if img_lq is None:
            raise ValueError(f"Failed to load LQ image: {lq_path}")
        
        if img_gt.shape[:2] != (self.out_size, self.out_size):
            img_gt = cv2.resize(img_gt, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        if img_lq.shape[:2] != (self.out_size, self.out_size):
            img_lq = cv2.resize(img_lq, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        
        if self.usm_opt.get('enable', False):
            usm_args = {}
            if 'weight' in self.usm_opt:
                usm_args['weight'] = self.usm_opt['weight']
            if 'radius' in self.usm_opt:
                usm_args['radius'] = self.usm_opt['radius']
            if 'threshold' in self.usm_opt:
                usm_args['threshold'] = self.usm_opt['threshold']
            img_gt = usm_sharp(img_gt, **usm_args)

        img_gt, status = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=True)
        img_lq = augment(img_lq, hflip=status[0], rotation=False)
        
        if self.crop_components:
            locations = self.get_component_coordinates(index, status)
            loc_left_eye, loc_right_eye, loc_mouth = locations

        if self.color_jitter_prob is not None and np.random.uniform() < self.color_jitter_prob:
            img_lq = self.color_jitter(img_lq, self.color_jitter_shift)
        
        if self.gray_prob is not None and np.random.uniform() < self.gray_prob:
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2GRAY)
            img_lq = np.tile(img_lq[:, :, None], [1, 1, 3])
            if self.gt_gray:
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        img_lq = img_lq.permute(1, 2, 0).cpu().numpy()
        img_gt = img_gt.permute(1, 2, 0).cpu().numpy()

        img_lq = img_lq * 0.5 + 0.5
        img_gt = img_gt * 0.5 + 0.5

        prompts = "high quality portrait photo, detailed face, sharp focus, professional photography"
        
        return_dict = {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': gt_path,
            'lq_path': lq_path,
            'caption': prompts
        }

        if self.crop_components:
            return_dict['loc_left_eye'] = loc_left_eye
            return_dict['loc_right_eye'] = loc_right_eye
            return_dict['loc_mouth'] = loc_mouth

        return return_dict

    def __len__(self):
        return len(self.gt_paths)


if __name__ == '__main__':
    from torchvision.utils import make_grid
    from omegaconf import OmegaConf
    from basicsr.utils import tensor2img, imwrite

    def prepare_images_for_saving(image_tensor):
        grid_image = make_grid(image_tensor.cpu(), nrow=int(np.sqrt(image_tensor.size(0))), padding=2)
        image = grid_image.permute(1, 2, 0).mul(255).byte().numpy()
        return image

    test_config = {
        'dataroot_gt': ['path/to/gt/folder'], 
        'dataroot_lq': ['path/to/lq/folder'],  
        'io_backend': {'type': 'disk'},
        'use_hflip': True,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'out_size': 512,
        'crop_components': False,
        'usm': {'enable': False}
    }

    dataset = FFHQPairedDataset(**test_config)
    
    print(f"Dataset size: {len(dataset)}")
    
    os.makedirs('exp_data/paired_test', exist_ok=True)
    
    for i in range(min(10, len(dataset))):
        data_dict = dataset[i]
        gt = data_dict["gt"]
        lq = data_dict["lq"]
        gt_path = data_dict["gt_path"]
        lq_path = data_dict["lq_path"]
        
        print(f"Sample {i}:")
        print(f"  GT: {gt_path}")
        print(f"  LQ: {lq_path}")
        print(f"  GT shape: {gt.shape}, range: [{np.min(gt):.3f}, {np.max(gt):.3f}]")
        print(f"  LQ shape: {lq.shape}, range: [{np.min(lq):.3f}, {np.max(lq):.3f}]")
        
        gt_tensor = torch.tensor(gt).permute(2, 0, 1)
        lq_tensor = torch.tensor(lq).permute(2, 0, 1)
        
        name = os.path.basename(gt_path).split('.')[0]
        
        gt_img = tensor2img(gt_tensor)
        lq_img = tensor2img(lq_tensor)
        
        imwrite(gt_img, f'exp_data/paired_test/{name}_gt.png')
        imwrite(lq_img, f'exp_data/paired_test/{name}_lq.png')
    
    print("Test completed!")
