import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
import random


class PortraitRestorationDataset(Dataset):
    
    def __init__(
        self,
        root_dir,
        resolution=1024,
        degradation_types=None,
        prompts_list=None,
    ):
        self.root_dir = root_dir
        self.resolution = resolution
        
        self.degraded_dir = os.path.join(root_dir, "degraded")
        self.target_dir = os.path.join(root_dir, "target")
        
        self.image_files = sorted([
            f for f in os.listdir(self.degraded_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
        ])
        
        if prompts_list is None:
            self.prompts = [
                "a high quality portrait photo, detailed face, sharp focus, professional photography",
                "clear face portrait, high resolution, detailed skin texture, natural lighting",
                "restored portrait photo, enhanced details, sharp facial features, vivid colors",
                "professional headshot, crystal clear, perfect skin, studio lighting",
                "high definition portrait, realistic face, detailed eyes, natural expression",
            ]
        else:
            self.prompts = prompts_list
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        degraded_path = os.path.join(self.degraded_dir, img_name)
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        target_path = os.path.join(self.target_dir, img_name)
        target_img = Image.open(target_path).convert('RGB')
        
        degraded_tensor = self.transform(degraded_img)
        target_tensor = self.transform(target_img)
        
        caption = random.choice(self.prompts)
        
        return {
            "degraded": degraded_tensor,
            "target": target_tensor,
            "caption": caption,
            "filename": img_name,
        }


class SyntheticDegradationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        resolution=1024,
        degradation_config=None,
        prompts_list=None,
    ):
        self.root_dir = root_dir
        self.resolution = resolution
        
        self.image_files = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if degradation_config is None:
            self.degradation_config = {
                'blur_kernel_size': [7, 9, 11, 13, 15],
                'blur_sigma': [0.5, 1.0, 1.5, 2.0],
                'noise_level': [0, 5, 10, 15, 20],
                'jpeg_quality': [30, 40, 50, 60, 70, 80],
                'downsample_factor': [2, 3, 4],
            }
        else:
            self.degradation_config = degradation_config
        
        self.base_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
        ])
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        if prompts_list is None:
            self.prompts = [
                "a high quality portrait photo, detailed face, sharp focus, professional photography",
                "clear face portrait, high resolution, detailed skin texture, natural lighting",
                "restored portrait photo, enhanced details, sharp facial features, vivid colors",
            ]
        else:
            self.prompts = prompts_list
    
    def apply_degradation(self, img_pil):
        import cv2
        import numpy as np
        from io import BytesIO
        
        img_np = np.array(img_pil)
        
        if random.random() < 0.7:
            factor = random.choice(self.degradation_config['downsample_factor'])
            h, w = img_np.shape[:2]
            img_np = cv2.resize(img_np, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
            img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if random.random() < 0.8:
            kernel_size = random.choice(self.degradation_config['blur_kernel_size'])
            sigma = random.choice(self.degradation_config['blur_sigma'])
            img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        
        if random.random() < 0.6:
            noise_level = random.choice(self.degradation_config['noise_level'])
            if noise_level > 0:
                noise = np.random.randn(*img_np.shape) * noise_level
                img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.9:
            quality = random.choice(self.degradation_config['jpeg_quality'])
            img_pil_tmp = Image.fromarray(img_np)
            buffer = BytesIO()
            img_pil_tmp.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_pil_tmp = Image.open(buffer)
            img_np = np.array(img_pil_tmp)
        
        return Image.fromarray(img_np)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        target_img = Image.open(img_path).convert('RGB')
        target_img = self.base_transform(target_img)
        
        degraded_img = self.apply_degradation(target_img)
        
        degraded_tensor = self.to_tensor(degraded_img)
        target_tensor = self.to_tensor(target_img)
        
        caption = random.choice(self.prompts)
        
        return {
            "degraded": degraded_tensor,
            "target": target_tensor,
            "caption": caption,
            "filename": os.path.basename(img_path),
        }


def build_dataset(config):
    
    dataset_type = config.get('type', 'PortraitRestoration')
    
    if dataset_type == 'PortraitRestoration':
        return PortraitRestorationDataset(
            root_dir=config['root_dir'],
            resolution=config.get('resolution', 1024),
            prompts_list=config.get('prompts_list', None),
        )
    elif dataset_type == 'SyntheticDegradation':
        return SyntheticDegradationDataset(
            root_dir=config['root_dir'],
            resolution=config.get('resolution', 1024),
            degradation_config=config.get('degradation_config', None),
            prompts_list=config.get('prompts_list', None),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

if __name__ == "__main__":
    config = {
        'type': 'PortraitRestoration',
        'root_dir': '/path/to/your/data',
        'resolution': 1024,
    }
    
    dataset = build_dataset(config)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Degraded shape: {sample['degraded'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Caption: {sample['caption']}")