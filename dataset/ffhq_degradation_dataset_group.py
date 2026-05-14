import os
import cv2
import math
import random
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize
from typing import Sequence, Dict, Union, List, Mapping, Any, Optional

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, usm_sharp
from basicsr.utils.registry import DATASET_REGISTRY

def _u8_to_f01(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _f01_to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).round().astype(np.uint8)


def _usm_sharp_cv(img_f01: np.ndarray,
                  weight: float = 0.5,
                  radius: int = 50,
                  threshold: int = 10) -> np.ndarray:
    """USM sharpening — identical to degrade_like_yaml's usm_sharp_cv."""
    img_u8 = _f01_to_u8(img_f01)
    k = int(max(1, radius) * 2 + 1)
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(img_u8, (k, k), 0)
    residual = img_u8.astype(np.int16) - blur.astype(np.int16)
    if threshold is None:
        mask = np.ones_like(img_u8, dtype=np.uint8) * 255
    else:
        mask = (np.abs(residual) > int(threshold)).astype(np.uint8) * 255
    sharpen = np.clip(
        img_u8.astype(np.float32) + float(weight) * residual.astype(np.float32),
        0, 255
    ).astype(np.uint8)
    out = img_u8.copy()
    out[mask > 0] = sharpen[mask > 0]
    return _u8_to_f01(out)


def _gaussian_kernel_iso(ksize: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2 + 1e-8))
    return kernel / kernel.sum()


def _gaussian_kernel_aniso(ksize: int,
                            sigma_x: float,
                            sigma_y: float,
                            theta: float) -> np.ndarray:
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    c, s = math.cos(theta), math.sin(theta)
    x_rot = c * xx + s * yy
    y_rot = -s * xx + c * yy
    kernel = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2 + 1e-8)
                      + y_rot ** 2 / (2 * sigma_y ** 2 + 1e-8)))
    return kernel / kernel.sum()


def _sample_kernel(ksize: int,
                   kernel_list: list,
                   kernel_prob: Optional[list],
                   blur_sigma_minmax: list) -> np.ndarray:
    """Sample a blur kernel — identical to degrade_like_yaml's sample_kernel."""
    if kernel_prob is None or len(kernel_prob) != len(kernel_list):
        choice = random.choice(kernel_list)
    else:
        p = np.array(kernel_prob, dtype=np.float64)
        p /= p.sum() + 1e-12
        choice = np.random.choice(kernel_list, p=p)

    smin, smax = float(blur_sigma_minmax[0]), float(blur_sigma_minmax[1])

    if choice == "aniso":
        sigma_x = random.uniform(smin, smax)
        sigma_y = random.uniform(smin, smax)
        theta   = random.uniform(-math.pi, math.pi)
        return _gaussian_kernel_aniso(ksize, sigma_x, sigma_y, theta)
    else:                                   # "iso" or fallback
        sigma = random.uniform(smin, smax)
        return _gaussian_kernel_iso(ksize, sigma)


def _add_gaussian_noise(img_f01: np.ndarray,
                        noise_range: Optional[list]) -> np.ndarray:
    if noise_range is None:
        return img_f01
    sigma = random.uniform(float(noise_range[0]), float(noise_range[1]))
    noise = np.random.randn(*img_f01.shape).astype(np.float32) * (sigma / 255.0)
    return np.clip(img_f01 + noise, 0.0, 1.0)


def _jpeg_compress(img_f01: np.ndarray,
                   jpeg_range: Optional[list]) -> np.ndarray:
    if jpeg_range is None:
        return img_f01
    quality = int(random.randint(int(jpeg_range[0]), int(jpeg_range[1])))
    img_u8 = _f01_to_u8(img_f01)
    ok, enc = cv2.imencode(".jpg", img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img_f01
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img_f01 if dec is None else _u8_to_f01(dec)


def degrade_online(img_f01: np.ndarray, params: dict) -> np.ndarray:
    
    # 1. Resize to canonical size
    img_f01 = cv2.resize(img_f01, (512, 512), interpolation=cv2.INTER_LINEAR)
    h, w = 512, 512

    # 2. USM sharpening (applied to the *source* before degradation, same as offline)
    usm_cfg = params.get("usm") or {}
    if usm_cfg.get("enable", False):
        img_f01 = _usm_sharp_cv(
            img_f01,
            weight=usm_cfg.get("weight", 0.5),
            radius=usm_cfg.get("radius", 50),
            threshold=usm_cfg.get("threshold", 10),
        )

    # 3. Blur
    kmin, kmax = [int(x) for x in params.get("blur_kernel_size", [3, 10])]
    assert kmin < kmax, f"blur_kernel_size must satisfy min < max, got {[kmin, kmax]}"
    ksize  = random.randint(kmin, kmax) * 2 + 1          # odd kernel size
    kernel = _sample_kernel(
        ksize,
        kernel_list=params.get("kernel_list", ["iso", "aniso"]),
        kernel_prob=params.get("kernel_prob", None),
        blur_sigma_minmax=params.get("blur_sigma", [0.2, 3.0]),
    )
    img_f01 = cv2.filter2D(img_f01, -1, kernel)

    # 4. Downsample
    smin, smax = [float(x) for x in params.get("downsample_range", [1.0, 4.0])]
    scale  = random.uniform(smin, smax)
    new_w  = max(1, int(w // scale))
    new_h  = max(1, int(h // scale))
    img_f01 = cv2.resize(img_f01, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 5. Gaussian noise
    img_f01 = _add_gaussian_noise(img_f01, params.get("noise_range", None))

    # 6. JPEG compression
    img_f01 = _jpeg_compress(img_f01, params.get("jpeg_range", None))

    # 7. Resize back
    img_f01 = cv2.resize(img_f01, (w, h), interpolation=cv2.INTER_LINEAR)

    # 8. Color jitter (LQ only — caller decides whether to apply)
    cj_prob = params.get("color_jitter_prob", None)
    if cj_prob is not None and random.random() < float(cj_prob):
        shift  = float(params.get("color_jitter_shift", 20)) / 255.0
        jitter = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img_f01 = np.clip(img_f01 + jitter, 0.0, 1.0)

    # 9. Grayscale (LQ only — caller decides whether to apply)
    gray_prob = params.get("gray_prob", None)
    if gray_prob is not None and random.random() < float(gray_prob):
        gray    = cv2.cvtColor(_f01_to_u8(img_f01), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        img_f01 = np.tile(gray[:, :, None], (1, 1, 3))

    return np.clip(img_f01, 0.0, 1.0)

@DATASET_REGISTRY.register()
class FFHQDegradationDataset(data.Dataset):
   
    def __init__(
        self,
        dataroot_gt: Union[str, List[str]],
        io_backend: Mapping[str, Any],
        use_hflip: bool,
        mean: Sequence[float],
        std: Sequence[float],
        out_size: int = 512,
        blur_kernel_size: Optional[list] = None,
        kernel_list: Optional[list] = None,
        kernel_prob: Optional[list] = None,
        blur_sigma: Optional[list] = None,
        downsample_range: Optional[list] = None,
        noise_range: Optional[list] = None,
        jpeg_range: Optional[list] = None,
        color_jitter_prob: Optional[float] = None,
        color_jitter_shift: Optional[int] = 20,
        color_jitter_pt_prob: Optional[float] = None,
        gray_prob: Optional[float] = None,
        gt_gray: bool = True,
        usm: Optional[Mapping[str, Any]] = None,
        crop_components: bool = False,
        component_path: Optional[str] = None,
        eye_enlarge_ratio: float = 1.4,
        **kwargs,
    ):
        super().__init__()
        logger = get_root_logger()

        self.file_client    = None
        self.io_backend_opt = io_backend
        self.use_hflip      = use_hflip
        self.mean           = mean
        self.std            = std
        self.out_size       = out_size

        # Pack all degradation knobs into one dict so degrade_online can read them
        self.degrade_params = {
            "blur_kernel_size" : blur_kernel_size  or [3, 10],
            "kernel_list"      : kernel_list       or ["iso", "aniso"],
            "kernel_prob"      : kernel_prob,
            "blur_sigma"       : blur_sigma        or [0.2, 3.0],
            "downsample_range" : downsample_range  or [1.0, 4.0],
            "noise_range"      : noise_range,
            "jpeg_range"       : jpeg_range,
            "color_jitter_prob": color_jitter_prob,
            "color_jitter_shift": color_jitter_shift,
            "gray_prob"        : gray_prob,
            "usm"              : usm if usm is not None else {"enable": False},
        }

        self.gt_gray             = gt_gray
        self.color_jitter_pt_prob = color_jitter_pt_prob   # kept for completeness

        # USM is also applied to GT inside __getitem__ (via basicsr's usm_sharp)
        self.usm_opt = usm if usm is not None else {"enable": False}
        logger.info(f"USM sharpening on GT: {self.usm_opt.get('enable', False)}")

        # Facial components
        self.crop_components   = crop_components
        self.eye_enlarge_ratio = eye_enlarge_ratio
        if crop_components:
            assert component_path and os.path.exists(component_path), \
                f"component_path not found: {component_path}"
            self.components_list = torch.load(component_path)
            logger.info(f"Loaded facial landmarks from {component_path}")

        # Collect GT paths
        if isinstance(dataroot_gt, str):
            dataroot_gt = [dataroot_gt]

        self.gt_paths: List[str] = []
        IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

        for folder in dataroot_gt:
            if not os.path.isdir(folder):
                logger.warning(f"GT folder not found, skipping: {folder}")
                continue
            logger.info(f"Scanning GT folder: {folder}")
            for root, _, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith(IMG_EXTS):
                        self.gt_paths.append(os.path.join(root, fname))

        if not self.gt_paths:
            raise ValueError(f"No images found in GT folders: {dataroot_gt}")

        self.gt_paths.sort()
        logger.info(f"Total GT images: {len(self.gt_paths)}")

    def _get_component_coordinates(self, index: int, hflip: bool) -> List[torch.Tensor]:
        """Return [loc_left_eye, loc_right_eye, loc_mouth] as float tensors."""
        key = f"{(index % 70000):08d}"
        bbox = dict(self.components_list[key])   # shallow copy to avoid mutation

        if hflip:
            bbox["left_eye"], bbox["right_eye"] = bbox["right_eye"], bbox["left_eye"]
            for part in ("left_eye", "right_eye", "mouth"):
                bbox[part] = list(bbox[part])
                bbox[part][0] = self.out_size - bbox[part][0]

        locations = []
        for part in ("left_eye", "right_eye", "mouth"):
            mean_xy  = np.array(bbox[part][:2], dtype=np.float32)
            half_len = float(bbox[part][2])
            if "eye" in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.concatenate([mean_xy - half_len + 1, mean_xy + half_len])
            locations.append(torch.from_numpy(loc).float())
        return locations

    def __getitem__(self, index: int) -> dict:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_path = self.gt_paths[index]

        # ---- Load GT ----
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)   # float32 [0,1], BGR, HWC
        if img_gt is None:
            raise ValueError(f"Failed to load GT image: {gt_path}")

        # Resize GT to out_size
        if img_gt.shape[:2] != (self.out_size, self.out_size):
            img_gt = cv2.resize(
                img_gt, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA
            )

        # ---- Online degradation → LQ ----
        # degrade_online internally resizes to 512 first, applies USM, then degrades.
        # We pass the already-resized GT so the resize inside is a no-op (same size).
        img_lq = degrade_online(img_gt.copy(), self.degrade_params)

        # ---- USM sharpening on GT (basicsr version, same as original dataset) ----
        if self.usm_opt.get("enable", False):
            usm_kwargs = {k: self.usm_opt[k] for k in ("weight", "radius", "threshold")
                         if k in self.usm_opt}
            img_gt = usm_sharp(img_gt, **usm_kwargs)

        # ---- Synchronised horizontal flip ----
        img_gt, status = augment(img_gt, hflip=self.use_hflip, rotation=False, return_status=True)
        img_lq = augment(img_lq, hflip=status[0], rotation=False)

        # ---- Facial component coordinates ----
        if self.crop_components:
            locations = self._get_component_coordinates(index, hflip=status[0])
            loc_left_eye, loc_right_eye, loc_mouth = locations

        # ---- Grayscale (gt_gray mirrors offline script) ----
        # Note: color_jitter and gray_prob are already applied inside degrade_online
        # for LQ. Here we only handle gt_gray.
        if self.degrade_params.get("gray_prob") is not None:
            # Check whether degrade_online converted LQ to gray this call.
            # We cannot know after the fact, so we apply gt_gray probabilistically
            # with the same probability to keep GT/LQ consistent.
            if self.gt_gray and random.random() < float(self.degrade_params["gray_prob"]):
                img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
                img_gt = np.tile(img_gt[:, :, None], (1, 1, 3))

        # ---- BGR → RGB, HWC → CHW, numpy → tensor ----
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # ---- Normalise to [-1, 1] then back to [0, 1] for training ----
        normalize(img_gt, self.mean, self.std, inplace=True)
        normalize(img_lq, self.mean, self.std, inplace=True)

        img_gt = (img_gt.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)
        img_lq = (img_lq.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5)

        prompts = "high quality portrait photo, detailed face, sharp focus, professional photography"

        return_dict = {
            "lq"      : img_lq,
            "gt"      : img_gt,
            "gt_path" : gt_path,
            'caption' : prompts,
        }
        if self.crop_components:
            return_dict["loc_left_eye"]  = loc_left_eye
            return_dict["loc_right_eye"] = loc_right_eye
            return_dict["loc_mouth"]     = loc_mouth

        return return_dict

    def __len__(self) -> int:
        return len(self.gt_paths)


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    from basicsr.utils import tensor2img, imwrite

    test_cfg = dict(
        dataroot_gt=["path/to/gt"],   
        io_backend={"type": "disk"},
        use_hflip=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        out_size=512,
        blur_kernel_size=[40, 41],
        kernel_list=["iso", "aniso"],
        kernel_prob=[0.5, 0.5],
        blur_sigma=[0.1, 10],
        downsample_range=[0.8, 8],
        noise_range=[0, 20],
        jpeg_range=[60, 100],
        usm={"enable": True},
        crop_components=False,
    )

    ds = FFHQDegradationDataset(**test_cfg)
    print(f"Dataset size: {len(ds)}")

    os.makedirs("exp_data/online_degrade_test", exist_ok=True)
    for i in range(min(5, len(ds))):
        sample = ds[i]
        gt, lq = sample["gt"], sample["lq"]
        print(f"[{i}] gt {gt.shape} [{gt.min():.3f}, {gt.max():.3f}]  "
              f"lq {lq.shape} [{lq.min():.3f}, {lq.max():.3f}]")

        name = os.path.splitext(os.path.basename(sample["gt_path"]))[0]
        imwrite(tensor2img(torch.tensor(gt).permute(2, 0, 1)),
                f"exp_data/online_degrade_test/{name}_gt.png")
        imwrite(tensor2img(torch.tensor(lq).permute(2, 0, 1)),
                f"exp_data/online_degrade_test/{name}_lq.png")

    print("Done.")
