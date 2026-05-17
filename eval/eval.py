"""
HDRFace - Image Quality Assessment (IQA) Evaluation Script
"""

import os
import csv
import math
import argparse
import glob

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import pyiqa

from vqfr.archs import init_alignment_model
from vqfr.data import build_dataset
from vqfr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3
from vqfr.utils import img2tensor
from arcface.config.config import Config
from arcface.models.resnet import resnet_face18

def is_number(value: str) -> bool:
    """Return True if value can be parsed as a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_csv_to_dict(filename: str) -> dict:
    data = {}
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        key_field = reader.fieldnames[0]
        for row in reader:
            key = row[key_field]
            data[key] = {
                field: (float(value) if is_number(value) else value)
                for field, value in row.items()
                if field != key_field
            }
    return data

def _get_target_filename(output_filename: str) -> str:
    """Map a predicted filename to its ground-truth counterpart."""
    return output_filename.replace('_LR.png', '_HR.png')

def _png_files(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.endswith('.png'))

class IQA:
    FR_METRICS       = ['psnr', 'ssim', 'psnr_y', 'ssim_y', 'lpips', 'dists', 'topiq_fr']
    NR_METRICS       = ['niqe', 'musiq', 'maniqa', 'maniqa-pipal', 'clipiqa']
    OPTIONAL_NR_METRICS = ['qalign', 'liqe', 'deqa-score', 'qinsight']

    def __init__(
        self,
        hasref: bool = True,
        device: torch.device = None,
        optional_metrics: list[str] | None = None,
    ):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.hasref          = hasref
        self.optional_metrics = [m.lower() for m in (optional_metrics or [])]
        self.metrics          = self._build_metrics(hasref)

    def _build_metrics(self, hasref: bool) -> dict:
        metrics = {}

        if hasref:
            metrics['psnr']       = pyiqa.create_metric('psnr', device=self.device)
            metrics['ssim']       = pyiqa.create_metric('ssim', device=self.device)
            metrics['psnr_y']     = pyiqa.create_metric(
                                        'psnr', test_y_channel=True,
                                        color_space='ycbcr').to(self.device)
            metrics['ssim_y']     = pyiqa.create_metric(
                                        'ssim', test_y_channel=True,
                                        color_space='ycbcr').to(self.device)
            metrics['lpips']      = pyiqa.create_metric('lpips', device=self.device)
            metrics['dists']      = pyiqa.create_metric('dists', device=self.device)
            metrics['topiq_fr']   = pyiqa.create_metric('topiq_fr', device=self.device)

        metrics['niqe']         = pyiqa.create_metric('niqe',         device=self.device)
        metrics['musiq']        = pyiqa.create_metric('musiq',        device=self.device)
        metrics['maniqa']       = pyiqa.create_metric('maniqa',       device=self.device)
        metrics['maniqa-pipal'] = pyiqa.create_metric('maniqa-pipal', device=self.device)
        metrics['clipiqa']      = pyiqa.create_metric('clipiqa',      device=self.device)

        for name in self.OPTIONAL_NR_METRICS:
            if name in self.optional_metrics:
                try:
                    print(f"[IQA] Loading optional metric: {name}")
                    metrics[name] = pyiqa.create_metric(name, device=self.device)
                except Exception as exc:
                    print(f"[IQA] Failed to load '{name}': {exc}")

        return metrics

    @staticmethod
    def _to_nchw_tensor(image) -> torch.Tensor:
        if isinstance(image, (np.ndarray, torch.Tensor)):
            t = torch.as_tensor(image).float()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[-1] in (3, 4):
                t = t[..., :3]
                t = rearrange(t, "b h w c -> b c h w").contiguous()
        else:
            t = F.to_tensor(image).unsqueeze(0)
        return t

    @staticmethod
    def _align_shapes(
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred.shape == target.shape:
            return pred, target
        print(f"[IQA] Shape mismatch — pred: {pred.shape}, target: {target.shape}. "
              "Resizing to the smaller resolution.")
        h = min(pred.shape[2], target.shape[2])
        w = min(pred.shape[3], target.shape[3])
        resize = transforms.Resize((h, w))
        return resize(pred), resize(target)

    def calculate(self, output_image, target_image=None) -> dict | None:
        if target_image is not None:
            assert type(output_image) == type(target_image), (
                "output_image and target_image must be the same type."
            )

        pred = self._to_nchw_tensor(output_image).to(self.device)
        gt   = (self._to_nchw_tensor(target_image).to(self.device)
                if target_image is not None else None)

        if gt is not None:
            pred, gt = self._align_shapes(pred, gt)

        try:
            result = {}

            if gt is not None:
                result['PSNR']      = self.metrics['psnr'](pred, gt).item()
                result['SSIM']      = self.metrics['ssim'](pred, gt).item()
                result['PSNR_Y']    = self.metrics['psnr_y'](pred, gt).item()
                result['SSIM_Y']    = self.metrics['ssim_y'](pred, gt).item()
                result['LPIPS']     = self.metrics['lpips'](pred, gt).item()
                result['DISTS']     = self.metrics['dists'](pred, gt).item()
                result['TOPIQ_FR']  = self.metrics['topiq_fr'](pred, gt).item()

            result['NIQE']         = self.metrics['niqe'](pred).item()
            result['MUSIQ']        = self.metrics['musiq'](pred).item()
            result['MANIQA']       = self.metrics['maniqa'](pred).item()
            result['MANIQA-pipal'] = self.metrics['maniqa-pipal'](pred).item()
            result['CLIP-IQA']     = self.metrics['clipiqa'](pred).item()

            optional_key_map = {
                'qalign':     'Q-Align',
                'liqe':       'LIQE',
                'deqa-score': 'DEQA-score',
                'qinsight':   'Q-Insight',
            }
            for internal_key, result_key in optional_key_map.items():
                if internal_key in self.metrics:
                    result[result_key] = self.metrics[internal_key](pred).item()

        except Exception as exc:
            print(f"[IQA] Error while computing metrics: {exc}")
            return None

        return result

def evaluate_landmark(
    output_folder: str,
    target_folder: str,
    output_files: list[str],
    device: torch.device,
) -> float:
    """Return average landmark L2 distance."""
    print("\n" + "=" * 50)
    print("Evaluating Landmark Distance...")
    print("=" * 50)

    landmark_detector = init_alignment_model().to(device)
    distances = []

    for filename in tqdm(output_files, desc="Landmark", unit="img"):
        pred_path = os.path.join(output_folder, filename)
        gt_path   = os.path.join(target_folder, _get_target_filename(filename))

        img_pred = cv2.imread(pred_path)
        img_gt   = cv2.imread(gt_path)

        pred_lm = landmark_detector.get_landmarks(img_pred)
        gt_lm   = landmark_detector.get_landmarks(img_gt)

        dist = np.sqrt(((gt_lm - pred_lm) ** 2).sum(1)).mean()
        distances.append(dist)

    avg = float(np.mean(distances))
    print(f"Average Landmark L2: {avg:.6f}")
    return avg

def evaluate_arcface(
    output_folder: str,
    target_folder: str,
    output_files: list[str],
    device: torch.device,
    arcface_model_path: str,
) -> tuple[float, int]:
    """Return (average cosine distance in degrees, identical-pair count)."""
    print("\n" + "=" * 50)
    print("Evaluating ArcFace Cosine Distance...")
    print("=" * 50)

    opt   = Config()
    model = resnet_face18(opt.use_se) if opt.backbone == 'resnet18' else None
    assert model is not None, f"Backbone {opt.backbone} not implemented"
    model = DataParallel(model)
    model.load_state_dict(torch.load(arcface_model_path))
    model.to(device).eval()

    def _load(path: str) -> torch.Tensor:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis].astype(np.float32)
        img = (img - 127.5) / 127.5
        return torch.from_numpy(img)

    def _cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    dist_list      = []
    identical_count = 0

    for filename in tqdm(output_files, desc="ArcFace", unit="img"):
        pred_path = os.path.join(output_folder, filename)
        gt_path   = os.path.join(target_folder, _get_target_filename(filename))

        data = torch.stack([_load(gt_path), _load(pred_path)], dim=0).to(device)
        with torch.no_grad():
            feats = model(data).cpu().numpy()

        deg = np.arccos(np.clip(_cosine(feats[0], feats[1]), -1, 1)) / math.pi * 180
        if deg < 1:
            identical_count += 1
        else:
            dist_list.append(deg)

    avg_dist = float(np.mean(dist_list)) if dist_list else 0.0
    print(f"Average ArcFace Distance : {avg_dist:.6f}°")
    print(f"Identical Count (< 1°)   : {identical_count}")
    return avg_dist, identical_count

def evaluate_fid(
    output_folder: str,
    fid_stats_path: str,
    device: torch.device,
    batch_size: int,
    num_sample: int,
    num_workers: int,
    backend: str,
) -> float:
    """Return FID score."""
    print("\n" + "=" * 50)
    print("Evaluating FID Score...")
    print("=" * 50)

    inception = load_patched_inception_v3(device)

    opt = {
        'name': 'SingleImageDataset',
        'type': 'SingleImageDataset',
        'dataroot_lq': output_folder,
        'io_backend': {'type': backend},
        'mean': [0.5, 0.5, 0.5],
        'std':  [0.5, 0.5, 0.5],
    }
    dataset     = build_dataset(opt)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    num_sample  = min(num_sample, len(dataset))
    total_batch = math.ceil(num_sample / batch_size)

    def _gen(loader, n_batch):
        for idx, data in enumerate(loader):
            if idx >= n_batch:
                break
            yield data['lq']

    features = extract_inception_features(
        _gen(data_loader, total_batch), inception, total_batch, device
    ).numpy()[:num_sample]

    sample_mean = np.mean(features, axis=0)
    sample_cov  = np.cov(features, rowvar=False)

    stats     = torch.load(fid_stats_path, weights_only=False)
    fid_score = calculate_fid(sample_mean, sample_cov, stats['mean'], stats['cov'])

    print(f"FID Score: {fid_score:.4f}")
    return fid_score

def evaluate_iqa_partition(
    output_folder: str,
    target_folder: str | None,
    output_files: list[str],
    device: torch.device,
    rank: int,
    optional_metrics: list[str],
) -> dict:

    iqa = IQA(
        hasref=(target_folder is not None),
        device=device,
        optional_metrics=optional_metrics,
    )
    local_results = {}

    for filename in tqdm(output_files, desc=f"GPU {rank}", unit="img"):
        pred_path = os.path.join(output_folder, filename)
        pred_img  = Image.open(pred_path)

        if target_folder is not None:
            gt_path = os.path.join(target_folder, _get_target_filename(filename))
            assert os.path.exists(gt_path), f"Ground-truth file not found: {gt_path}"
            gt_img = Image.open(gt_path)
        else:
            gt_img = None

        values = iqa.calculate(pred_img, gt_img)
        if values is not None:
            local_results[filename] = values

    return local_results

def _worker_entry(rank, gpu_id, output_folder, target_folder,
                  output_files, return_dict, num_gpus, optional_metrics):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] using device: {device}")

    chunk_size = len(output_files) // num_gpus
    start      = rank * chunk_size
    end        = len(output_files) if rank == num_gpus - 1 else start + chunk_size

    return_dict[rank] = evaluate_iqa_partition(
        output_folder, target_folder,
        output_files[start:end],
        device, rank, optional_metrics,
    )

def save_results(
    iqa_results: dict,
    scalar_results: dict,
    txt_path: str,
    csv_path: str,
) -> None:

    all_keys = sorted({k for v in iqa_results.values() for k in v.keys()})
    averages = {
        k: float(np.mean([v.get(k, 0) for v in iqa_results.values()]))
        for k in all_keys
    }
    # Merge scalar results
    averages.update(scalar_results)

    os.makedirs(os.path.dirname(txt_path) or '.', exist_ok=True)
    with open(txt_path, 'w') as f:
        for key, value in averages.items():
            f.write(f"{key:<20}\t{value}\n")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename'] + all_keys)
        for filename, values in iqa_results.items():
            writer.writerow([filename] + [values.get(k, '') for k in all_keys])

    print("\nAverage metrics:")
    for k, v in averages.items():
        print(f"  {k:<20} {v:.4f}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDRFace — Image Quality Assessment evaluation script"
    )
    parser.add_argument("--dataset",       type=str, default=None)
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Directory containing predicted images.")
    parser.add_argument("--target_folder", type=str, default=None,
                        help="Directory containing ground-truth images.")
    parser.add_argument("--resume_csv",    type=str, default=None,
                        help="Path to a previous result CSV to resume from.")
    parser.add_argument("--gpu_ids",       type=str, default="0",
                        help="Comma-separated GPU IDs, e.g. '0,1,2'.")
    parser.add_argument("--eval_qalign",       action="store_true")
    parser.add_argument("--eval_liqe",         action="store_true")
    parser.add_argument("--eval_deqa",         action="store_true")
    parser.add_argument("--eval_qinsight",     action="store_true")
    parser.add_argument("--eval_all_optional", action="store_true",
                        help="Enable all optional NR metrics.")
    parser.add_argument("--eval_landmark", action="store_true",
                        help="Evaluate landmark L2 distance.")
    parser.add_argument("--eval_arcface",       action="store_true",
                        help="Evaluate ArcFace cosine distance.")
    parser.add_argument("--arcface_model_path", type=str,
                        default="pretrained_models/resnet18_110.pth")
    parser.add_argument("--eval_fid",    action="store_true",
                        help="Evaluate FID score.")
    parser.add_argument("--fid_stats",   type=str,
                        default="pretrained_models/inception_FFHQ_512.pth")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--num_sample",  type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--backend",     type=str, default="disk",
                        choices=["disk", "lmdb"])

    return parser.parse_args()

if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    mp.set_start_method('spawn')

    args = parse_args()
    if args.eval_all_optional:
        optional_metrics = IQA.OPTIONAL_NR_METRICS.copy()
    else:
        optional_metrics = []
        if args.eval_qalign:   optional_metrics.append('qalign')
        if args.eval_liqe:     optional_metrics.append('liqe')
        if args.eval_deqa:     optional_metrics.append('deqa-score')
        if args.eval_qinsight: optional_metrics.append('qinsight')

    if optional_metrics:
        print(f"Optional metrics enabled: {optional_metrics}")

    resume_results: dict = {}
    evaluated_files: set = set()
    if args.resume_csv is not None:
        resume_results  = read_csv_to_dict(args.resume_csv)
        evaluated_files = set(resume_results.keys())
        print(f"Resuming: {len(evaluated_files)} files already evaluated.")

    output_files = [f for f in _png_files(args.output_folder) if f not in evaluated_files]

    if args.target_folder is not None:
        target_files = [f for f in _png_files(args.target_folder) if f not in evaluated_files]
        assert len(output_files) == len(target_files), (
            f"File count mismatch: {len(output_files)} predicted vs "
            f"{len(target_files)} ground-truth."
        )

    main_device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scalar_results: dict = {}

    if args.eval_landmark:
        assert args.target_folder, "--target_folder is required for landmark evaluation."
        scalar_results['Landmark-L2'] = evaluate_landmark(
            args.output_folder, args.target_folder, output_files, main_device
        )

    if args.eval_arcface:
        assert args.target_folder, "--target_folder is required for ArcFace evaluation."
        avg_dist, identical = evaluate_arcface(
            args.output_folder, args.target_folder, output_files,
            main_device, args.arcface_model_path,
        )
        scalar_results['ArcFace-dist']      = avg_dist
        scalar_results['ArcFace-identical'] = identical

    if args.eval_fid:
        scalar_results['FID'] = evaluate_fid(
            args.output_folder, args.fid_stats,
            main_device, args.batch_size,
            args.num_sample, args.num_workers, args.backend,
        )

    gpu_ids  = [int(g) for g in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    print(f"\nLaunching {num_gpus} IQA worker(s) on GPU(s): {gpu_ids}")

    manager     = mp.Manager()
    return_dict = manager.dict()

    processes = [
        mp.Process(
            target=_worker_entry,
            args=(rank, gpu_id, args.output_folder, args.target_folder,
                  output_files, return_dict, num_gpus, optional_metrics)
        )
        for rank, gpu_id in enumerate(gpu_ids)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    iqa_results: dict = {}
    for rank_results in return_dict.values():
        iqa_results.update(rank_results)
    iqa_results.update(resume_results)

    if not iqa_results and not scalar_results:
        print("No results to save.")
        exit(0)

    folder_tag = os.path.basename(args.output_folder)
    prefix     = f"IQA_results/{args.dataset}-{folder_tag}"

    save_results(iqa_results, scalar_results, f"{prefix}.txt", f"{prefix}.csv")
    print(f"\nResults saved → {prefix}.csv")
    print(f"Output folder : {args.output_folder}")
