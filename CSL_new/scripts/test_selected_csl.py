#!/usr/bin/env python3
"""
Run a CSL-trained model checkpoint on a fixed set of PASCAL images and save original/result/label.

Example:
  python scripts/test_selected_csl.py \
    --config /workspace/coredata/CSL_new/configs/pascal.yaml \
    --checkpoint /workspace/coredata/CSL/exp/pascal/CSL/r101/1_4/checkpoints/epoch=54-val_mIOU=79.57.ckpt \
    --images /workspace/coredata/CSL_new/data/Pascal/JPEGImages \
    --labels /workspace/coredata/CSL_new/data/Pascal/SegmentationClass \
    --outdir /workspace/coredata/CSL_new/results_selected_csl
"""
import argparse
import os
from pathlib import Path
import shutil
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.model_helper import ModelBuilder


SELECTED_IDS = [
    '2007_001587','2007_001884','2007_002119','2007_005547','2007_005857',
    '2007_006449','2007_007165','2007_009084','2008_002942','2008_003141',
    '2008_003709','2008_004396','2008_006219','2009_003904','2010_001752',
    '2010_003468','2010_005245','2010_005705','2011_000953','2011_002124'
]


def pascal_palette():
    palette = []
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette.extend([r, g, b])
    return palette


def load_ckpt_to_model(model, ckpt_path, map_location='cpu'):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)


def preprocess_image(img_path, device):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img).to(device)
    return img, tensor


def infer_and_save(model, img_pil, tensor, out_folder, palette, device):
    model.to(device).eval()
    with torch.no_grad():
        inp = tensor.unsqueeze(0)
        out = model(inp, False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = out.softmax(dim=1)
        pred = probs.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # save P-mode mask with palette
    mask_pil = Image.fromarray(pred, mode='P')
    mask_pil.putpalette(palette)
    mask_pil.save(os.path.join(out_folder, 'result.png'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='/workspace/coredata/CSL_new/configs/pascal.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', default='/workspace/coredata/CSL_new/data/Pascal/JPEGImages')
    p.add_argument('--labels', default='/workspace/coredata/CSL_new/data/Pascal/SegmentationClass')
    p.add_argument('--outdir', default='./results_selected_csl')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    import yaml
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # switch cwd to project root so relative pretrained paths (e.g. pretrained/resnet101.pth)
    # resolve correctly when model builder attempts to load backbone weights
    try:
        config_abspath = os.path.abspath(args.config)
        project_root = os.path.dirname(os.path.dirname(config_abspath))
        if os.path.isdir(project_root):
            os.chdir(project_root)
    except Exception:
        pass

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model = ModelBuilder(cfg['model'])
    load_ckpt_to_model(model, args.checkpoint, map_location='cpu')

    palette = pascal_palette()

    images_root = Path(args.images)
    labels_root = Path(args.labels)

    for img_id in SELECTED_IDS:
        img_name = f"{img_id}.jpg"
        label_name = f"{img_id}.png"
        img_path = images_root / img_name
        label_path = labels_root / label_name

        if not img_path.exists():
            print(f"Image not found: {img_path}, skipping")
            continue

        out_folder = outroot / img_id
        out_folder.mkdir(parents=True, exist_ok=True)

        # copy original
        shutil.copy2(img_path, out_folder / 'original.jpg')

        # copy label if exists
        if label_path.exists():
            shutil.copy2(label_path, out_folder / 'label.png')
        else:
            Image.new('P', (1, 1)).save(out_folder / 'label.png')

        img_pil, tensor = preprocess_image(str(img_path), device)
        infer_and_save(model, img_pil, tensor, str(out_folder), palette, device)

        print(f"Saved results for {img_id} -> {out_folder}")


if __name__ == '__main__':
    main()
