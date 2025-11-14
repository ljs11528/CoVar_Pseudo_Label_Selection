#!/usr/bin/env python3
"""
Quick inference script to run a trained UniMatch-V2 / DPT model on a selected set of PASCAL images
and save for each image a folder containing:
  - original.jpg (the input image)
  - result.png   (predicted segmentation, P mode with PASCAL palette)
  - label.png    (ground-truth segmentation from SegmentationClass, if present)

Example:
  python scripts/test_selected.py \
    --checkpoint /workspace/coredata/UniMatch-V2/exp/pascal/unimatch_v2/dinov2_base/366/best.pth \
    --images /workspace/coredata/CSL_new/data/Pascal/JPEGImages \
    --labels /workspace/coredata/CSL_new/data/Pascal/SegmentationClass \
    --outdir ./results_selected

"""
import argparse
import os
from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

import sys
# make repo root importable when script is executed directly
# script is in <repo>/scripts, so parent (.. ) is repo root which contains `model/`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.semseg.dpt import DPT


SELECTED_IDS = [
    '2007_001587','2007_001884','2007_002119','2007_005547','2007_005857',
    '2007_006449','2007_007165','2007_009084','2008_002942','2008_003141',
    '2008_003709','2008_004396','2008_006219','2009_003904','2010_001752',
    '2010_003468','2010_005245','2010_005705','2011_000953','2011_002124'
]


def pascal_palette():
    # standard PASCAL VOC palette generation (256 x 3 -> 768 entries)
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


def load_checkpoint(model, ckpt_path, use_ema=False, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # checkpoint might be a state_dict directly or a dict with key 'model'/'model_ema'
    if isinstance(ckpt, dict):
        key = 'model_ema' if use_ema and 'model_ema' in ckpt else 'model'
        if key in ckpt:
            state_dict = ckpt[key]
        else:
            # maybe user saved the full state_dict at top-level
            state_dict = ckpt
    else:
        state_dict = ckpt

    # strip `module.` prefixes if present
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if k.startswith('module.'):
            nk = k[len('module.'):]
        new_state[nk] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(img_path, device, patch_size=14):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size

    # resize to nearest multiple of patch_size (ceil) to satisfy backbone assertion
    new_h = int(((orig_h + patch_size - 1) // patch_size) * patch_size)
    new_w = int(((orig_w + patch_size - 1) // patch_size) * patch_size)
    if (new_w, new_h) != (orig_w, orig_h):
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    else:
        img_resized = img

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized).unsqueeze(0).to(device)
    return img, tensor, (orig_w, orig_h), (new_w, new_h)


def predict_and_save(model, img_pil, tensor, out_folder, palette, orig_size=None, input_size=None):
    # model -> logits [1, C, H_out, W_out]
    device = tensor.device
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # upsample prediction to original image size using nearest
    pred_pil = Image.fromarray(pred, mode='P')
    # pred is in the model input resolution (input_size). Resize back to original image size
    if orig_size is None:
        target_size = img_pil.size
    else:
        target_size = orig_size
    pred_resized = pred_pil.resize(target_size, resample=Image.NEAREST)
    pred_resized.putpalette(palette)
    pred_resized.save(os.path.join(out_folder, 'result.png'))

    # save visualization also as a colored PNG (optional): here we reuse the palette image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', required=True, help='path to JPEGImages folder')
    parser.add_argument('--labels', default='/workspace/coredata/CSL_new/data/Pascal/SegmentationClass', help='path to SegmentationClass')
    parser.add_argument('--outdir', default='./results_selected')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--encoder', default='base', help='DINOv2 encoder size: small|base|large|giant')
    parser.add_argument('--nclass', type=int, default=21)
    parser.add_argument('--use_ema', action='store_true', help='load model_ema if available')
    args = parser.parse_args()

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model = DPT(encoder_size=args.encoder, nclass=args.nclass)
    model = load_checkpoint(model, args.checkpoint, use_ema=args.use_ema, device=device)

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

        # copy original image
        shutil.copy2(img_path, out_folder / 'original.jpg')

        # copy label if exists
        if label_path.exists():
            shutil.copy2(label_path, out_folder / 'label.png')
        else:
            # create an empty placeholder
            Image.new('P', (1, 1)).save(out_folder / 'label.png')

        # preprocess and predict
        img_pil, tensor, orig_size, input_size = preprocess_image(str(img_path), device)
        predict_and_save(model, img_pil, tensor, str(out_folder), palette, orig_size=orig_size, input_size=input_size)

        print(f"Saved results for {img_id} -> {out_folder}")


if __name__ == '__main__':
    main()
