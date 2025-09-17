import os
import json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from groundingdino.util.base_api import preprocess_caption


def point_density(shape, points):
    """Create a full-resolution discrete density map."""
    h, w = shape
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in points:
        x = min(w - 1, max(0, int(round(x))))
        y = min(h - 1, max(0, int(round(y))))
        density[y, x] += 1
    return density


def main(image_root, anno_file, split_file, out_dir):
    with open(anno_file, "r") as f:
        annotations = json.load(f)
    with open(split_file, "r") as f:
        splits = json.load(f)

    pairs = set(tuple(x) for split in splits.values() for x in split)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id, caption in tqdm(pairs, desc="Generating point maps"):
        img_path = Path(image_root) / img_id
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        points = annotations[img_id][caption]["points"]
        pt_map = point_density((h, w), points)
        img_out = out_dir / img_id[:-4]
        img_out.mkdir(parents=True, exist_ok=True)
        base_name = preprocess_caption(caption).strip(".")
        np.save(img_out / f"{base_name}_pt.npy", pt_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate discrete point density maps for REC-8K")
    parser.add_argument("--image-root", default="datasets/rec-8k/rec-8k", help="path to image folder")
    parser.add_argument("--anno-file", default="datasets/rec-8k/annotations.json")
    parser.add_argument("--split-file", default="datasets/rec-8k/splits.json")
    parser.add_argument("--out-dir", default="datasets/rec-8k/density_maps")
    args = parser.parse_args()

    main(args.image_root, args.anno_file, args.split_file, args.out_dir)
