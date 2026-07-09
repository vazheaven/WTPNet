import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model.gim.matcher import Matcher


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GIM-aligned WTPNet match frames.")
    parser.add_argument("--dataset_root", type=str, default="/home/ubuntu/nvme1/IRDST-H/")
    parser.add_argument("--splits", nargs="+", default=["train_1.txt", "val_1.txt"])
    parser.add_argument("--num_frame", type=int, default=5)
    return parser.parse_args()


def sorted_sequence_frames(image_path):
    """List frames in the same sequence as image_path using numeric order when possible."""
    p = Path(image_path)
    frames = [q for q in p.parent.iterdir() if q.is_file() and q.suffix.lower() == p.suffix.lower()]

    def sort_key(q):
        try:
            return (0, int(q.stem))
        except ValueError:
            return (1, q.stem)

    frames.sort(key=sort_key)
    return frames


def previous_frames(image_path, num_frame):
    """
    Return T raw frame paths ending at image_path.
    Early frames are padded with the first frame in the sequence, matching the
    WTPNet training/test dataloader behavior.
    """
    p = Path(image_path)
    frames = sorted_sequence_frames(p)
    name_to_idx = {q.name: i for i, q in enumerate(frames)}
    if p.name not in name_to_idx:
        raise FileNotFoundError(f"Current frame is not listed under its sequence directory: {p}")

    cur_idx = name_to_idx[p.name]
    seq = frames[max(0, cur_idx - (num_frame - 1)):cur_idx + 1]
    while len(seq) < num_frame:
        seq.insert(0, frames[0])
    return [str(q) for q in seq]


def output_match_dir(image_path):
    """Map images/<seq>/<frame>.bmp to matches/<seq>/<frame>/."""
    p = Path(image_path)
    parts = ["matches" if part == "images" else part for part in p.parts[:-1]]
    return Path(*parts) / p.stem


def match(images):
    image_data = []
    Xc = images[-1]
    for Xr in images:
        image_data.append(gim.match(Xc, Xr))
    return image_data


if __name__ == "__main__":
    args = parse_args()
    split_paths = [os.path.join(args.dataset_root, split) for split in args.splits]

    global gim
    gim = Matcher()

    for txt_path in split_paths:
        with open(txt_path) as f:
            img_idx = [line.strip("\n").split()[0] for line in f.readlines()]

        for file_name in tqdm(img_idx, desc=Path(txt_path).name):
            save_path = output_match_dir(file_name)
            os.makedirs(save_path, exist_ok=True)

            images = previous_frames(file_name, args.num_frame)
            image_data = match(images)
            for i, img_arr in enumerate(image_data):
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                Image.fromarray(img_arr).save(os.path.join(save_path, f"match_{i + 1}.png"))
