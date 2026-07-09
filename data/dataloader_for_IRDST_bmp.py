# dataloader_for_IRDST_bmp.py
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


def cvtColor(image: Image.Image) -> Image.Image:
    """Ensure RGB."""
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        return image.convert("RGB")


def preprocess(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image /= 255.0
    return image


def _letterbox_pil(img: Image.Image, out_w: int, out_h: int):
    """
    Letterbox resize to (out_w, out_h).
    Returns:
      new_img: PIL Image RGB
      params: (iw, ih, nw, nh, dx, dy)
    """
    iw, ih = img.size
    scale = min(out_w / iw, out_h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (out_w - nw) // 2
    dy = (out_h - nh) // 2

    img = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new("RGB", (out_w, out_h), (128, 128, 128))
    new_img.paste(img, (dx, dy))
    return new_img, (iw, ih, nw, nh, dx, dy)


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _sorted_frames_in_dir(image_dir: Path, suffix: str):
    """
    Return sorted list of frames in directory with given suffix (e.g. '.bmp').
    Prefer numeric sort by stem; fallback to lexical.
    """
    frames = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()]

    def sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    frames.sort(key=sort_key)
    return frames


class seqDataset(Dataset):
    """
    WTPNet style dataset:
      - reads matches (aligned frames) from matches/<seq>/<frame_stem>/img_k.png or match_k.png  (k=1..T)
      - reads raw consecutive frames from images/<seq>/... (T frames: earliest -> current)
      - outputs concatenated frames: [matches_1..T, raw_1..T]  -> total 2T frames
    """

    def __init__(self, dataset_path, image_size, num_frame=5, type="train"):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        self.txt_path = dataset_path

        with open(self.txt_path, "r") as f:
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip("\n").split()
                self.img_idx.append(line[0])
                # each box: x1,y1,x2,y2,cls (int)
                self.anno_idx.append(np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]]))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        images, box = self.get_data(index)
        images = np.array(images)
        # [2T,H,W,3] -> normalize -> [3,2T,H,W]
        images = np.transpose(preprocess(images), (3, 0, 1, 2))

        if len(box) != 0:
            # xyxy -> xywh(center)
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + (box[:, 2:4] / 2)
        return images, box

    def get_data(self, index):
        h, w = self.image_size, self.image_size
        file_name = self.img_idx[index]
        p = Path(file_name)

        # -------------------------
        # 1) matches: matches/<seq>/<frame_stem>/(img_k.png or match_k.png)
        # -------------------------
        parts = list(p.parts)
        parts = ["matches" if part == "images" else part for part in parts]
        matches_dir = Path(*parts[:-1]) / p.stem  # remove .bmp/.png etc

        images = []
        for i in range(self.num_frame):
            cand = [
                str(matches_dir / f"img_{i + 1}.png"),
                str(matches_dir / f"match_{i + 1}.png"),
            ]
            img_path = _find_first_existing(cand)
            if img_path is None:
                raise FileNotFoundError(
                    f"[WTPNet] Missing matches for index={index}\n"
                    f"Expected one of: {cand}\n"
                    f"matches_dir = {matches_dir}\n"
                    f"raw file = {file_name}"
                )

            img = Image.open(img_path)
            img = cvtColor(img)
            new_img, _ = _letterbox_pil(img, w, h)
            images.append(np.array(new_img, np.float32))

        # -------------------------
        # 2) raw frames: take T consecutive frames ending at current frame
        #    robustly by sorting directory files (instead of '%d.png' hardcode)
        # -------------------------
        image_dir = p.parent
        raw_suffix = p.suffix.lower()  # e.g. '.bmp'

        frames = _sorted_frames_in_dir(image_dir, raw_suffix)
        if not frames:
            raise FileNotFoundError(f"[WTPNet] No raw frames with suffix {raw_suffix} under: {image_dir}")

        # locate current frame in sorted list
        try:
            cur_idx = next(i for i, q in enumerate(frames) if q.name == p.name)
        except StopIteration:
            raise FileNotFoundError(f"[WTPNet] Current frame not found in its directory list: {p}")

        start = max(0, cur_idx - (self.num_frame - 1))
        seq = frames[start:cur_idx + 1]
        while len(seq) < self.num_frame:
            seq.insert(0, frames[0])  # pad with first frame

        # read raw frames (earliest -> current), and keep keyframe letterbox params for bbox transform
        image_data = []
        key_params = None
        for q in seq:
            img = Image.open(str(q))
            img = cvtColor(img)
            new_img, params = _letterbox_pil(img, w, h)
            image_data.append(np.array(new_img, np.float32))
            key_params = params  # last one will be keyframe params

        # append raw frames after matches
        for im in image_data:
            images.append(im.copy())

        # -------------------------
        # 3) bbox transform (must copy to avoid corrupting cached labels)
        # -------------------------
        label_data = self.anno_idx[index].copy()
        if len(label_data) > 0:
            np.random.shuffle(label_data)

            iw, ih, nw, nh, dx, dy = key_params
            label_data[:, [0, 2]] = label_data[:, [0, 2]] * nw / iw + dx
            label_data[:, [1, 3]] = label_data[:, [1, 3]] * nh / ih + dy

            label_data[:, 0:2][label_data[:, 0:2] < 0] = 0
            label_data[:, 2][label_data[:, 2] > w] = w
            label_data[:, 3][label_data[:, 3] > h] = h

            box_w = label_data[:, 2] - label_data[:, 0]
            box_h = label_data[:, 3] - label_data[:, 1]
            label_data = label_data[np.logical_and(box_w > 1, box_h > 1)]

        images = np.array(images, dtype=np.float32)
        label_data = np.array(label_data, dtype=np.float32)
        return images, label_data


def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes


