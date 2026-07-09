import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from model.WTPNet.WTPNet import WTPNet
from model.WTPNet.tdc_repconv import RepConv3D
from utils.utils import get_classes, show_config
from utils.utils_bbox import decode_outputs, non_max_suppression


# ----------------------------- dataset setup -----------------------------
# WTPNet reports results on IRDST-H and DAUB-R. The default here is IRDST-H.
cocoGt_path = "/home/ubuntu/nvme1/IRDST-H/test.json"
dataset_root = "/home/ubuntu/nvme1/IRDST-H/images"

# To evaluate DAUB-R, switch to:
# cocoGt_path = "/home/ubuntu/nvme1/DAUB-R/test.json"
# dataset_root = "/home/ubuntu/nvme1/DAUB-R/images"

temp_save_path = "results/IRDST-H/WTPNet_eval_T5"
model_path = "logs/IRDST-H/WTPNet_epoch_100_batch_4_optim_sgd_lr_0.01_T_5/best_epoch_weights.pth"
num_frame = 5
device = "cuda:0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def letterbox_image_batch(images, target_size=(640, 640), color=(128, 128, 128)):
    """Resize a list of RGB images with letterbox padding."""
    w, h = target_size
    output = np.full((len(images), h, w, 3), color, dtype=np.uint8)
    for i, img in enumerate(images):
        ih, iw = img.shape[:2]
        scale = min(w / iw, h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (h - nh) // 2, (w - nw) // 2
        output[i, top:top + nh, left:left + nw, :] = resized
    return output


def sorted_numeric_frames(image_dir: Path, suffix: str):
    """Sort frames by numeric filename when possible, otherwise lexically."""
    frames = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix.lower()]

    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    frames.sort(key=key)
    return frames


def first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


_FRAMES_CACHE = {}
_NAME2IDX_CACHE = {}


def get_history_imgs_irdst(image_path: str, T: int):
    """
    Return the 2T-frame WTPNet input:
      [aligned match_1..match_T] + [raw frame_{t-T+1}..frame_t]
    """
    p = Path(image_path)
    if not p.is_file():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    parts = ["matches" if part == "images" else part for part in p.parts]
    matches_dir = Path(*parts[:-1]) / p.stem
    match_paths = []
    for i in range(1, T + 1):
        mp = first_existing(
            [
                str(matches_dir / f"img_{i}.png"),
                str(matches_dir / f"match_{i}.png"),
                str(matches_dir / f"img_{i}.jpg"),
                str(matches_dir / f"match_{i}.jpg"),
            ]
        )
        if mp is None:
            raise FileNotFoundError(f"Missing aligned frame {i} under {matches_dir}")
        match_paths.append(mp)

    image_dir = p.parent
    suffix = p.suffix.lower()
    cache_key = (str(image_dir), suffix)
    if cache_key not in _FRAMES_CACHE:
        frames = sorted_numeric_frames(image_dir, suffix)
        if not frames:
            raise FileNotFoundError(f"No raw frames with suffix {suffix} under {image_dir}")
        _FRAMES_CACHE[cache_key] = frames
        _NAME2IDX_CACHE[cache_key] = {q.name: idx for idx, q in enumerate(frames)}

    frames = _FRAMES_CACHE[cache_key]
    name2idx = _NAME2IDX_CACHE[cache_key]
    if p.name not in name2idx:
        raise FileNotFoundError(f"Current frame not found in sorted list: {p}")

    cur_idx = name2idx[p.name]
    seq = frames[max(0, cur_idx - (T - 1)):cur_idx + 1]
    while len(seq) < T:
        seq.insert(0, frames[0])
    return match_paths + [str(q) for q in seq]


class WTPNetCOCOEvaluator:
    def __init__(self):
        self.model_path = model_path
        self.classes_path = "model_data/classes.txt"
        self.input_shape = [640, 640]
        self.confidence = 0.001
        self.nms_iou = 0.5
        self.letterbox_image = True

        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.net = WTPNet(self.num_classes, num_frame=num_frame)
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict, strict=True)

        # The temporal-difference branch uses re-parameterizable RepConv3D blocks.
        for m in self.net.modules():
            if isinstance(m, RepConv3D):
                m.switch_to_deploy()

        self.net = nn.DataParallel(self.net).to(device).eval()
        show_config(**self.__dict__)

    @torch.no_grad()
    def detect_one(self, image_id, rgb_images_2t, results, clsid2catid):
        image_shape = np.array(rgb_images_2t[-1].shape[:2])
        imgs = letterbox_image_batch(rgb_images_2t, target_size=tuple(self.input_shape))
        imgs = imgs.astype(np.float32) / 255.0
        imgs = imgs.transpose(3, 0, 1, 2)[None]
        imgs_t = torch.from_numpy(imgs).to(device)

        outputs = self.net(imgs_t)
        outputs = decode_outputs(outputs, self.input_shape)
        outputs = non_max_suppression(
            outputs,
            self.num_classes,
            self.input_shape,
            image_shape,
            self.letterbox_image,
            conf_thres=self.confidence,
            nms_thres=self.nms_iou,
        )

        if outputs[0] is None:
            return results

        top_label = np.array(outputs[0][:, 6], dtype="int32")
        top_conf = outputs[0][:, 4] * outputs[0][:, 5]
        top_boxes = outputs[0][:, :4]
        for i, c in enumerate(top_label):
            top, left, bottom, right = top_boxes[i]
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(clsid2catid[c]),
                    "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                    "score": float(top_conf[i]),
                }
            )
        return results


if __name__ == "__main__":
    os.makedirs(temp_save_path, exist_ok=True)
    cocoGt = COCO(cocoGt_path)
    img_ids = cocoGt.getImgIds()
    cat_ids = sorted(cocoGt.getCatIds())
    clsid2catid = {i: cat_ids[i] for i in range(len(cat_ids))}

    evaluator = WTPNetCOCOEvaluator()
    results = []
    for image_id in tqdm(img_ids):
        info = cocoGt.loadImgs(image_id)[0]
        image_path = os.path.join(dataset_root, info["file_name"])
        paths_2t = get_history_imgs_irdst(image_path, num_frame)
        rgb_images = []
        for p in paths_2t:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(f"cv2.imread failed: {p}")
            rgb_images.append(im[:, :, ::-1])
        results = evaluator.detect_one(image_id, rgb_images, results, clsid2catid)

    out_json = os.path.join(temp_save_path, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f)

    cocoDt = cocoGt.loadRes(out_json)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print("Done. Results saved to:", out_json)
