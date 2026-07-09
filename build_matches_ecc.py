import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    return img

def to_align_input(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    return mag

def ecc_align_to_ref(src_gray: np.ndarray, ref_gray: np.ndarray, motion: str = "affine",
                     iters: int = 200, eps: float = 1e-6, max_size: int = 512) -> np.ndarray:
    """
    Align src to ref with ECC and return the aligned grayscale uint8 frame.
    Falls back to the original frame if ECC estimation fails.
    """
    H, W = ref_gray.shape[:2]

    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }
    if motion not in motion_map:
        raise ValueError(f"motion must be one of {list(motion_map.keys())}")

        h, w = img.shape[:2]
        s = 1.0
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return img, s

    ref_small, s_ref = resize_keep_ar(ref_gray, max_size)
    src_small, s_src = resize_keep_ar(src_gray, max_size)
    if abs(s_ref - s_src) > 1e-6:
        s = s_ref
    else:
        s = s_ref

    ref_in = to_align_input(ref_small)
    src_in = to_align_input(src_small)

    warp_mode = motion_map[motion]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    try:
        # template=ref, input=src
        cv2.findTransformECC(ref_in, src_in, warp, warp_mode, criteria, None, 5)
    except cv2.error:
        return src_gray

        warp_full = warp.copy()
        warp_full[0, 2] /= s
        warp_full[1, 2] /= s
        aligned = cv2.warpAffine(
            src_gray, warp_full, (W, H),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        warp_full = warp.copy()
        warp_full[0, 2] /= s
        warp_full[1, 2] /= s
        aligned = cv2.warpPerspective(
            src_gray, warp_full, (W, H),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )

    return aligned

def list_sequences(images_dir: Path, ext: str):
    exts = {f".{ext.lower()}"}
    seqs = []
    bmp_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if bmp_files:
        seqs.append((images_dir.name, sorted(bmp_files, key=lambda x: x.stem)))
        return seqs

    for sub in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
        frames = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not frames:
            continue
        def sort_key(p: Path):
            try:
                return int(p.stem)
            except ValueError:
                return p.stem
        frames = sorted(frames, key=sort_key)
        seqs.append((sub.name, frames))
    return seqs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="Dataset root containing images, labels, and txt splits.")
    ap.add_argument("--T", type=int, default=5, help="Temporal window length.")
    ap.add_argument("--ext", type=str, default="bmp", help="Raw frame extension.")
    ap.add_argument("--motion", type=str, default="affine", choices=["translation","euclidean","affine","homography"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max_size", type=int, default=512, help="Maximum side length used during ECC estimation.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing match_*.png files.")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    images_dir = root / "images"
    matches_dir = root / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    seqs = list_sequences(images_dir, args.ext)
    if not seqs:
        raise RuntimeError(f"No *.{args.ext} found under {images_dir}")

    for seq_name, frames in seqs:
        seq_out_dir = matches_dir / seq_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        for t in tqdm(range(len(frames)), desc=f"seq {seq_name}", ncols=100):
            key_path = frames[t]
            key_stem = key_path.stem
            out_dir = seq_out_dir / key_stem
            out_dir.mkdir(parents=True, exist_ok=True)

            ref = imread_gray(key_path)

            start = max(0, t - (args.T - 1))
            idxs = list(range(start, t + 1))
            while len(idxs) < args.T:
                idxs = [0] + idxs
            for k, idx in enumerate(idxs, start=1):
                out_path = out_dir / f"match_{k}.png"
                if out_path.exists() and not args.overwrite:
                    continue

                src = imread_gray(frames[idx])
                if idx == t:
                    aligned = src
                else:
                    aligned = ecc_align_to_ref(src, ref, motion=args.motion,
                                               iters=args.iters, eps=args.eps, max_size=args.max_size)

                cv2.imwrite(str(out_path), aligned)

    print("Done. matches generated under:", matches_dir)

if __name__ == "__main__":
    main()


# python build_matches_ecc.py --dataset_root /home/ubuntu/nvme1/IRDST-H --T 5 --ext bmp --motion affine



