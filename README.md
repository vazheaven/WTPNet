# Wavelet-Guided Temporal Photometric Network for Multi-Frame Infrared Small Target Detection

This repository contains the official implementation of **WTPNet**, a prior-guided three-branch cross-attention network for multi-frame infrared small target detection.

WTPNet follows the training and evaluation environment of [TDCNet](https://github.com/IVPLabs/TDCNet), while reorganizing the model implementation around the three modules described in the paper:

- **WTCP**: wavelet-guided temporal context prior, implemented in `model/WTPNet/WTCP.py`.
- **DPRE**: directional photometric residual enhancement, implemented in `model/WTPNet/DPRE.py`.
- **LCSF**: large-context scale-focused fusion, implemented in `model/WTPNet/LCSF.py`.

The main model class is `WTPNet` in `model/WTPNet/WTPNet.py`.

## Framework

WTPNet decouples multi-frame information into motion-sensitive temporal cues, context-aware temporal representations, and current-frame spatial appearance. WTCP builds temporal-frequency and wavelet priors to guide the temporal context branch. DPRE preserves weak shallow photometric responses in the current-frame branch. LCSF refocuses hierarchical features during multi-scale aggregation.

## Environment

The environment is the same as TDCNet:

- Python 3.12.5
- PyTorch 2.7.0 + CUDA 12.6
- tqdm 4.65.2
- pycocotools 2.0.8
- OpenCV 4.12.0
- NumPy 2.2.6

Example setup:

```bash
conda create -n wtpnet python=3.12.5
conda activate wtpnet
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default wheel is not suitable.

## Dataset

WTPNet uses the **IRDST-H** and **DAUB-R** datasets provided by the [MoPKL project](https://github.com/UESTC-nnLab/MoPKL). We thank the MoPKL authors for collecting, reorganizing, and releasing these bounding box-based moving infrared small target detection datasets.

Dataset links from MoPKL:

- [IRDST-H](https://pan.baidu.com/s/1rLNyw1sIitSVU1yzzCQnyA?pwd=c4ar), extraction code: `c4ar`
- [DAUB-R](https://pan.baidu.com/s/1Bay8pNlDJIKCD75O3tdhuA?pwd=jya7), extraction code: `jya7`

Expected dataset structure:

```text
DATASET_ROOT/
  images/
    1/
      0.bmp
      1.bmp
      ...
    2/
      ...
  matches/
    1/
      0/
        img_1.png
        ...
        img_5.png
      1/
        img_1.png
        ...
  train_1.txt
  val_1.txt
  test.json
```

The training txt files use the common detection format:

```text
image_path x1,y1,x2,y2,class ...
```

For example:

```text
/home/ubuntu/nvme1/IRDST-H/images/1/0.bmp 10,20,18,28,0
```

The released training and testing scripts are configured for IRDST-H by default.
Use the same structure with the root name changed to `DAUB-R` when reproducing
DAUB-R experiments.

The aligned `matches/` frames should be generated before training and testing.
WTPNet reads `img_1.png` ... `img_5.png` or `match_1.png` ... `match_5.png`
under each `matches/<sequence>/<frame_stem>/` directory.

If background-aligned reference frames are required, use:

```bash
python background_alignment.py
```

For ECC-based matching on BMP-style datasets, use:

```bash
python build_matches_ecc.py --dataset_root /path/to/dataset --T 5 --ext bmp
```

## Training

Edit dataset paths in `train.py`:

```python
DATA_PATH = "/path/to/dataset/"
train_annotation_path = "/path/to/dataset/train.txt"
val_annotation_path = "/path/to/dataset/val.txt"
```

The default released script uses IRDST-H:

```python
DATA_PATH = "/home/ubuntu/nvme1/IRDST-H/"
train_annotation_path = "/home/ubuntu/nvme1/IRDST-H/train_1.txt"
val_annotation_path = "/home/ubuntu/nvme1/IRDST-H/val_1.txt"
```

Switch the three paths to `DAUB-R` for DAUB-R training.

Then run:

```bash
python train.py
```

Checkpoints and logs are saved under `logs/`.

## Testing

Edit dataset, checkpoint, and COCO annotation paths in `test.py`. The default is IRDST-H:

```python
cocoGt_path = "/home/ubuntu/nvme1/IRDST-H/test.json"
dataset_root = "/home/ubuntu/nvme1/IRDST-H/images"
model_path = "logs/IRDST-H/WTPNet_epoch_100_batch_4_optim_sgd_lr_0.01_T_5/best_epoch_weights.pth"
```

Switch the paths to `DAUB-R` for DAUB-R evaluation. Then run:

```bash
python test.py
```

Detection results are saved under `results/`.

## Model Import

```python
from model.WTPNet import WTPNet

model = WTPNet(num_classes=1, num_frame=5)
```

## Acknowledgements

This codebase is built on the public TDCNet implementation and follows its environment and training pipeline. We thank the TDCNet authors for their excellent work. We also thank the MoPKL authors for releasing the IRDST-H and DAUB-R datasets used by WTPNet.

## Citation

Citation information will be added after the paper metadata is finalized.
