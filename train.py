import datetime
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataloader_for_IRDST_bmp import seqDataset, dataset_collate
from model.WTPNet.WTPNet import WTPNet
from model.nets.yolo_training import (
    ModelEMA,
    YOLOLoss,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import EvalCallback, LossHistory
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch


def seed_everything(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ----------------------------- basic setup -----------------------------
    num_frame = 5
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = False

    classes_path = "model_data/classes.txt"
    model_path = ""
    input_shape = [640, 640]

    # ----------------------------- training setup --------------------------
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 4
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    Freeze_Train = False

    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 1e-4
    lr_decay_type = "cos"

    save_period = 15
    save_dir = (
        f"logs/IRDST-H/WTPNet_epoch_{UnFreeze_Epoch}_batch_{Unfreeze_batch_size}_"
        f"optim_{optimizer_type}_lr_{Init_lr}_T_{num_frame}"
    )
    eval_flag = True
    eval_period = 200
    num_workers = 8

    # ----------------------------- dataset setup ---------------------------
    # WTPNet is trained on IRDST-H and DAUB-R in the paper.
    # The txt files follow the detection format used by this WTPNet release:
    # image_path x1,y1,x2,y2,class ...
    DATA_PATH = "/home/ubuntu/nvme1/IRDST-H/"
    train_annotation_path = "/home/ubuntu/nvme1/IRDST-H/train_1.txt"
    val_annotation_path = "/home/ubuntu/nvme1/IRDST-H/val_1.txt"

    # To train on DAUB-R, switch to:
    # DATA_PATH = "/home/ubuntu/nvme1/DAUB-R/"
    # train_annotation_path = "/home/ubuntu/nvme1/DAUB-R/train_1.txt"
    # val_annotation_path = "/home/ubuntu/nvme1/DAUB-R/val_1.txt"
    # save_dir = save_dir.replace("IRDST-H", "DAUB-R")

    # ----------------------------- device setup ----------------------------
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] rank={rank}, local_rank={local_rank}")
            print("GPU count:", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0

    seed_everything(2026)

    class_names, num_classes = get_classes(classes_path)
    model = WTPNet(num_classes=1, num_frame=num_frame)
    weights_init(model)

    if model_path:
        if local_rank == 0:
            print(f"Load weights from {model_path}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        matched, skipped, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                matched.append(k)
            else:
                skipped.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print(f"Loaded keys: {len(matched)}; skipped keys: {len(skipped)}")
            print("Unloaded detection head keys are normal when class count changes.")

    yolo_loss = YOLOLoss(num_classes, fp16, strides=[8])

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + time_str)
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        log_dir = None
        loss_history = None

    scaler = None
    if fp16:
        from torch.cuda.amp import GradScaler

        scaler = GradScaler()

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("SyncBN requires distributed multi-GPU training.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model_train = torch.nn.DataParallel(model).cuda()

    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=log_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
    lr_limit_min = 1e-5 if optimizer_type == "adam" else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # Keep normalization, bias, scalar gates, and relative-position bias out of weight decay.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            name.endswith(".bias")
            or "relative_position_bias_table" in name
            or name.endswith(".alpha")
            or name.endswith(".beta")
            or p.ndim == 1
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    if optimizer_type == "adam":
        optimizer = optim.Adam(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": 0.0},
            ],
            lr=Init_lr_fit,
            betas=(momentum, 0.999),
        )
    else:
        optimizer = optim.SGD(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": weight_decay},
            ],
            lr=Init_lr_fit,
            momentum=momentum,
            nesterov=True,
        )

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small for the selected batch size.")

    train_dataset = seqDataset(train_annotation_path, input_shape[0], num_frame, "train")
    val_dataset = seqDataset(val_annotation_path, input_shape[0], num_frame, "val")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset_collate,
        sampler=train_sampler,
    )
    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset_collate,
        sampler=val_sampler,
    )

    eval_callback = None
    if local_rank == 0:
        eval_callback = EvalCallback(
            model,
            input_shape,
            class_names,
            num_classes,
            val_lines,
            log_dir,
            Cuda,
            eval_flag=eval_flag,
            period=eval_period,
        )

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if distributed:
            train_sampler.set_epoch(epoch)
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(
            model_train,
            model,
            ema,
            yolo_loss,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            fp16,
            scaler,
            save_period,
            log_dir,
            local_rank,
        )
        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
