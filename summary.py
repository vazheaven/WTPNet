import os

from torch import nn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from thop import clever_format, profile

from model.WTPNet.WTPNet import WTPNet
from model.WTPNet.tdc_repconv import RepConv3D

if __name__ == "__main__":
    input_shape = [640, 640]
    num_classes = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frame = 5
    m = WTPNet(num_classes, num_frame=num_frame)
    for mm in m.modules():
        if isinstance(mm, RepConv3D):
            mm.switch_to_deploy()

    # WTPNet receives 2T frames: T aligned match frames and T raw temporal frames.
    dummy_input = torch.randn(1, 3, num_frame * 2, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m.to(device), (dummy_input,), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))


    from data.dataloader_for_IRDST_bmp import seqDataset, dataset_collate
    from torch.utils.data import DataLoader
    import time

    val_annotation_path = "/home/ubuntu/nvme1/IRDST-H/val_1.txt"
    if not os.path.isfile(val_annotation_path):
        print(f"Skip FPS benchmark because validation txt was not found: {val_annotation_path}")
        raise SystemExit(0)

    max_iter = 2000
    log_interval = 50
    num_warmup = 20
    pure_inf_time = 0
    fps = 0
    val_dataset = seqDataset(val_annotation_path, input_shape[0], num_frame, 'val')
    gen_val     = DataLoader(val_dataset, shuffle = False, batch_size = 1, num_workers = 10, pin_memory=True,
                                    drop_last=True, collate_fn=dataset_collate)
    m = nn.DataParallel(m).cuda()

    # benchmark with 2000 image and take the average
    for i, data in enumerate(gen_val):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            m(data[0].to('cuda'))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    print("FPS:" ,fps)
