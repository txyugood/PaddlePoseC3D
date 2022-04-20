import os
import argparse

import paddle

from datasets import SampleFrames, RawFrameDecode, Resize, RandomCrop, CenterCrop, Flip, Normalize, FormatShape, Collect
from datasets import RawframeDataset
from models.c3d import C3D
from models.i3d_head import I3DHead
from models.recognizer3d import Recognizer3D
from utils import load_pretrained_model
from progress_bar import ProgressBar


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/Users/alex/baidu/mmaction2/data/ucf101/')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default='output/best_model/model.pdparams')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    tranforms = [
        SampleFrames(clip_len=16, frame_interval=1, num_clips=10, test_mode=True),
        RawFrameDecode(),
        Resize(scale=(128, 171)),
        CenterCrop(crop_size=112),
        Normalize(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    dataset = RawframeDataset(ann_file=os.path.join(args.dataset_root, 'ucf101_val_split_1_rawframes.txt'),
                              pipeline=tranforms, data_prefix=os.path.join(args.dataset_root, "rawframes"))

    loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=5,
        shuffle=False,
        drop_last=False,
        return_list=True,
    )

    backbone = C3D(dropout_ratio=0.5, init_std=0.005)
    head = I3DHead(num_classes=101, in_channels=4096, spatial_type=None, dropout_ratio=0.5, init_std=0.01)
    model = Recognizer3D(backbone=backbone, cls_head=head)
    load_pretrained_model(model, args.pretrained)

    model.eval()
    results = []
    prog_bar = ProgressBar(len(dataset))
    for batch_id, data in enumerate(loader):
        with paddle.no_grad():
            imgs = data['imgs']
            label = data['label']
            result = model(imgs, label, return_loss=False)
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    eval_res = dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
    for name, val in eval_res.items():
        print(f'{name}: {val:.04f}')
