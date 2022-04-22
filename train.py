import os
import time
import argparse
import random

import paddle
import numpy as np

from datasets import UniformSampleFrames, PoseDecode, PoseCompact, Resize, RandomCrop, CenterCrop, Flip, Normalize, \
    FormatShape, Collect, GeneratePoseTarget, RandomResizedCrop
from datasets import PoseDataset, RepeatDataset
from timer import TimeAverager, calculate_eta
from precise_bn import do_preciseBN

from models.i3d_head import I3DHead
from models.recognizer3d import Recognizer3D
from models.resnet3d_slowonly import ResNet3dSlowOnly
from utils import load_pretrained_model


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    parser.add_argument(
        '--dataset_root',
        dest='dataset_root',
        help='The path of dataset root',
        type=str,
        default='/Users/alex/Downloads/ucf101.pkl')

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        help='The pretrained of model',
        type=str,
        default=None)

    parser.add_argument(
        '--resume',
        dest='resume',
        help='The path of resume model',
        type=str,
        default=None
    )

    parser.add_argument(
        '--last_epoch',
        dest='last_epoch',
        help='The last epoch of resume model',
        type=int,
        default=-1
    )

    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--max_epochs',
        dest='max_epochs',
        help='max_epochs',
        type=int,
        default=12
    )

    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='log_iters',
        type=int,
        default=10
    )

    parser.add_argument(
        '--seed',
        dest='seed',
        help='random seed',
        type=int,
        default=0
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
    right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
    tranforms = [
        UniformSampleFrames(clip_len=48),
        PoseDecode(),
        PoseCompact(hw_ratio=1., allow_imgpad=True),
        Resize(scale=(-1, 64)),
        RandomResizedCrop(area_range=(0.56, 1.0)),
        Resize(scale=(48, 48), keep_ratio=False),
        Flip(flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
        GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    sub_dataset = PoseDataset(ann_file=args.dataset_root, split='train1', data_prefix='', pipeline=tranforms)
    dataset = RepeatDataset(dataset=sub_dataset, times=10)

    val_tranforms = [
        UniformSampleFrames(clip_len=48, num_clips=1, test_mode=True),
        PoseDecode(),
        PoseCompact(hw_ratio=1., allow_imgpad=True),
        Resize(scale=(-1, 56)),
        CenterCrop(crop_size=56),
        GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False),
        FormatShape(input_format='NCTHW'),
        Collect(keys=['imgs', 'label'], meta_keys=[])
    ]
    val_dataset = PoseDataset(ann_file=args.dataset_root, split='test1', data_prefix='',
                              pipeline=val_tranforms)
    #
    backbone = ResNet3dSlowOnly(
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1)
    )
    head = I3DHead(num_classes=101, in_channels=512, spatial_type='avg', dropout_ratio=0.5)
    model = Recognizer3D(backbone=backbone, cls_head=head, train_cfg=dict(), test_cfg=dict(average_clips='prob'))
    if args.pretrained is not None:
        load_pretrained_model(model, args.pretrained)

    batch_size = args.batch_size
    train_loader = paddle.io.DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        return_list=True,
    )

    iters_per_epoch = len(train_loader)
    val_loader = paddle.io.DataLoader(val_dataset,
                                      batch_size=batch_size, shuffle=False, drop_last=False, return_list=True)

    max_epochs = args.max_epochs
    lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=1e-2 / 8, milestones=[9 * iters_per_epoch, 11 * iters_per_epoch],
                                            gamma=0.1)
    grad_clip = paddle.nn.ClipGradByNorm(40)
    optimizer = paddle.optimizer.Momentum(learning_rate=lr, weight_decay=3e-4, parameters=model.parameters(),
                                          grad_clip=grad_clip)

    epoch = 1

    log_iters = args.log_iters
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()

    iters = iters_per_epoch * max_epochs
    iter = 0
    batch_start = time.time()
    best_accuracy = -0.01
    while epoch <= max_epochs:
        total_loss = 0.0
        total_acc = 0.0
        model.train()
        for batch_id, data in enumerate(train_loader):
            reader_cost_averager.record(time.time() - batch_start)
            iter += 1

            outputs = model.train_step(data, optimizer)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            model.clear_gradients()
            lr.step()
            log_vars = outputs['log_vars']
            total_loss += log_vars['loss']
            total_acc += log_vars['top1_acc']

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=batch_size)
            if iter % log_iters == 0:
                avg_loss = total_loss / (batch_id + 1)
                avg_acc = total_acc / (batch_id + 1)
                remain_iters = iters - iter
                avg_train_batch_cost = batch_cost_averager.get_average()
                avg_train_reader_cost = reader_cost_averager.get_average()
                eta = calculate_eta(remain_iters, avg_train_batch_cost)

                print(
                    "[TRAIN] epoch={}, batch_id={}, loss={:.6f}, lr={:.6f},acc={:.3f},"
                    "avg_reader_cost: {} sec, avg_batch_cost: {} sec, avg_samples: {}, avg_ips: {} images/sec  ETA {}"
                        .format(epoch, batch_id + 1,
                                avg_loss, optimizer.get_lr(), avg_acc,
                                avg_train_reader_cost, avg_train_batch_cost,
                                batch_size, batch_size / avg_train_batch_cost,
                                eta))
                reader_cost_averager.reset()
                batch_cost_averager.reset()
            batch_start = time.time()
        if epoch % 2 == 0:
            do_preciseBN(
                model, train_loader, False,
                min(200, len(train_loader)))
        model.eval()
        results = []
        total_val_avg_loss = 0.0
        total_val_avg_acc = 0.0
        for batch_id, data in enumerate(val_loader):
            with paddle.no_grad():
                # outputs = model.val_step(data, optimizer)
                imgs = data['imgs']
                label = data['label']
                result = model(imgs, label, return_loss=False)
            results.extend(result)
        print(f"[EVAL] epoch={epoch}")
        key_score = val_dataset.evaluate(results, metrics=['top_k_accuracy', 'mean_class_accuracy'])

        if key_score['top1_acc'] > best_accuracy:
            best_accuracy = key_score['top1_acc']
            current_save_dir = os.path.join("output", 'best_model')
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            paddle.save(model.state_dict(),
                        os.path.join(current_save_dir, 'model.pdparams'))
            paddle.save(optimizer.state_dict(),
                        os.path.join(current_save_dir, 'model.pdopt'))
        epoch += 1
