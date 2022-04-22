import os
import argparse

import paddle

from models.resnet3d_slowonly import ResNet3dSlowOnly
from models.i3d_head import I3DHead
from models.recognizer3d import Recognizer3D
from utils import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument(
        "--input_shape",
        nargs='+',
        help="Export the model with fixed input shape, such as 1 3 1024 1024.",
        type=int,
        default=None)

    return parser.parse_args()


def main(args):
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
    net = Recognizer3D(backbone=backbone, cls_head=head)

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        print('Loaded trained params of model successfully.')


    shape = [None, 1, 3, 16, 112, 112]

    new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    # yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    # with open(yml_file, 'w') as file:
    #     transforms = cfg.export_config.get('transforms', [{
    #         'type': 'Normalize'
    #     }])
    #     data = {
    #         'Deploy': {
    #             'transforms': transforms,
    #             'model': 'model.pdmodel',
    #             'params': 'model.pdiparams'
    #         }
    #     }
    #     yaml.dump(data, file)

    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)