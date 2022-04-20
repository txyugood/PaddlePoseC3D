# Learning Spatiotemporal Features with 3D Convolutional Networks（C3D 基于Paddle复现）
## 1.简介
该论文提出了一种使用在大规模视频数据集上训练的3D卷积网络，可以简单快速的学习时空特征的方法。

该论文有以下三个方法的发现：

1）与2D卷积网络相比，3D卷积网络更适合时空特征学习。

2)在所有层中，3x3x3的卷积核的同构架构是3D卷积网络性能最好的架构之一。

3）该文中的C3D网络，使用一个简单的线性分
类器就达到了SOTA性能。并且由于卷积网络推理速度快，计算也非常高效。
最后，该网络概念上非常简单，易于训练和使用。
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043383-8c26f5d6-d45e-47ae-be18-c23456eb84b9.png" width="800"/>
</div>

## 2.复现精度
在UCF-101数据的测试效果如下表。

| NetWork | epochs | opt | image_size | batch_size | dataset | top1 acc | top5 acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C3D | 100 | SGD | 128x171 | 32 | UCF-101 | 83.27 |95.96 |

## 3.数据集
UCF-101:

第一部分：[https://aistudio.baidu.com/aistudio/datasetdetail/118203](https://aistudio.baidu.com/aistudio/datasetdetail/118203)

第二部分：[https://aistudio.baidu.com/aistudio/datasetdetail/118316](https://aistudio.baidu.com/aistudio/datasetdetail/118316)

预训练模型：

链接: https://pan.baidu.com/s/1s836WYAixWBMnCckXblHbA 

提取码: b6wm 




## 4.环境依赖
PaddlePaddle == 2.2.0
## 5.快速开始
训练：
```shell
cd paddle_c3d
nohup python -u train.py --dataset_root ../UCF-101  --pretrained ../c3d.pdparams > train.log &
tail -f train.log
```
dataset_root: 训练集路径

pretrained: 预训练模型路径

测试：

使用最优模型进行评估.

最优模型下载地址：

链接: https://pan.baidu.com/s/1s836WYAixWBMnCckXblHbA 

提取码: b6wm 

```shell
python test.py --dataset_root ../UCF-101 --pretrained ../best_model.pdparams
```

dataset_root: 训练集路径

pretrained: 预训练模型路径

测试结果

```shell
W1129 19:35:42.080695   591 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W1129 19:35:42.084153   591 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Loading pretrained model from output/best_model/model.pdparams
There are 22/22 variables loaded into Recognizer3D.
[                                                  ] 0/3783, elapsed: 0s, ETA:/home/aistudio/paddle_c3d/datasets/pipelines/transforms.py:377: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
/home/aistudio/paddle_c3d/datasets/pipelines/transforms.py:433: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  results['frame_inds'] = frame_inds.astype(np.int)
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3783/3783, 2.1 task/s, elapsed: 1780s, ETA:     0s
Evaluating top_k_accuracy ...

top1_acc        0.8327
top5_acc        0.9596

Evaluating mean_class_accuracy ...

mean_acc        0.8330
top1_acc: 0.8327
top5_acc: 0.9596
mean_class_accuracy: 0.8330
```

### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://github.com/LDOUBLEV/AutoLog
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.0.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/c3d/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/c3d/train_infer_python.txt 'lite_trai
n_lite_infer'
```

测试结果如截图所示：

![](https://github.com/txyugood/PaddleC3D/blob/master/test_tipc/data/tipc_result.png?raw=true)


## 6.代码结构与详细说明
```shell
├── README.md
├── datasets # 数据集包
│   ├── __init__.py
│   ├── base.py #数据集基类
│   ├── file_client.py # 文件处理类
│   ├── pipelines
│   │   └── transforms.py # 数据增强类
│   ├── rawframe_dataset.py # 数据集类
│   └── utils.py #数据集工具类
├── models
│   ├── __init__.py
│   ├── base.py # 模型基类
│   ├── c3d.py # c3d模型实现
│   ├── i3d_head.py # c3d模型头部实现
│   └── recognizer3d.py # 识别模型框架
├── progress_bar.py #进度条工具
├── test.py # 评估程序
├── test_tipc # TIPC脚本
│   ├── README.md
│   ├── common_func.sh # 通用脚本程序
│   ├── configs
│   │   └── c3d
│   │       └── train_infer_python.txt # 单机单卡配置
│   ├── data
│   │   ├── example.npy # 推理用样例数据
│   │   └── mini_ucf.zip # 训练用小规模数据集
│   ├── output
│   ├── prepare.sh # 数据准备脚本
│   └── test_train_inference_python.sh # 训练推理测试脚本
├── timer.py # 时间工具类
├── train.log # 训练日志
├── train.py # 训练脚本
└── utils.py # 训练工具包
```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| C3D |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 动作识别 |
