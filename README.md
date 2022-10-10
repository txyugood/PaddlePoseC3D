# Revisiting Skeleton-based Action Recognition（PoseC3D 基于Paddle复现）

## 该项目以添加到[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/model_zoo/recognition/posec3d.md)

## 1.简介
人体骨架作为人类行为的一种简洁的表现形式，近年来受到越来越多的关注。许多基于骨架的动作识别方法都采用了图卷积网络（GCN）来提取人体骨架上的特征。尽管在以前的工作中取得了积极的成果，但基于GCN的方法在健壮性、互操作性和可扩展性方面受到限制。在本文中，作者提出了一种新的基于骨架的动作识别方法PoseC3D，它依赖于3D热图堆栈而不是图形序列作为人体骨架的基本表示。与基于GCN的方法相比，PoseC3D在学习时空特征方面更有效，对姿态估计噪声更具鲁棒性，并且在跨数据集环境下具有更好的通用性。此外，PoseC3D可以在不增加计算成本的情况下处理多人场景，其功能可以在早期融合阶段轻松与其他模式集成，这为进一步提升性能提供了巨大的设计空间。在四个具有挑战性的数据集上，PoseC3D在单独用于Keletons和与RGB模式结合使用时，持续获得优异的性能。



## 2.复现精度
在UCF-101数据集上spilt1的测试效果如下表。

| NetWork | epochs | opt | image_size | batch_size | dataset | top1 acc |
| --- | --- | --- | --- | --- | --- | --- |
| PoseC3D | 12 | SGD | 56x56 | 16 | UCF-101 | 87.05% |

## 3.数据集
UCF-101以及预训练模型下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/140593](https://aistudio.baidu.com/aistudio/datasetdetail/140593)



## 4.环境依赖
PaddlePaddle == 2.2.2
## 5.快速开始
### 训练：
```shell
cd PaddlePoseC3D
nohup python -u train.py --dataset_root ucf101.pkl --pretrained res3d_k400.pdparams --max_epochs 12 --batch_size 16  --log_iters 100 > train.log &
tail -f train.log
```
dataset_root: 训练集路径

pretrained: 预训练模型路径

max_epochs: 最大epoch数量

batch_size: 批次大小

### 测试：

使用最优模型进行评估.

最优模型下载地址：

链接: https://pan.baidu.com/s/1J9_X_CNkXQbhBhj-xHHBDw 

提取码: uq9m 


```shell
python -u test.py --dataset_root ucf101.pkl --pretrained best_model/model.pdparams
```

dataset_root: 训练集路径

pretrained: 预训练模型路径

测试结果

```shell
3783 videos remain after valid thresholding
W0423 20:29:01.821447 17086 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0423 20:29:01.826694 17086 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Loading pretrained model from output/best_model/model.pdparams
There are 217/217 variables loaded into Recognizer3D.
[                                                  ] 0/3783, elapsed: 0s, ETA:/home/aistudio/PaddlePoseC3D/datasets/pipelines/transforms.py:1467: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  results['frame_inds'] = inds.astype(np.int)
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 3783/3783, 0.4 task/s, elapsed: 9310s, ETA:     0s
Evaluating top_k_accuracy ...

top1_acc	0.8705
top5_acc	0.9635

Evaluating mean_class_accuracy ...

mean_acc	0.8693
top1_acc: 0.8705
top5_acc: 0.9635
mean_class_accuracy: 0.8693

```

### 单张图片预测


```
python predict.py --input_file test_tipc/data/predict_example.pkl --pretrained ../posec3d_output/best_model/model.pdparams 
```
参数说明: 

input_file: 输入文件，按照ucf-101.pkl格式。可以使用test_tipc/data中的predict_example.pkl数据进行测试。

pretrained: 训练好的模型


```
/home/aistudio/PaddlePoseC3D/datasets/pipelines/transforms.py:1467: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  results['frame_inds'] = inds.astype(np.int)
W0423 23:38:54.291606 32315 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0423 23:38:54.296748 32315 device_context.cc:465] device: 0, cuDNN Version: 7.6.
Loading pretrained model from ../posec3d_output/best_model/model.pdparams
There are 217/217 variables loaded into Recognizer3D.
File v_ApplyEyeMakeup_g01_c01 is class 0
File v_ApplyEyeMakeup_g01_c02 is class 0
File v_ApplyEyeMakeup_g01_c03 is class 0
```

### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py --model_path best_model.pdparams --save_dir ./output/
```

参数说明：

model_path: 模型路径

save_dir: 输出图片保存路径

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python infer.py
--use_gpu=False --enable_mkldnn=False --cpu_threads=2 --model_file=output/model.pdmodel --batch_size=2 --input_file=test_tipc/data/predict_example.pkl --enable_benchmark=False --precision=fp32 --params_file=output/model.pdiparams
```

参数说明:

use_gpu:是否使用GPU

enable_mkldnn:是否使用mkldnn

cpu_threads: cpu线程数
 
model_file: 模型路径

batch_size: 批次大小

input_file: 输入文件路径

enable_benchmark: 是否开启benchmark

precision: 运算精度

params_file: 模型权重文件，由export_model.py脚本导出。


### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh test_tipc/configs/posec3d/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh test_tipc/configs/posec3d/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>


## 6.代码结构与详细说明
```shell
PaddlePoseC3D
├── README.md # 使用说明
├── datasets # 数据集包
│   ├── __init__.py
│   ├── base.py #数据集基类
│   ├── file_client.py # 文件处理类
│   ├── pipelines
│   │   └── transforms.py # 数据增强类
│   ├── pose_dataset.py # 数据集类
│   ├── dataset_wrappers.py # 数据集类
│   └── utils.py #数据集工具类
├── models
│   ├── __init__.py
│   ├── base.py # 模型基类
│   ├── resnet3d.py # backbone
│   ├── resnet3d_slowfast.py # backbone
│   └── resnet3d_slowonly.py # backbone
│   ├── i3d_head.py # c3d模型头部实现
│   └── recognizer3d.py # 识别模型框架
├── progress_bar.py #进度条工具
├── test.py # 评估程序
├── test_tipc # TIPC脚本
│   ├── README.md
│   ├── common_func.sh # 通用脚本程序
│   ├── configs
│   │   └── posec3d
│   │       └── train_infer_python.txt # 单机单卡配置
│   ├── data
│   │   ├── example.npy # 推理用样例数据
│   │   └── mini_ucf.zip # 训练用小规模数据集
│   ├── output
│   ├── prepare.sh # 数据准备脚本
│   └── test_train_inference_python.sh # 训练推理测试脚本
├── timer.py # 时间工具类
├── train.log # 训练日志
├── test.log # 测试日志
├── train.py # 训练脚本
└── utils.py # 训练工具包
```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| PoseC3D |
|框架版本| PaddlePaddle==2.2.2|
|应用场景| 骨骼识别 |
