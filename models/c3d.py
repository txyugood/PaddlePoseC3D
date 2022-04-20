import paddle
import paddle.nn as nn

def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)

class ConvBNLayer(nn.Layer):
    """A conv block that bundles conv/norm/activation layers.

        This block simplifies the usage of convolution layers, which are commonly
        used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
        It is based upon three build methods: `build_conv_layer()`,
        `build_norm_layer()` and `build_activation_layer()`.

        Besides, we add some additional features in this module.
        1. Automatically set `bias` of the conv layer.
        2. Spectral norm is supported.
        3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
        supports zero and circular padding, and we add "reflect" padding mode.

        Args:
            in_channels (int): Number of channels in the input feature map.
                Same as that in ``nn._ConvNd``.
            out_channels (int): Number of channels produced by the convolution.
                Same as that in ``nn._ConvNd``.
            kernel_size (int | tuple[int]): Size of the convolving kernel.
                Same as that in ``nn._ConvNd``.
            stride (int | tuple[int]): Stride of the convolution.
                Same as that in ``nn._ConvNd``.
            padding (int | tuple[int]): Zero-padding added to both sides of
                the input. Same as that in ``nn._ConvNd``.
            dilation (int | tuple[int]): Spacing between kernel elements.
                Same as that in ``nn._ConvNd``.
            groups (int): Number of blocked connections from input channels to
                output channels. Same as that in ``nn._ConvNd``.
        """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        act=None,
    ):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=None)

        # self._batch_norm = SyncBatchNorm(out_channels, momentum=0.1)
        self.act = act
        if act is not None:
            self._act_op = nn.ReLU()

    def forward(self, inputs):
        y = self._conv(inputs)
        # y = self._batch_norm(y)
        if self.act is not None:
            y = self._act_op(y)

        return y



class C3D(nn.Layer):
    """C3D backbone.

    Args:
        pretrained (str | None): Name of pretrained model.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config dict for convolution layer.
            If set to None, it uses ``dict(type='Conv3d')`` to construct
            layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. required keys are
            ``type``, Default: None.
        act_cfg (dict | None): Config dict for activation layer. If set to
            None, it uses ``dict(type='ReLU')`` to construct layers.
            Default: None.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation of fc layers. Default: 0.01.
    """

    def __init__(self,
                 pretrained=None,
                 dropout_ratio=0.5,
                 init_std=0.005):
        super().__init__()
        self.pretrained = pretrained
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        c3d_conv_param = dict(
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            act='relu')

        self.conv1a = ConvBNLayer(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3D(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvBNLayer(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3D(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = ConvBNLayer(128, 256, **c3d_conv_param)
        self.conv3b = ConvBNLayer(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3D(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = ConvBNLayer(256, 512, **c3d_conv_param)
        self.conv4b = ConvBNLayer(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3D(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = ConvBNLayer(512, 512, **c3d_conv_param)
        self.conv5b = ConvBNLayer(512, 512, **c3d_conv_param)
        self.pool5 = nn.MaxPool3D(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_ratio)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass



    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1a(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.pool5(x)

        x = x.flatten(start_axis=1)
        x = self.relu6(self.fc6(x))
        x = self.dropout(x)
        x = self.relu7(self.fc7(x))

        return x
