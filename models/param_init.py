import warnings
import math
import paddle


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with paddle.no_grad():
        initializer = paddle.nn.initializer.Normal(mean=0, std=std)
        initializer(tensor)
        # return tensor.normal_(0, std)


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with paddle.no_grad():
        initializer = paddle.nn.initializer.Uniform(-a, a)
        initializer(tensor)


def constant_(tensor, value=0.0):
    initializer = paddle.nn.initializer.Constant(value=value)
    initializer(tensor)


def orthogonal_(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flattened = paddle.zeros([rows, cols])
    initializer = paddle.nn.initializer.Normal(0, 1)
    initializer(flattened)
    # flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = paddle.linalg.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = paddle.diag(r, 0)
    ph = d.sign()
    q *= ph

    if rows < cols:
        q = q.T

    with paddle.no_grad():
        tensor = paddle.reshape(tensor, q.shape)
        initializer = paddle.nn.initializer.Assign(q)
        initializer(tensor)
        tensor = tensor * gain
    return tensor


def _no_grad_normal_(tensor, mean, std):
    with paddle.no_grad():
        initializer = paddle.nn.initializer.Normal(mean, std)
        initializer(tensor)


def xavier_normal_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std)