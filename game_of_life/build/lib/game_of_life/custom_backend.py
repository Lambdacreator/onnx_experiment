import numpy as np

def relu(x):
    return np.maximum(0, x)

def add(a, b):
    return np.add(a, b)

def matmul(a, b):
    return np.dot(a, b)

def reshape(tensor, shape):
    return np.reshape(tensor, shape)

def conv(x, w, b=None, strides=(1, 1), pads=(0, 0, 0, 0)):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    stride_height, stride_width = strides
    pad_height_begin, pad_width_begin, pad_height_end, pad_width_end = pads

    out_height = (in_height + pad_height_begin + pad_height_end - kernel_height) // stride_height + 1
    out_width = (in_width + pad_width_begin + pad_width_end - kernel_width) // stride_width + 1

    y = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_height_begin, pad_height_end), (pad_width_begin, pad_width_end)), mode='constant')
    
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride_height
            h_end = h_start + kernel_height
            w_start = j * stride_width
            w_end = w_start + kernel_width
            x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
            for k in range(out_channels):
                y[:, k, i, j] = np.sum(x_slice * w[k, :, :, :], axis=(1, 2, 3))
                
    if b is not None:
        y += b.reshape(1, -1, 1, 1)
        
    return y

def maxpool(x, kernel_shape, strides=(1, 1), pads=(0, 0, 0, 0)):
    batch_size, in_channels, in_height, in_width = x.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = strides
    pad_height_begin, pad_width_begin, pad_height_end, pad_width_end = pads

    out_height = (in_height + pad_height_begin + pad_height_end - kernel_height) // stride_height + 1
    out_width = (in_width + pad_width_begin + pad_width_end - kernel_width) // stride_width + 1

    y = np.zeros((batch_size, in_channels, out_height, out_width), dtype=np.float32)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_height_begin, pad_height_end), (pad_width_begin, pad_width_end)), mode='constant')

    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride_height
            h_end = h_start + kernel_height
            w_start = j * stride_width
            w_end = w_start + kernel_width
            x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
            y[:, :, i, j] = np.max(x_slice, axis=(2, 3))

    return y

def transpose(x, perm):
    return np.transpose(x, perm)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def shape(x):
    return np.array(x.shape)

def gather(x, indices, axis=0):
    return np.take(x, indices, axis=axis)

def cast(x, to):
    onnx_to_numpy_dtype = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        8: np.str_,
        9: np.bool_,
        10: np.float16,
        11: np.double,
        12: np.uint32,
        13: np.uint64,
        14: np.complex64,
        15: np.complex128
    }
    return x.astype(onnx_to_numpy_dtype[to])

def reduceprod(x, axis=None, keepdims=False):
    if axis is not None:
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                x = np.prod(x, axis=ax, keepdims=keepdims)
            return x
        else:
            return np.prod(x, axis=axis, keepdims=keepdims)
    else:
        return np.prod(x, keepdims=keepdims)

def unsqueeze(x, axes):
    for axis in sorted(axes):
        x = np.expand_dims(x, axis)
    return x

def concat(*tensors, axis=0):
    tensors = [np.atleast_1d(tensor) for tensor in tensors]  # Ensure all tensors are at least 1-dimensional
    return np.concatenate(tensors, axis=axis)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gemm(a, b, c=None, alpha=1.0, beta=1.0, trans_a=False, trans_b=False):
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    y = alpha * np.dot(a, b)
    if c is not None:
        y += beta * c
    return y

# Map ONNX operator names to functions
operator_map = {
    'Relu': relu,
    'Add': add,
    'MatMul': matmul,
    'Reshape': reshape,
    'Conv': conv,
    'MaxPool': maxpool,
    'Transpose': transpose,
    'Softmax': softmax,
    'Shape': shape,
    'Gather': gather,
    'Cast': cast,
    'ReduceProd': reduceprod,
    'Unsqueeze': unsqueeze,
    'Concat': concat,
    'Sigmoid': sigmoid,
    'Gemm': gemm,
    # Add more operators
}
