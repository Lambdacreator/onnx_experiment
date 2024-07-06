
import numpy as np
import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper
from .custom_backend import operator_map
from concurrent.futures import ThreadPoolExecutor

def relu(x):
    return np.maximum(0, x)

def add(a, b):
    return np.add(a, b)

def matmul(a, b):
    return np.dot(a, b)

def reshape(tensor, shape):
    return np.reshape(tensor, shape)

def conv_worker(x_slice, w, b, i, j, out_channels):
    result = np.zeros((x_slice.shape[0], out_channels), dtype=np.float32)
    for k in range(out_channels):
        result[:, k] = np.sum(x_slice * w[k, :, :, :], axis=(1, 2, 3))
    if b is not None:
        result += b.reshape(1, -1, 1, 1)[:, :, 0, 0]
    return i, j, result

def conv(x, w, b=None, strides=(1, 1), pads=(0, 0, 0, 0)):
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = w.shape
    stride_height, stride_width = strides
    pad_height_begin, pad_width_begin, pad_height_end, pad_width_end = pads

    out_height = (in_height + pad_height_begin + pad_height_end - kernel_height) // stride_height + 1
    out_width = (in_width + pad_width_begin + pad_width_end - kernel_width) // stride_width + 1

    y = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_height_begin, pad_height_end), (pad_width_begin, pad_width_end)), mode='constant')
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + kernel_height
                w_start = j * stride_width
                w_end = w_start + kernel_width
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                futures.append(executor.submit(conv_worker, x_slice, w, b, i, j, out_channels))
        
        for future in futures:
            i, j, result = future.result()
            y[:, :, i, j] = result

    return y

def maxpool_worker(x_slice, i, j, in_channels):
    result = np.zeros((x_slice.shape[0], in_channels), dtype=np.float32)
    for c in range(in_channels):
        result[:, c] = np.max(x_slice[:, c, :, :], axis=(1, 2))
    return i, j, result

def maxpool(x, kernel_shape, strides=(1, 1), pads=(0, 0, 0, 0)):
    batch_size, in_channels, in_height, in_width = x.shape
    kernel_height, kernel_width = kernel_shape
    stride_height, stride_width = strides
    pad_height_begin, pad_width_begin, pad_height_end, pad_width_end = pads

    out_height = (in_height + pad_height_begin + pad_height_end - kernel_height) // stride_height + 1
    out_width = (in_width + pad_width_begin + pad_width_end - kernel_width) // stride_width + 1

    y = np.zeros((batch_size, in_channels, out_height, out_width), dtype=np.float32)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad_height_begin, pad_height_end), (pad_width_begin, pad_width_end)), mode='constant')

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * stride_height
                h_end = h_start + kernel_height
                w_start = j * stride_width
                w_end = w_start + kernel_width
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                futures.append(executor.submit(maxpool_worker, x_slice, i, j, in_channels))
        
        for future in futures:
            i, j, result = future.result()
            y[:, :, i, j] = result

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

def reduceprod_worker(x_slice, axes):
    return np.prod(x_slice, axis=axes, keepdims=False)

def reduceprod(x, axis=None, keepdims=False):
    if axis is not None:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(reduceprod_worker, x, ax) for ax in axis]
            results = [future.result() for future in futures]
        for res in results:
            x = res
        return x
    else:
        return np.prod(x, keepdims=keepdims)

def unsqueeze(x, axes):
    for axis in sorted(axes):
        x = np.expand_dims(x, axis)
    return x

def concat_worker(tensor, axis):
    return np.atleast_1d(tensor)

def concat(*tensors, axis=0):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(concat_worker, tensor, axis) for tensor in tensors]
        tensors = [future.result() for future in futures]
    return np.concatenate(tensors, axis=axis)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gemm_worker(a, b, alpha, beta, trans_a, trans_b):
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    return alpha * np.dot(a, b)

def gemm(a, b, c=None, alpha=1.0, beta=1.0, trans_a=False, trans_b=False):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(gemm_worker, a, b, alpha, beta, trans_a, trans_b)
        y = future.result()
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


def execute_node(node, inputs):
    input_tensors = [inputs[input_name] for input_name in node.input]
    
    if node.op_type == 'Reshape':
        shape_tensor = input_tensors[1]
        if isinstance(shape_tensor, list):
            shape_tensor = np.array(shape_tensor)
        output_tensors = operator_map[node.op_type](input_tensors[0], shape_tensor)
    elif node.op_type == 'Conv':
        attrs = {attr.name: attr.ints for attr in node.attribute}
        strides = tuple(attrs.get('strides', [1, 1]))
        pads = tuple(attrs.get('pads', [0, 0, 0, 0]))
        b = input_tensors[2] if len(input_tensors) > 2 else None
        output_tensors = operator_map[node.op_type](input_tensors[0], input_tensors[1], b, strides, pads)
    elif node.op_type == 'MaxPool':
        attrs = {attr.name: attr.ints for attr in node.attribute}
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = tuple(attrs.get('strides', [1, 1]))
        pads = tuple(attrs.get('pads', [0, 0, 0, 0]))
        output_tensors = operator_map[node.op_type](input_tensors[0], kernel_shape, strides, pads)
    elif node.op_type == 'Transpose':
        perm = node.attribute[0].ints if node.attribute else []
        output_tensors = operator_map[node.op_type](input_tensors[0], perm)
    elif node.op_type == 'Gather':
        indices = input_tensors[1]
        axis = node.attribute[0].i if node.attribute else 0
        output_tensors = operator_map[node.op_type](input_tensors[0], indices, axis)
    elif node.op_type == 'Cast':
        to = node.attribute[0].i  # The data type to cast to (ONNX data type enum)
        output_tensors = operator_map[node.op_type](input_tensors[0], to)
    elif node.op_type == 'ReduceProd':
        axis = node.attribute[0].ints if node.attribute else None
        if axis is not None:
            axis = list(axis)  # Convert to list if it's a RepeatedScalarContainer
        keepdims = node.attribute[1].i if len(node.attribute) > 1 else False
        output_tensors = operator_map[node.op_type](input_tensors[0], axis, keepdims)
    elif node.op_type == 'Unsqueeze':
        axes = node.attribute[0].ints if node.attribute else []
        output_tensors = operator_map[node.op_type](input_tensors[0], axes)
    elif node.op_type == 'Concat':
        axis = node.attribute[0].i if node.attribute else 0
        output_tensors = operator_map[node.op_type](*input_tensors, axis=axis)
    elif node.op_type == 'Sigmoid':
        output_tensors = operator_map[node.op_type](input_tensors[0])
    elif node.op_type == 'Gemm':
        attrs = {attr.name: attr for attr in node.attribute}
        alpha = attrs['alpha'].f if 'alpha' in attrs else 1.0
        beta = attrs['beta'].f if 'beta' in attrs else 1.0
        trans_a = attrs['transA'].i if 'transA' in attrs else 0
        trans_b = attrs['transB'].i if 'transB' in attrs else 0
        c = input_tensors[2] if len(input_tensors) > 2 else None
        output_tensors = operator_map[node.op_type](input_tensors[0], input_tensors[1], c, alpha, beta, trans_a, trans_b)
    else:
        output_tensors = operator_map[node.op_type](*input_tensors)

    if not isinstance(output_tensors, tuple):
        output_tensors = (output_tensors,)

    for output_name, output_tensor in zip(node.output, output_tensors):
        inputs[output_name] = output_tensor

def execute_graph(graph, input_data):
    inputs = {}

    # Use provided input data
    for input_tensor in graph.input:
        input_name = input_tensor.name
        inputs[input_name] = input_data

    # Initialize tensors for constants (initializers)
    for initializer in graph.initializer:
        tensor = numpy_helper.to_array(initializer)
        inputs[initializer.name] = tensor

    # Execute nodes
    for node in graph.node:
        execute_node(node, inputs)

    # Extract outputs
    outputs = {output.name: inputs[output.name] for output in graph.output}
    return outputs