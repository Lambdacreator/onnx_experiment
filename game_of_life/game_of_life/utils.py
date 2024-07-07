import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper

from .custom_backend import operator_map

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


