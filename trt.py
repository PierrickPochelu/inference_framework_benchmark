import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,224,224,3]):

    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file.
       shape : Shape of the input of the ONNX file.
   """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        builder.max_batch_size=shape[0]
        engine = builder.build_engine(network, config)
        #engine.max_batch_size=shape[0]
        return engine

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit

def allocate_buffers(engine, batch_size, data_type=trt.float32):

    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(1 * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(1 * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream

def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
       engine : Path to the TensorRT engine
       pics_1 : Input images to the model.
       h_input_1: Input in the host
       d_input_1: Input in the device
       h_output_1: Output in the host
       d_output_1: Output in the device
       stream: CUDA stream
       batch_size : Batch size for execution time
       height: Height of the output image
       width: Width of the output image

    Output:
       The list of output images

    """
    load_images_to_buffer(pics_1, h_input_1)
    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.
        context.profiler = trt.Profiler()
        context.execute(batch_size=batch_size, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        #out = h_output.reshape((batch_size,-1, height, width))
        out=h_output
        return out

# https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
path="../models_lib/ONNX/ResNet50.onnx"

import numpy as np
x=np.random.uniform(0,1,(32,224,224,3)).astype(np.float32)

batch_size=32

engine=build_engine(path, shape = [batch_size,224,224,3])
h_input_1, d_input_1, h_output, d_output, stream=allocate_buffers(engine, batch_size, trt.float32)

for i in range(10):
    out=do_inference(engine, x, h_input_1, d_input_1, h_output, d_output, stream, batch_size, 224, 224)
    out=out.reshape((len(x),1000))
    print(np.sum(out))
    print(out)
print("ok")