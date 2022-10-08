from smart_allocation.schedule_gpus.abstract_inference_engine import AbstractRuntimeEngine
import os
import tensorrt as trt
import numpy as np
import gc
from smart_allocation.schedule_gpus.knowledge import Option
import pycuda.driver as cuda
import numpy as np

def batch_generator(data,batch_size):
    for i in range(0,len(data),batch_size):
        batch_data=data[i:i+batch_size]
        yield i,batch_data
    return


def build_engine(onnx_path, shape = [1,224,224,3]):

    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file.
       shape : Shape of the input of the ONNX file.
   """
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = (256 << 20)
        builder.max_batch_size=shape[0]
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine



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
    h_input_1 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
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
        #context.profiler = trt.Profiler()
        context.execute(batch_size=batch_size, bindings=[int(d_input_1), int(d_output)])
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        #out = h_output.reshape((batch_size,-1, height, width))
        out=h_output
        return out



class TrtInferenceEngine(AbstractRuntimeEngine):
    def __init__(self, path, config):
        if not os.path.exists(path):
            raise ValueError(f"ERROR in TensorflowInferenceEngine: \n  --->  {path} does not exist")
        assert(config["gpuid"]!=-1)
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpuid"])
        self.batch_size=config["batch_size"]
        self.nb_class = 1000
        self.data_height, self.data_width, self.channels = 224, 224, 3

        input_shape=[self.batch_size,self.data_height, self.data_width, self.channels]
        self.engine=build_engine(path, shape = input_shape)
        self.h_input_1, self.d_input_1, self.h_output, self.d_output, self.stream=\
            allocate_buffers(self.engine, self.batch_size, trt.float32)

        #warmup
        dumb_data=np.random.uniform(0,1,input_shape).astype(np.float32)
        _=self.predict(dumb_data)
    def predict(self, x):
        pred = np.zeros((len(x) , self.nb_class)).astype(np.float32)
        gen = batch_generator(x, self.batch_size)
        for i, batch_data in gen:
            o=do_inference(self.engine,
                           batch_data,
                           self.h_input_1,
                           self.d_input_1,
                           self.h_output,
                           self.d_output,
                           self.stream,
                           self.batch_size, self.data_height, self.data_width)
            pred[i:(i + self.batch_size)] = o.reshape((len(batch_data),self.nb_class))
        return pred

    def __del__(self):
        pass
        #del self.engine
        #gc.collect()#help the garbage collector


from pierrick_tools.benchmark import BENCH
import sys
if __name__ == "__main__":
    for g in [1]:
        config = {}
        config["gpuid"] = g
        print(config)
        for path in ["./models_lib/ONNX/DenseNet201.onnx",
                      "./models_lib/ONNX/ResNet50.onnx" ,
                      "./models_lib/ONNX/EfficientNetB0.onnx",
                      "./models_lib/ONNX/VGG19.onnx"]:
            print(path)
            BENCH(TrtInferenceEngine,path,config,[64])
