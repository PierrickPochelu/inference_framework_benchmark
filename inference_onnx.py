import os
import gc
import glob
import numpy as np
import logging
from PIL import Image
import onnx
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table
from smart_allocation.schedule_gpus.abstract_inference_engine import AbstractRuntimeEngine

def batch_generator(data,batch_size):
    for i in range(0,len(data),batch_size):
        batch_data=data[i:i+batch_size]
        yield i,batch_data
    return



class ONNXInferenceEngine(AbstractRuntimeEngine):
    def __init__(self, path, config):
        if not os.path.exists(path):
            raise ValueError(f"ERROR in ONNXInferenceEngine: \n  --->  {path} does not exist")
        

        
        self.session=None
        
        #Default values
        config["enable_caching"] = True
        config["enable_trt_backend"]=False
        config["enable_optimization"]=True
        config["type"] = "FP32"
        
        
        self.gpuid=config["gpuid"]
        self.batch_size=config["batch_size"]
        self.nb_classes=1000

        enable_trt_backend=config["enable_trt_backend"]
        type=config["type"]
        enable_caching=config["enable_caching"]
        enable_optimization=config["enable_optimization"]

        if type=="FP16":
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"  # Enable FP16 precision
        elif type=="INT8":
            os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"  # Enable INT8 precision
        elif type=="FP32":
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"
            os.environ["ORT_TENSORRT_INT8_ENABLE"] = "0"
        else:
            raise ValueError("Type not understood")
        
        if enable_caching:
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
        else:
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "0"
        
        if self.gpuid!=-1 and enable_trt_backend:
            exec_providers=["CUDAExecutionProvider"]
            #exec_providers="TensorrtExecutionProvider"
        elif self.gpuid!=-1 and not enable_trt_backend:
            exec_providers=["CUDAExecutionProvider"]
        elif self.gpuid==-1:
            exec_providers=['CPUExecutionProvider']
        else:
            raise ValueError("Execution provider not understood")


        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 3

        if enable_optimization:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.session = onnxruntime.InferenceSession(path,
                                               sess_options=sess_options,
                                               providers=exec_providers)

        self.input_name=self.session.get_inputs()[0].name
        input_shape=self.session.get_inputs()[0].shape
        self.output_name=self.session.get_outputs()[0].name
        output_shape=self.session.get_outputs()[0].shape

        dumb_data=np.random.uniform(0,1,(self.batch_size,224,224,3)).astype(np.float32)
        self.predict(dumb_data)
    """    
    def predict(self,x):
        def i
        output = self.session.run([self.output_name], 
        {self.input_name:x},run_options=None)
        return output
    """
    def predict(self,x):
        #fast init
        pred = np.zeros((x.shape[0], self.nb_classes), dtype=np.float32)
        gen = batch_generator(x, self.batch_size)
        
        for i, batch_data in gen:
            n=len(batch_data)
            out = self.session.run(
                [self.output_name],
                input_feed={self.input_name:batch_data},
                run_options=None
            )[0]
            pred[i:i + n]=out[:n]
        return pred

    def __del__(self):
        if self.session is not None:
            del self.session
        gc.collect()

import time
from pierrick_tools.benchmark import BENCH
import sys
if __name__ == "__main__":
    """
    model_path="./models_lib/ONNX/EfficientNetB0.onnx"
    config = {}
    config["gpuid"] = 0
    BENCH(ONNXInferenceEngine,model_path,config,[1])
    """
    for g in [0]:
        config = {}
        config["gpuid"] = g
        print(config)
        #for model_path in ["./models_lib/ONNX/VGG19.onnx"]:
        for model_path in [["./models_lib/ONNX/DenseNet201.onnx", "./models_lib/ONNX/ResNet50.onnx" , "./models_lib/ONNX/EfficientNetB0.onnx", "./models_lib/ONNX/VGG19.onnx"][int(sys.argv[1])]]:
            print(model_path)
            BENCH(ONNXInferenceEngine,model_path,config,[128])

