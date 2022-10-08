#from pierrick_tools.inference_openvino import OpenVINOInferenceEngine
#from pierrick_tools.inference_trt import TrtInferenceEngine
#from pierrick_tools.inference_onnx import ONNXInferenceEngine
#from pierrick_tools.inference_tf import TensorflowInferenceEngine

import numpy as np
import time
import os


models_lib_path="/data/appli_PITSI/users/pochelu/project/inference_system/models_lib/"
x = np.random.uniform(0, 1, (4096*2, 224, 224, 3)).astype(np.float32)
x2 = np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
GPU_ID=0



def BENCH(model_ptr, path, config,POSSIBLE_BATCH_SIZE=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    #print(model_ptr)
    print(path)
    
    
    throughputs=[]
    for b in POSSIBLE_BATCH_SIZE:
        config["batch_size"]=b

        st=time.time()
        model=model_ptr(path,config)
        build_time = time.time() - st
        print("Build time: ",build_time)

        st=time.time()
        y=model.predict(x)
        enlapsted_time = time.time() - st
        through=round(len(x) / enlapsted_time)
        print("throughput: ", through)
        throughputs.append(through)
  
        """
        #del model
        if config["batch_size"]==1:
            pings = []
            for i in range(10):
                st = time.time()
                model.predict(x2)
                enlapsted_time = time.time() - st
                pings.append(enlapsted_time)
            mean_latency=round(np.mean(pings),6)
            std_latency=round(np.std(pings),6)
            print("Mean latency: ",mean_latency)
            print("Std latency: ", std_latency)
        """
        if config["batch_size"]==1:
            pings = []
            for i in range(1000):
                st = time.time()
                model.predict(x2)
                enlapsted_time = time.time() - st
                pings.append(enlapsted_time)
            print("PERCENTIL 50:", np.percentile(pings,50))
            print("PERCENTIL 95: ", np.percentile(pings,95))
            print("PERCENTIL 99: ", np.percentile(pings,99))


    print("Throughputs: ", throughputs)

def get_subdir(P):
    l=[os.path.join(P,fname) for fname in os.listdir(P) ]
    #return l[::-1]
    #l=["ResNet50.pb","DenseNet201.pb","VGG19.pb","EfficientNetB0.pb"]
    #l = ["EfficientNetB0.pb"]
    #return [os.path.join(P,li) for li in l]
    #return [os.path.join(P,"ResNet50.onnx")]
    return l




def config_openvino():
    config = {}
    config["batch_size"] = 32
    config["gpuid"] = -1
    return config

def config_ONNX():
    def configs(device,type):
        config = {}
        config["batch_size"] = 32
        config["gpuid"] = device
        config["enable_caching"] = True
        config["enable_trt_backend"] = False
        config["enable_optimization"] = True
        config["type"] = type
        return config

    configs_cpu=configs(-1, "INT8") + configs(-1, "FP32") + configs(-1, "FP16")
    configs_gpu = configs(GPU_ID, "INT8") + configs(GPU_ID, "FP32") + configs(GPU_ID, "FP16")
    res=configs_cpu+configs_gpu
    return res

def config_tensorRT():
    config={}
    config["batch_size"] = 32
    config["gpuid"] = GPU_ID
    return config

def config_TF():
    """
    config_cpu = {}
    config_cpu["batch_size"] = 32
    config_cpu["gpuid"] = -1
    configs_cpu=generate_batch_size_values(config_cpu)
    """
    config_gpu = {}
    config_gpu["batch_size"] = 32
    config_gpu["gpuid"] = GPU_ID
    return config_gpu




