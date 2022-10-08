from openvino.inference_engine import IECore
import sys
import os
import gc
import time
import numpy as np
import logging as log
from smart_allocation.schedule_gpus.abstract_inference_engine import AbstractRuntimeEngine





def get_path(P):
    model_name=os.path.split(os.path.abspath(P))[-1]
    model_xml = os.path.join( P ,(model_name + ".xml"))
    model_bin = os.path.join( P ,(model_name + ".bin"))
    return model_xml, model_bin

def batch_generator(data, batch_size):
    for i in range(0,len(data),batch_size):
        batch_data=data[i:i+batch_size]
        yield i, batch_data
    return

class OpenVINOInferenceEngine(AbstractRuntimeEngine):
    def __init__(self, path, config):
        if not os.path.exists(path):
            raise ValueError(f"ERROR in OpenVINOInferenceEngine: \n  --->  {path} does not exist")

        model_xml, model_bin=get_path(path)
        self.batch_size=config["batch_size"]
        self.is_nchw=True
        self.nb_classes=1000
        self.FAKE_PREDICT_MODE=False

        # Plugin initialization for specified device and load extensions library if specified
        ie = IECore()
        self.net = ie.read_network(model=model_xml, weights=model_bin)

        device="CPU"

        assert len(self.net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(self.net.outputs) == 1, "Sample supports only single output topologies"


        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = config["batch_size"]

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=self.net, device_name=device)

        # Read and pre-process input images
        n, c, h, w = self.net.inputs[self.input_blob].shape

        #x = np.random.uniform(0,1,(4096,224,224,3))
    def predict(self,x):
        x=x.transpose((0,3,1,2))

        #fast init
        pred = np.zeros((x.shape[0], self.nb_classes), dtype=float)
        gen = batch_generator(x, self.batch_size)

        #prediction
        if self.FAKE_PREDICT_MODE:
            return pred #usefull to measure Python overhead

        for i, batch_data in gen:
            res = self.exec_net.infer(inputs={
                self.input_blob: batch_data
            })
            out = res[self.out_blob]
            maxid = min(len(x), len(out))
            pred[i:i + maxid] = out[:maxid]
        return pred

    def __del__(self):
        self.exec_net=None
        self.net=None
        gc.collect()

from pierrick_tools.benchmark import BENCH
if __name__ == "__main__":
    for g in [-1]:
        config = {}
        config["gpuid"] = g
        print(config)
        for path in ["./models_lib/OpenVINO/DenseNet201", "./models_lib/OpenVINO/ResNet50" , "./models_lib/OpenVINO/VGG19"]:
            print(path)
            BENCH(OpenVINOInferenceEngine,path,config,[1])


