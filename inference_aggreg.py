import gc
from smart_allocation.schedule_gpus.abstract_inference_engine import AbstractRuntimeEngine
from pierrick_tools.inference_onnx import ONNXInferenceEngine
from pierrick_tools.inference_openvino import OpenVINOInferenceEngine


def from_onnx_path_to_openvino_path(path):
    path=path.replace("ONNX", "OpenVINO")
    path=path.replace(".onnx", "")
    return path

class AggregInferenceEngine(AbstractRuntimeEngine):
    def __init__(self, path, config):
        if config["gpuid"] == -1 or config["gpuid"] == "-1":
            path=from_onnx_path_to_openvino_path(path)
            self.system = OpenVINOInferenceEngine(path, config)
        else:
            config["enable_caching"] = True
            config["enable_trt_backend"] = False
            config["enable_optimization"] = True
            config["type"] = "FP32"
            self.system = ONNXInferenceEngine(path, config)

    def predict(self, x):
        self.system.predict(x)

    def __del__(self):
        del self.system


import time
from pierrick_tools.benchmark import BENCH

if __name__ == "__main__":
    model_path = "./models_lib/ONNX/VGG19.onnx"
    config = {}
    config["gpuid"] = 0
    config["batch_size"] = 32

    BENCH(AggregInferenceEngine, model_path, config)
