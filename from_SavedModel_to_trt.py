import os

from tensorflow.python.saved_model import constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants



def from_SavedModelFile_to_trtFile(SavedModel_path, trt_path, arithmetic="FP32"):

    # Conversion Parameters
    if arithmetic=="FP16":
        conversion_params = trt.TrtConversionParams(
            precision_mode=trt.TrtPrecisionMode.FP16)
    else:
        conversion_params = trt.TrtConversionParams(
            precision_mode=trt.TrtPrecisionMode.FP32)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=SavedModel_path,
        conversion_params=conversion_params)

    #from tensorflow.experimental.tensorrt import Converter
    #converter = Converter(input_saved_model_dir=INPUT_SAVED_MODEL_DIR)
    # Converter method used to partition and optimize TensorRT compatible segments
    converter.convert()

    # Optionally, build TensorRT engines before deployment to save time at runtime
    # Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
    """
    def my_input_fn():
        x=np.random.uniform(0,1,(16,224,224,3))
        for i in range(10):
            yield x[i]
    converter.build(input_fn=None)
    """
    # Save the model to the disk
    converter.save(trt_path)

if __name__ == "__main__":
    ROOT = "."
    INPUT_SAVED_MODEL_DIR = ROOT + "/models_lib/TF_SavedModel/"

    arithmetic="FP32"
    force=True

    for fname in os.listdir(INPUT_SAVED_MODEL_DIR):
        SavedModel_path=os.path.join(INPUT_SAVED_MODEL_DIR,fname)

        try:
            #Compute folder according arithmetic
            if arithmetic=="FP16":
                OUTPUT_SAVED_MODEL_DIR = ROOT + "/models_lib/TensorRT_FP16/"
                trt_path = os.path.join(OUTPUT_SAVED_MODEL_DIR, fname)
            else: #FP32
                OUTPUT_SAVED_MODEL_DIR = ROOT + "/models_lib/TensorRT/"
                trt_path = os.path.join(OUTPUT_SAVED_MODEL_DIR, fname)

            if (not os.path.exists(trt_path)) or force:
                print(f"{trt_path}")
                from_SavedModelFile_to_trtFile(SavedModel_path, trt_path, arithmetic)
            else:
                print(f"alread exists: {trt_path}")
        except Exception as e:
            print(e)
