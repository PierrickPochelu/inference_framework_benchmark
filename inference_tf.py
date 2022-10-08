import tensorflow.compat.v1 as tf
from pierrick_tools import tensorflow_graph_parser
tf.disable_v2_behavior()
import numpy as np
import os
import sys


def identity(x):
    return x

def batch_generator(data,batch_size,prepro_f):
    for i in range(0,len(data),batch_size):
        batch_data=data[i:i+batch_size]
        batch_data=prepro_f(batch_data)
        yield i,batch_data
    return

def load_tf_model(sess, INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME, PB_PATH, GRAPH_NAME,
                  config):
    """
    load and warm up a model in memory for efficient deployment
    :param INPUT_TENSOR_NAME: e.g. GRAPH_NAME+"/input_1:0"
    :param OUTPUT_TENSOR_NAME: e.g. GRAPH_NAME+"/image_predictions/Softmax:0"
    :param PB_PATH: file path
    :param GRAPH_NAME: unique name, for example "g1", "g2", ...
    :param config:
    :return: input output tensors
    """
    batch_size=config["batch_size"]

    imgsize=config.get("imgsize",224)

    # Read graph
    with tf.gfile.GFile(PB_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name=GRAPH_NAME)

    # Extract input and output tensors
    newgraph = tf.get_default_graph()
    input_tensor = newgraph.get_tensor_by_name(INPUT_TENSOR_NAME)
    output_tensor = newgraph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

    # warm up the model with dumb data
    dumb_batch_shape = [batch_size] + input_tensor.get_shape().as_list()[1:]
    if dumb_batch_shape[1] is None: #This is a patch
        dumb_batch_shape[1]=imgsize
        dumb_batch_shape[2]=imgsize

    dumb_batch = np.random.uniform(0, 1, dumb_batch_shape)
    _ = sess.run(output_tensor, feed_dict={input_tensor: dumb_batch})

    return input_tensor, output_tensor

def get_input_output_tensor_name(pb_path,sess,graph_name="g"):
    # CATCH INPUT/OUTPUT TENSORS
    input_regex_tensor_name = "^"+graph_name+"/x:|^x:|input:"
    output_regex_tensor_name = ".*/Softmax:0"
    input_tensor_name, output_tensor_name = tensorflow_graph_parser.explore_tf_file(
        pb_path,
        input_regex_tensor_name,
        output_regex_tensor_name,
        sess=sess,
        GRAPH_NAME=graph_name,
        verbose=False)
    return input_tensor_name, output_tensor_name


class TensorflowInferenceEngine:
    def __init__(self, path, config):
        if not os.path.exists(path):
            raise ValueError(f"ERROR in TensorflowInferenceEngine: \n  --->  {path} does not exist")
        #tf.keras.backend.clear_session()#MANDATORY to be sure no other graph is in memory
        self.allow_growth=True
        self.batch_size=config["batch_size"]
        self.gpuid=config["gpuid"]
        self.graph_name="g"+str(np.random.uniform(0,999999,(1,))[0].astype(np.int32))
        self.is_XLA=config["XLA"] #TRUE FOR MAXIMUM PERFORMANCE BUT MULTIPLY BY 6 THE INITIALIZATION TIME

        if "prepro_f" in config:
            self.prepro_f=config["prepro_f"]
        else:
            self.prepro_f=identity

        # Use XLA
        if self.gpuid!=-1 and self.is_XLA:
            # Check XLA is true for better performance (but slower init. speed).
            assert(os.environ["TF_XLA_FLAGS"]=="--tf_xla_auto_jit=2")
            assert(os.environ["XLA_FLAGS"]=="--xla_gpu_cuda_data_dir="+os.environ["CUDA_HOME"])

        # create TF session and bind device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        nbgpu = 1 if self.gpuid != -1 else 0
        tfconf=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,device_count={'CPU': 1, 'GPU': nbgpu})
        if nbgpu>0:
            tfconf.gpu_options.allow_growth=self.allow_growth
        self.sess = tf.Session(config=tfconf)

        # This portion of code is risky
        inout = get_input_output_tensor_name(path,self.sess,self.graph_name)
        self.input_tensor_name=inout[0]
        self.output_tensor_name=inout[1]
        self.FAKE_PREDICT_MODE=False#usefull to measure Python overhead

        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]=str(self.allow_growth)


        #try:
        self.input_tensor, self.output_tensor=load_tf_model(
            self.sess,
            self.input_tensor_name,
            self.output_tensor_name,
            path,
            self.graph_name,
            config
        )


    def predict(self, x):
        #fast init
        pred = np.zeros((x.shape[0], self.output_tensor.shape[1]), dtype=np.float32)
        gen = batch_generator(x, self.batch_size,self.prepro_f)

        #prediction
        if self.FAKE_PREDICT_MODE:
            return pred #usefull to measure Python overhead

        for i, batch_data in gen:
            pred[i:i + self.batch_size] = self.sess.run(
                self.output_tensor,
                feed_dict={self.input_tensor: batch_data})
        return pred

    def _free_gpu_memory(self):
        #tf.keras.backend.clear_session()
        if self.sess is not None:
            self.sess.close()
            self.sess = None
        self.input_tensor=None
        self.output_tensor=None


    def is_ok(self):
        sess_ok=self.sess is not None
        input_ok=self.input_tensor is not None
        output_ok=self.output_tensor is not None
        return sess_ok and input_ok and output_ok

    def __del__(self):
        self._free_gpu_memory()
        #gc.collect()#help the garbage collector


if __name__ == "__main__":
    from pierrick_tools.benchmark import BENCH
    for g in [0]:
        config = {}
        config["gpuid"] = g
        config["XLA"] = True
        print(config)
        for model_path in [["./models_lib/TF_PB/DenseNet201.pb", "./models_lib/TF_PB/ResNet50.pb" , "./models_lib/TF_PB/EfficientNetB0.pb", "./models_lib/TF_PB/VGG19.pb"][int(sys.argv[1])]]:
            print(model_path)
            BENCH(TensorflowInferenceEngine,model_path,config,[128])
