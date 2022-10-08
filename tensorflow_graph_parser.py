import tensorflow.compat.v1 as tf
import re
import sys

#This script is usefull to parse TF file and take input/output tensor
def _log(t,v):
    if v:
        print(t)

def explore_tf_file(PB_PATH,
                    INPUT_TENSOR_NAME_REGEX,
                    OUTPUT_TENSOR_NAME_REGEX,
                    GRAPH_NAME="g",
                    sess=None,
                    verbose=True):
    if sess is None:
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))


    # Read graph
    with tf.gfile.GFile(PB_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name=GRAPH_NAME)

    # Extract input and output
    newgraph = tf.get_default_graph()
    all_nodes=[]
    for op in newgraph.get_operations():
        try:
            name=op.values()[0].name
            all_nodes.append(name)
        except:
            _log(f"Error with the node {op}",verbose)

    INPUT_TENSOR_NAME=None
    OUTPUT_TENSOR_NAME=None
    for n in all_nodes:
        _log(n,verbose)
        if re.search(INPUT_TENSOR_NAME_REGEX, n) is not None:
            _log("It matches input regex! ",verbose)
            INPUT_TENSOR_NAME=n
        elif re.search(OUTPUT_TENSOR_NAME_REGEX, n) is not None:
            _log("It matches output regex! ",verbose)
            OUTPUT_TENSOR_NAME=n
        else:
            pass #this is an intermediate tensor

    if INPUT_TENSOR_NAME is None:
        _log("Warning no operation matches the input regex", verbose)
    if OUTPUT_TENSOR_NAME is None:
        _log("Warning no operation matches the output regex", verbose)


    return INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME

if __name__ == '__main__':
    try:
        pb_path=sys.argv[1]
        input_regex=sys.argv[2]
        output_regex=sys.argv[3]
    except:
        pb_path="../models_lib/TF_PB/ResNet50.pb"
        input_regex="^x:|input:"
        output_regex="Softmax:"

    inp,out=explore_tf_file(pb_path,input_regex,output_regex,verbose=False)

    print(f"{inp} {out}")
