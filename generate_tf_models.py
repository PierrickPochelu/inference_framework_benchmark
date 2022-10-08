from tensorflow.keras import applications
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from pierrick_tools import preprocessing
import os

def from_kerasObject_to_pbFile2(keras_model,pb_path):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    from tensorflow.python.tools import optimize_for_inference_lib

    #loaded = tf.saved_model.load(keras_model)
    infer = keras_model.signatures['serving_default']
    f = tf.function(infer).get_concrete_function(
        flatten_input=tf.TensorSpec(shape=[None, 224, 224, 3],
                                    dtype=tf.float32))  # change this line for your own inputs
    f2 = convert_variables_to_constants_v2(f)
    graph_def = f2.graph.as_graph_def()

    frozen_func = convert_variables_to_constants_v2(graph_def)

    # Remove NoOp nodes
    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op == 'NoOp':
            del graph_def.node[i]
    for node in graph_def.node:
        for i in reversed(range(len(node.input))):
            if node.input[i][0] == '^':
                del node.input[i]
    # Parse graph's inputs/outputs
    graph_inputs = [x.name.rsplit(':')[0] for x in frozen_func.inputs]
    graph_outputs = [x.name.rsplit(':')[0] for x in frozen_func.outputs]

    graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def,
                                                                  graph_inputs,
                                                                  graph_outputs,                                                                                   tf.float32.as_datatype_enum)

    # Export frozen graph
    with tf.io.gfile.GFile(pb_path, 'wb') as f:
        f.write(graph_def.SerializeToString())


def from_kerasObject_to_pbFile(keras_model,pb_path):

    filename=os.path.basename(pb_path)
    dirname=os.path.dirname(pb_path)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: keras_model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    input_tensor_name=frozen_func.inputs
    print("Frozen model outputs: ")
    output_tensor_name=frozen_func.outputs

    """
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=dirname,
                      name=filename,
                      as_text=False)
    """
    # Export frozen graph
    #https://stackoverflow.com/questions/58119155/freezing-graph-to-pb-in-tensorflow2
    with tf.io.gfile.GFile(os.path.join(dirname,filename), 'wb') as f:
        f.write(frozen_func.graph.as_graph_def().SerializeToString())

    return input_tensor_name, output_tensor_name

def CREATE_BUILDERS():
    builders = []
    model_names = []



    """
    model_names.append("MobileNetV2")
    builders.append(applications.MobileNetV2)
    model_names.append("MobileNet")
    builders.append(applications.MobileNet)
    model_names.append("MobileNetV3Small")
    builders.append(applications.MobileNetV3Small)
    model_names.append("MobileNetV3Large")
    builders.append(applications.MobileNetV3Large)
    model_names.append("ResNet50")
    builders.append(applications.ResNet50)
    model_names.append("ResNet101V2")
    builders.append(applications.ResNet101V2)
    model_names.append("ResNet152V2")
    builders.append(applications.ResNet152V2)
    """

    """
    model_names.append("InceptionResNetV2")
    builders.append(applications.InceptionResNetV2)
    model_names.append("NASNetMobile")
    builders.append(applications.NASNetMobile)
    model_names.append("NASNetLarge")
    builders.append(applications.NASNetLarge)

    model_names.append("Xception")
    builders.append(applications.Xception)
    model_names.append("InceptionV3")
    builders.append(applications.InceptionV3)
    """
    """
    model_names.append("DenseNet121")
    builders.append(applications.DenseNet121)
    model_names.append("DenseNet169")
    builders.append(applications.DenseNet169)
    model_names.append("DenseNet201")
    builders.append(applications.DenseNet201)

    model_names.append("VGG16")
    builders.append(applications.VGG16)
    model_names.append("VGG19")
    builders.append(applications.VGG19)
    """

    """
    model_names.append("ResNet50V2")
    builders.append(applications.resnet_v2.ResNet50V2)
    model_names.append("ResNet101")
    builders.append(applications.resnet.ResNet101)
    model_names.append("ResNet152")
    builders.append(applications.resnet.ResNet152)
    model_names.append("ResNet152")
    builders.append(applications.resnet.ResNet152)
    """

    """
    model_names.append("EfficientNetB0")
    builders.append(applications.efficientnet.EfficientNetB0)
    model_names.append("EfficientNetB1")
    builders.append(applications.efficientnet.EfficientNetB1)
    model_names.append("EfficientNetB2")
    builders.append(applications.efficientnet.EfficientNetB2)
    model_names.append("EfficientNetB3")
    builders.append(applications.efficientnet.EfficientNetB3)
    model_names.append("EfficientNetB4")
    builders.append(applications.efficientnet.EfficientNetB4)
    model_names.append("EfficientNetB5")
    builders.append(applications.efficientnet.EfficientNetB5)
    model_names.append("EfficientNetB6")
    builders.append(applications.efficientnet.EfficientNetB6)
    """
    model_names.append("EfficientNetB7")
    builders.append(applications.efficientnet.EfficientNetB7)
    """
    from env.install.NewNeuralNetworks import efficientnet_V2
    model_names.append("EfficientNetB0V2")
    builders.append(efficientnet_V2.EfficientNetV2B0)
    model_names.append("EfficientNetB1V2")
    builders.append(efficientnet_V2.EfficientNetV2B1)
    model_names.append("EfficientNetB2V2")
    builders.append(efficientnet_V2.EfficientNetV2B2)
    model_names.append("EfficientNetB3V2")
    builders.append(efficientnet_V2.EfficientNetV2B3)
    model_names.append("EfficientNetV2L")
    builders.append(efficientnet_V2.EfficientNetV2L)
    model_names.append("EfficientNetV2M")
    builders.append(efficientnet_V2.EfficientNetV2M)
    model_names.append("EfficientNetV2S")
    builders.append(efficientnet_V2.EfficientNetV2S)
    """
    return builders, model_names

def save_models(builder, model_name, SavedModel_path,PB_path,force=True):

    SavedModel_path_exist=os.path.exists(SavedModel_path)
    PB_path_exist=os.path.exists(PB_path)

    if not (SavedModel_path_exist and PB_path_exist) or force:
        print(f"{model_name}")
        from_modelname_to_imgsize=preprocessing.from_modelname_to_inputsize()
        #if model_name in from_modelname_to_imgsize:
        imgsize=from_modelname_to_imgsize.get(model_name,224)
        print(f" {model_name} is build with input size = {imgsize}")
        try:
            keras_model=builder(input_shape=(imgsize,imgsize,3),weights="imagenet")
        except ValueError:
            print("WARNING no imagenet found")
            keras_model=builder(input_shape=(imgsize,imgsize,3),weights=None)
    else:
        print(f"{model_name} already stored")

    if not SavedModel_path_exist or force:
        keras_model.save(SavedModel_path)

    if not PB_path_exist or force:
        from_kerasObject_to_pbFile(keras_model, PB_path)


def RUN():
    builders, model_names=CREATE_BUILDERS()
    for builder, model_name in zip(builders, model_names):
        #if model_name!="ResNet50":
        #    print("break")
        #else:
        SavedModel_path="./models_lib/TF_SavedModel/" + model_name
        PB_path="./models_lib/TF_PB/" + model_name+".pb"
        save_models(builder, model_name, SavedModel_path, PB_path)
        print("saved")

if __name__ == '__main__':
    RUN()
