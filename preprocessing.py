from scipy.ndimage.interpolation import zoom
import numpy as np

def identity(x):
    return x
def divizion_by_255(x):
    return (x/255.).astype(np.float)

def from_modelname_to_inputsize():
    from_B_to_pixels={0:224,1:240,2:260,3:300,
                      4:380,5:456,6:528,7:600}
    m = {}
    for v in ["","V2"]:
        for i in range(0,7+1):
            m["EfficientNetB"+str(i)+str(v)]=from_B_to_pixels[i]
    m["EfficientNetV2L"]=480
    m["EfficientNetV2M"]=480
    m["EfficientNetV2S"]=384

    m["NASNetLarge"] = 331
    m["InceptionV3"]=299
    m["Xception"]=299
    m["InceptionResNetV2"]=299
    return m


def rescale(newsize):
    def f(x):
        assert (x.shape[1] == x.shape[2])  # assert square
        oldsize = x.shape[1]
        if oldsize != newsize:
            zoom_factor = float(newsize) / oldsize
            x2 = zoom(x, (1, zoom_factor, zoom_factor, 1), order=1)
        else:
            x2 = x
        assert(x2.shape[1]==x2.shape[2] and x2.shape[1]==newsize)
        return x2
    return f

def from_modelname_to_rescale():
    out={}
    kv=from_modelname_to_inputsize()
    for k,v in kv.items():
        out[k]=rescale(v)
    return out

def crop(v):
    def f(x):
        cropped_x=x[:,0:v,0:v,:]
        return cropped_x
    return f

def from_modelname_to_crop():
    out={}
    kv=from_modelname_to_inputsize()
    for k,v in kv.items():
        out[k]=crop(v)
    return out

def from_modelname_to_prepro():
    proc={}

    from tensorflow.keras.applications.mobilenet import preprocess_input
    proc["MobileNet"]=preprocess_input
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    proc["MobileNetV2"]=preprocess_input
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    proc["MobileNetV3Small"]=preprocess_input
    proc["MobileNetV3Large"]=preprocess_input

    from tensorflow.keras.applications.densenet import preprocess_input
    proc["DenseNet121"] = preprocess_input
    proc["DenseNet169"] = preprocess_input
    proc["DenseNet201"] = preprocess_input

    from tensorflow.keras.applications.resnet50 import preprocess_input
    proc["ResNet50"]=preprocess_input
    proc["ResNet101"] = preprocess_input
    proc["ResNet152"] = preprocess_input

    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    proc["ResNet50V2"]=preprocess_input
    proc["ResNet101V2"] = preprocess_input
    proc["ResNet152V2"] = preprocess_input
    proc["InceptionResNetV2"]=preprocess_input

    from tensorflow.keras.applications.vgg16 import preprocess_input
    proc["VGG16"]=preprocess_input
    from tensorflow.keras.applications.vgg19 import preprocess_input
    proc["VGG19"]=preprocess_input

    from tensorflow.keras.applications.nasnet import preprocess_input
    proc["NASNetLarge"]=preprocess_input
    proc["NASNetMobile"]=preprocess_input

    from tensorflow.keras.applications.xception import preprocess_input
    proc["Xception"]=preprocess_input

    from tensorflow.keras.applications.inception_v3 import preprocess_input
    proc["InceptionV3"]=preprocess_input

    from tensorflow.keras.applications.efficientnet import preprocess_input as prep_eff
    from env.install.NewNeuralNetworks.efficientnet_V2 import preprocess_input as prep_eff2
    # Get all fname
    effnames=[]
    for v in ["","V2"]:
        for i in range(0,7+1):
            effname="EfficientNetB"+str(i)+str(v)
            effnames.append(effname)
    effnames.append("EfficientNetV2L")
    effnames.append("EfficientNetV2M")
    effnames.append("EfficientNetV2S")

    for effname in effnames:
        #pxl=M_efficient_name_to_size[effname] # from effname to size
        # from effname to prepro
        if "V2" in effname:
            p=prep_eff2
        else:
            p=prep_eff
        proc[effname]=p
    return proc



def from_modelname_to_preparation_f():#TODO: is it used somewhere ?
    from_modelname_to_prepro_f = from_modelname_to_prepro()
    from_modelname_to_rescale_f = from_modelname_to_rescale()

    k1=from_modelname_to_prepro_f.keys()
    k2=from_modelname_to_rescale_f.keys()
    modelnames= set(k1) and set(k2)

    out={}
    for name in modelnames:
        f1 = from_modelname_to_rescale_f.get(name,identity)
        f2 = from_modelname_to_prepro_f[name] # crash if unknown
        def preparation(x:np.ndarray):
            rescaled_x=f1(x)
            prepared_x=f2(rescaled_x)
            return prepared_x
        out[name]=preparation
    return out
