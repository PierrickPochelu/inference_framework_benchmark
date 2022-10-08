
export CUDA_VISIBLE_DEVICES="-1"

#for dnn in EfficientNetB0 VGG19 ResNet50 DenseNet201
#for dnn in DenseNet121 DenseNet169 InceptionResNetV2 InceptionV3 MobileNet MobileNetV2 MobileNetV3Large
for dnn in MobileNetV3Small NASNetLarge NASNetMobile ResNet101V2 ResNet152V2 ResNet50 VGG16
do
echo $dnn
#./from_PB_to_OpenVINO.sh ../models_lib/TF_PB/${dnn}.pb ../models_lib/OpenVINO_FP16/${dnn} FP16
#./from_PB_to_OpenVINO.sh ../models_lib/TF_PB/${dnn}.pb ../models_lib/OpenVINO/${dnn} FP32
./from_PB_to_ONNX.sh ../models_lib/TF_PB/${dnn}.pb ../models_lib/ONNX/${dnn}.onnx
done

