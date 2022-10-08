#!/bin/sh

LIB_PB_PATH=./models_lib/TF_PB/
LIB_ONNX_PATH=./models_lib/ONNX/
LIB_OPENVINO_PATH=./models_lib/OpenVINO/

# FROM PB TO ONNX
for f in "$LIB_PB_PATH"/*
do
dnn_name=$(basename $f .pb)
pb_dnn_path="${LIB_PB_PATH}${dnn_name}.pb"
onnx_dnn_path="${LIB_ONNX_PATH}${dnn_name}.onnx"
./from_PB_to_ONNX.sh $pb_dnn_path $onnx_dnn_path
done
