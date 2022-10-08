#!/bin/bash

INPUT_PB_PATH=$1
OUTPUT_ONNX_PATH=$2

GET_IN_OUT_PY=/data/appli_PITSI/users/pochelu/project/inference_system/pierrick_tools/tensorflow_graph_parser.py

# Regex taking the input name and the output name of the DAG
TENSOR_IN_OUT_NAME=$(python3 ${GET_IN_OUT_PY} ${INPUT_PB_PATH} "^x:|input:" "Softmax:")

echo "========================"
echo $TENSOR_IN_OUT_NAME

tensor_names=($TENSOR_IN_OUT_NAME)
tensor_in_name=${tensor_names[0]}
tensor_out_name=${tensor_names[1]}
echo $tensor_in_name
echo $tensor_out_name


python3 -m tf2onnx.convert --input $INPUT_PB_PATH --inputs $tensor_in_name --output $OUTPUT_ONNX_PATH --outputs $tensor_out_name
