INPUT_PB_PATH=$1
OUTPUT_PATH=$2
INPUT_SIZE=$3

#e.g., ../models_lib/TF_PB/ResNet50.pb ../models_lib/OpenVINO/ResNet50/

export TOOLS_ROOT=/data/appli_PITSI/users/pochelu/project/inference_system/scripts/
. ${TOOLS_ROOT}/env_python.sh

GET_IN_OUT_PY=/data/appli_PITSI/users/pochelu/project/inference_system/pierrick_tools/tensorflow_graph_parser.py

TENSOR_IN_OUT_NAME=$(python3 ${GET_IN_OUT_PY} ${INPUT_PB_PATH} "^x:|input:" "Softmax:")

echo "========================"
echo $TENSOR_IN_OUT_NAME

tensor_names=($TENSOR_IN_OUT_NAME)
tensor_in_name=${tensor_names[0]}
tensor_out_name=${tensor_names[1]}
echo $tensor_in_name
echo $tensor_out_name

tensor_in_name=${tensor_in_name::-2}
tensor_out_name=${tensor_out_name::-2}

#INPUT_PB_PATH="${INPUT_PB_PATH}/saved_model.pb"

source /data/appli_PITSI/HGX2/hpml/openvino/scripts/env_softstack.sh

mo.py --input_model $INPUT_PB_PATH --input $tensor_in_name\
 --output_dir $OUTPUT_PATH --output $tensor_out_name\
 --input_shape [1,224,224,3]
# --enable_concat_optimization\
# --disable_fusing\
# --disable_gfusing

#exit
#mo.py --input_model $INPUT_PB_PATH --input $tensor_in_name\
# --output_dir $OUTPUT_PATH --output $tensor_out_name\
# --input_shape [1,224,224,3]\
# --data_type FP32\

