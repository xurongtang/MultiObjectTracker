#!/bin/bash
# convert_to_mnn.sh - 将模型转换为 MNN 格式
# 支持: ONNX (.onnx), TensorFlow (.pb), TFLite (.tflite), Caffe (.prototxt + .caffemodel)

set -e  # 遇错退出

# ================== 配置区 ==================
# MNNConvert 路径（请根据你的安装路径修改！）
MNN_CONVERT="/home/rton/MultiObjectTracker/3rdparty/mnn-install/bin/MNNConvert"  # 如果在 PATH 中可写 "MNNConvert"
# 或使用绝对路径：
# MNN_CONVERT="/path/to/mnn-install/bin/MNNConvert"

# 是否启用量化（0=不量化，1=INT8 量化）
QUANTIZE=0
CALIB_DATA_DIR=""  # 量化时需要校准数据目录（可选）

# ================== 函数定义 ==================
log() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

error() {
    echo -e "\033[31m[ERROR]\033[0m $1" >&2
    exit 1
}

# 自动推断框架类型
infer_framework() {
    local model_file="$1"
    local ext="${model_file##*.}"
    case "${ext,,}" in
        onnx)
            echo "ONNX"
            ;;
        pb)
            echo "TENSORFLOW"
            ;;
        tflite)
            echo "TFLITE"
            ;;
        prototxt)
            echo "CAFFE"
            ;;
        *)
            error "Unsupported model extension: .$ext (support: onnx, pb, tflite, prototxt)"
            ;;
    esac
}

# ================== 参数解析 ==================
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_file> [output_mnn_file]"
    echo "Example: $0 model.onnx model.mnn"
    exit 1
fi

INPUT_MODEL="$1"
OUTPUT_MNN="${2:-${INPUT_MODEL%.*}.mnn}"  # 默认输出为同名 .mnn

# ================== 检查 MNNConvert ==================
if [ ! -f "$MNN_CONVERT" ] && ! command -v "$MNN_CONVERT" &> /dev/null; then
    error "MNNConvert not found at: $MNN_CONVERT"
fi

# ================== 确定框架类型 ==================
FRAMEWORK=$(infer_framework "$INPUT_MODEL")
log "Detected framework: $FRAMEWORK"

# ================== 构建转换命令 ==================
CMD=("$MNN_CONVERT" "-f" "$FRAMEWORK")

# 根据框架设置参数
case "$FRAMEWORK" in
    ONNX|TENSORFLOW|TFLITE)
        CMD+=("--modelFile" "$INPUT_MODEL")
        ;;
    CAFFE)
        # Caffe 需要 .prototxt 和 .caffemodel
        PROTO="${INPUT_MODEL}"
        CAFFEMODEL="${INPUT_MODEL%.prototxt}.caffemodel"
        if [ ! -f "$CAFFEMODEL" ]; then
            error "Caffe model file not found: $CAFFEMODEL"
        fi
        CMD+=("--prototxt" "$PROTO" "--caffemodel" "$CAFFEMODEL")
        ;;
esac

CMD+=("--MNNModel" "$OUTPUT_MNN" "--bizCode" "MNN")

# ================== 量化配置（可选） ==================
if [ "$QUANTIZE" -eq 1 ]; then
    if [ -z "$CALIB_DATA_DIR" ]; then
        log "Quantization enabled but no calib data dir, using default"
    else
        CMD+=("--weightQuantBits" "8" "--calibration" "$CALIB_DATA_DIR")
    fi
fi

# ================== 执行转换 ==================
log "Running: ${CMD[*]}"
"${CMD[@]}"

if [ $? -eq 0 ]; then
    log "✅ Conversion successful! Output: $OUTPUT_MNN"
else
    error "❌ Conversion failed!"
fi