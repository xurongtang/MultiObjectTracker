#include "onnx_yolo_detecter.h"
#include <onnxruntime_cxx_api.h>
#include <iostream>

ONNXYoloDetector::ONNXYoloDetector(const std::string& modelPath,
                                   const std::vector<std::string>& classNames,
                                   int inputWidth,
                                   int inputHeight,
                                   float confThreshold,
                                   float nmsThreshold)
    : inputWidth_(inputWidth),
      inputHeight_(inputHeight),
      confThreshold_(confThreshold),
      nmsThreshold_(nmsThreshold),
      classNames_(classNames) {

    // 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    // 保存到类成员（需用智能指针或手动管理，此处简化）
    ortEnv = new Ort::Env(std::move(env));
    ortSession = new Ort::Session(std::move(session));

    // 创建内存信息
    ortMemoryInfo = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault));
}

ONNXYoloDetector::~ONNXYoloDetector() {
    delete static_cast<Ort::MemoryInfo*>(ortMemoryInfo);
    delete static_cast<Ort::Session*>(ortSession);
    delete static_cast<Ort::Env*>(ortEnv);
}

void ONNXYoloDetector::preprocess(const cv::Mat& frame, float* inputTensorValues) {
    // 保持预处理相同：YOLOv8 官方使用的是 "等比例缩放 + 中心填充到 640x640"

    int w = frame.cols;
    int h = frame.rows;
 
    // 计算缩放因子（保持宽高比）
    float scale = std::min((float)inputWidth_ / w, (float)inputHeight_ / h);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    // 保存参数供后处理使用
    scale_ = scale;
    pad_x_ = (inputWidth_ - new_w) / 2.0f;
    pad_y_ = (inputHeight_ - new_h) / 2.0f;
    
    // 等比例缩放
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    
    // 创建 640x640 黑色背景
    cv::Mat boxed = cv::Mat::zeros(inputHeight_, inputWidth_, CV_8UC3);
    
    // 中心填充
    cv::Rect roi((int)pad_x_, (int)pad_y_, new_w, new_h);
    resized.copyTo(boxed(roi));
    
    // HWC to CHW, normalize to [0,1]
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < inputWidth_ * inputHeight_; ++i) {
            inputTensorValues[c * inputWidth_ * inputHeight_ + i] =
                boxed.at<cv::Vec3b>(i / inputWidth_, i % inputWidth_)[c] / 255.0f;
        }
    }
}

void ONNXYoloDetector::detect(cv::Mat& frame, std::vector<detect_result>& results) {
    // 1. 预处理
    std::vector<float> inputTensorValues(3 * inputWidth_ * inputHeight_);
    preprocess(frame, inputTensorValues.data());

    // 2. 创建输入张量
    std::vector<int64_t> inputShape = {1, 3, inputWidth_, inputHeight_};
    Ort::MemoryInfo* memoryInfo = static_cast<Ort::MemoryInfo*>(ortMemoryInfo);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        *memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    // 3. 推理
    Ort::Session* session = static_cast<Ort::Session*>(ortSession);
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};
    auto outputTensors = session->Run(
        Ort::RunOptions{nullptr},
        inputNames, &inputTensor, 1,
        outputNames, 1
    );

    // 4. 获取输出数据
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> outputTensorValues(outputData, outputData + outputSize);

    // 5. 后处理
    postprocess(outputTensorValues, frame.size(), results);
}

void ONNXYoloDetector::postprocess(const std::vector<float>& outputTensorValues,
                                   const cv::Size& frameSize,
                                   std::vector<detect_result>& results) {
    const int numBoxes = 8400;
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < numBoxes; ++i) {
        // 获取原始输出坐标（相对于 640x640 输入）
        float cx = outputTensorValues[i];
        float cy = outputTensorValues[i + numBoxes];
        float w = outputTensorValues[i + 2 * numBoxes];
        float h = outputTensorValues[i + 3 * numBoxes];

        // 还原到 letterbox 图像坐标
        float x1 = (cx - w * 0.5f);
        float y1 = (cy - h * 0.5f);
        float x2 = (cx + w * 0.5f);
        float y2 = (cy + h * 0.5f);

        // 移除 padding 偏移
        x1 -= pad_x_;
        y1 -= pad_y_;
        x2 -= pad_x_;
        y2 -= pad_y_;

        // 通过缩放因子还原到原始图像
        x1 /= scale_;
        y1 /= scale_;
        x2 /= scale_;
        y2 /= scale_;

        // 转换为 OpenCV Rect
        int left = (int)std::max(0.0f, x1);
        int top = (int)std::max(0.0f, y1);
        int width = (int)std::max(0.0f, x2 - x1);
        int height = (int)std::max(0.0f, y2 - y1);

        // 类别分数
        float maxConf = 0.0f;
        int classId = 0;
        for (int c = 0; c < 80; ++c) {
            float score = outputTensorValues[i + (4 + c) * numBoxes];
            if (score > maxConf) {
                maxConf = score;
                classId = c;
            }
        }

        if (maxConf > confThreshold_) {
            boxes.emplace_back(left, top, width, height);
            classIds.push_back(classId);
            confidences.push_back(maxConf);
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold_, nmsThreshold_, indices);
    for (int idx : indices) {
        detect_result dr;
        dr.box = boxes[idx];
        dr.classId = classIds[idx];
        dr.confidence = confidences[idx];
        results.push_back(dr);
    }
}