#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct detect_result {
    cv::Rect box;
    int classId;
    float confidence;
};

class ONNXYoloDetector {
public:
    ONNXYoloDetector(const std::string& modelPath,
                     const std::vector<std::string>& classNames,
                     int inputWidth = 640,
                     int inputHeight = 640,
                     float confThreshold = 0.5f,
                     float nmsThreshold = 0.4f);

    ~ONNXYoloDetector();

    void detect(cv::Mat& frame, std::vector<detect_result>& results);

private:
    void preprocess(const cv::Mat& frame, float* inputTensorValues);
    void postprocess(const std::vector<float>& outputTensorValues,
                     const cv::Size& frameSize,
                     std::vector<detect_result>& results);

    // ONNX Runtime 对象
    void* ortEnv = nullptr;
    void* ortSession = nullptr;
    void* ortMemoryInfo = nullptr;

    const int inputWidth_;
    const int inputHeight_;
    const float confThreshold_;
    const float nmsThreshold_;
    const std::vector<std::string> classNames_;

    float scale_ = 1.0f;      // 缩放因子
    float pad_x_ = 0.0f;      // x 方向 padding
    float pad_y_ = 0.0f;      // y 方向 padding
};