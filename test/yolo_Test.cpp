#include "yolo/onnx_yolo_detecter.h"
#include <opencv2/opencv.hpp>

int main() {
    std::vector<std::string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    ONNXYoloDetector detector(
        "/home/rton/MultiObjectTracker/test/yolo12n.onnx", 
        classNames,
        640,640,
        0.2,
        0.4
    );

    cv::Mat frame = cv::imread("/home/rton/MultiObjectTracker/test/test.jpeg");
    std::vector<detect_result> results;
    detector.detect(frame, results);

    // 可视化
    for (const auto& r : results) {
        cv::rectangle(frame, r.box, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("result.jpg", frame);
    return 0;
}