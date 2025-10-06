#include "tracker/DeepSortTracker.h"
#include "yolo/onnx_yolo_detecter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // ==================== 配置路径 ====================
    std::string yolo_model_path = "/home/rton/MultiObjectTracker/test/yolo12n.onnx";
    std::string reid_model_path = "/home/rton/MultiObjectTracker/src/InferMNN/osnet/osnet_x1_0_market.mnn";

    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // ==================== 初始化检测器和跟踪器 ====================
    try {
        ONNXYoloDetector yolo_detector(
            yolo_model_path,
            class_names,
            640,
            640,
            0.6f,
            0.5f
        );

        DeepSortTracker tracker(
            reid_model_path,
            0.7f,
            30,
            3,
            0.2f
        );

        // ==================== 打开视频 ====================
        std::string input_video = "/home/rton/MultiObjectTracker/test/demo.mp4";
        std::string output_video = "/home/rton/MultiObjectTracker/test/output_deepsort.mp4";

        cv::VideoCapture cap(input_video);
        if (!cap.isOpened()) {
            std::cerr << "❌ 无法打开视频源: " << input_video << std::endl;
            return -1;
        }

        // 获取视频属性
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        std::cout << "📹 视频信息: " << width << "x" << height 
                  << " @ " << fps << " FPS, 总帧数: " << total_frames << std::endl;

        // 创建视频写入器（使用与原视频相同的 FPS 和尺寸）
        cv::VideoWriter writer;
        writer.open(output_video, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "❌ 无法创建输出视频: " << output_video << std::endl;
            return -1;
        }

        std::cout << "🚀 开始 YOLO + DeepSORT 跟踪...\n";
        cv::Mat frame;
        int frame_count = 0;

        while (cap.read(frame)) {
            if (frame.empty()) break;

            auto start = std::chrono::high_resolution_clock::now();

            // Step 1: YOLO 检测
            std::vector<detect_result> yolo_results;
            yolo_detector.detect(frame, yolo_results);

            // Step 2: 转换检测结果
            std::vector<cv::Rect_<float>> detections;
            for (const auto& det : yolo_results) {
                detections.emplace_back(
                    static_cast<float>(det.box.x),
                    static_cast<float>(det.box.y),
                    static_cast<float>(det.box.width),
                    static_cast<float>(det.box.height)
                );
            }

            // Step 3: DeepSORT 跟踪
            auto tracks = tracker.update(frame, detections);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "🕒 帧 " << frame_count << ": 处理时间 = " 
                      << duration.count() << " ms, 检测数 = " << yolo_results.size() 
                      << ", 跟踪数 = " << tracks.size() << std::endl;

            // Step 4: 可视化
            cv::Mat vis = frame.clone();
            for (const auto& track : tracks) {
                cv::Rect_<float> box = track.to_tlwh();
                cv::Rect draw_box(
                    static_cast<int>(box.x),
                    static_cast<int>(box.y),
                    static_cast<int>(box.width),
                    static_cast<int>(box.height)
                );
                cv::rectangle(vis, draw_box, cv::Scalar(0, 255, 0), 2);
                cv::putText(vis, "ID:" + std::to_string(track.id),
                            cv::Point(draw_box.x, draw_box.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }

            // 保存到输出视频
            writer.write(vis);

            // // 可选：显示实时窗口（可注释掉以加速处理）
            // cv::imshow("YOLO + DeepSORT", vis);
            // if (cv::waitKey(1) == 27) break; // ESC 退出

            frame_count++;
        }

        // 释放资源
        cap.release();
        writer.release();
        cv::destroyAllWindows();

        std::cout << "\n✅ 跟踪完成！输出视频已保存至: " << output_video << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}