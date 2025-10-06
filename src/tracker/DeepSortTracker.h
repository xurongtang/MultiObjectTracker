#ifndef DEEPSORTTRACKER_H
#define DEEPSORTTRACKER_H

// ==================== 标准库与第三方依赖 ====================
#include <iostream>        // 用于调试输出（可选）
#include <vector>          // 动态数组，存储轨迹、检测框、特征等
#include <memory>          // 智能指针，管理 ReID 模型生命周期
#include <opencv2/opencv.hpp> // OpenCV，用于图像处理和矩形框表示

// ==================== 自定义模块 ====================
#include "kalmanfilter/kalman.h"        // Kalman 滤波器，用于目标运动预测
#include "yolo/onnx_yolo_detecter.h"    // YOLO 检测器（此处仅声明依赖，实际在 cpp 中使用）
#include "InferMNN/mnnInfer.h"          // MNN ReID 特征提取器（用于外观特征）
#include "utils/utils.h"                // 工具函数：IoU、余弦距离、坐标转换等

// ==================== 轨迹状态枚举 ====================
// 定义轨迹的三种生命周期状态，用于控制轨迹是否输出
enum class TrackState {
    Tentative,   // 试探态：刚创建，尚未确认（避免误检输出）
    Confirmed,   // 已确认：连续命中多次，可信度高，参与输出和匹配
    Deleted      // 已删除：长时间未匹配，从跟踪列表中移除
};

// ==================== 轨迹类（Track） ====================
// 每个目标对应一个 Track 实例，封装状态、滤波器、特征和生命周期
class Track {
    public:
        // 构造函数：用首次检测框和 ReID 特征初始化轨迹
        // - id: 全局唯一轨迹 ID
        // - box: 检测框（格式：x, y, w, h，即 tlwh）
        // - feature: ReID 外观特征向量（如 128 维 float 向量）
        Track(int id, const cv::Rect_<float>& box, const std::vector<float>& feature, int n_init);
        
        // 预测：调用 Kalman 滤波器预测下一帧位置，并更新内部 box
        void predict();
        
        // 更新：当轨迹与检测匹配时，用新观测更新 Kalman 状态和特征
        // - box: 新的检测框（tlwh）
        // - feature: 新提取的 ReID 特征
        void update(const cv::Rect_<float>& box, const std::vector<float>& feature);

        // 获取当前轨迹框（tlwh 格式），用于输出或可视化
        cv::Rect_<float> to_tlwh() const;

        // =============== 公有成员变量（便于访问）===============
        int id;                          // 轨迹唯一 ID
        cv::Rect_<float> box;            // 当前位置（tlwh 格式）
        std::vector<float> feature;      // 最新 ReID 特征（用于外观匹配）
        KalmanFilter kalman;             // Kalman 滤波器实例（每轨迹独享）
        
        // 生命周期计数器
        int time_since_update;  // 自上次成功匹配以来经过的帧数（用于判断是否删除）
        int hits;               // 连续成功匹配的次数（用于从 Tentative 升级为 Confirmed）
        int age;                // 轨迹总存活帧数（从创建至今）
        TrackState state;       // 当前轨迹状态（Tentative / Confirmed / Deleted）
        int n_init_;            // Track 自己保存 n_init 轨迹确认所需最小命中次数
};

// ==================== DeepSORT 跟踪器主类 ====================
// 封装多目标跟踪的核心逻辑：预测、匹配、更新、创建、删除
class DeepSortTracker {
    public:
        // 构造函数：初始化跟踪器参数和 ReID 模型
        // - reid_model_path: ReID 模型文件路径（.mnn）
        // - max_iou_distance: IoU 匹配阈值（用于初步筛选）
        // - max_age: 轨迹最大存活时间（未匹配超过此帧数则删除）
        // - n_init: 轨迹确认所需最小连续命中次数（如 3 帧）
        // - max_cosine_distance: 余弦距离阈值（> 此值认为外观不匹配）
        DeepSortTracker(
            const std::string& reid_model_path,
            float max_iou_distance = 0.7f,
            int max_age = 30,
            int n_init = 3,
            float max_cosine_distance = 0.2f
        );

        // 主接口：输入当前帧图像和检测结果，输出跟踪轨迹
        // - frame: 当前视频帧（用于 ReID 特征提取）
        // - detections: YOLO 等检测器输出的边界框列表（tlwh 格式）
        // - 返回: 所有 Confirmed 状态的轨迹（可用于可视化或后续处理）
        std::vector<Track> update(const cv::Mat& frame, const std::vector<cv::Rect_<float>>& detections);

    private:
        // 匹配函数：将现有轨迹与当前检测进行关联
        // - detections: 当前帧检测框
        // - features: 对应的 ReID 特征
        // - matches: 输出匹配对（轨迹索引, 检测索引）
        // - unmatched_tracks: 未匹配的轨迹索引
        // - unmatched_dets: 未匹配的检测索引
        void _match(
            const std::vector<cv::Rect_<float>>& detections,
            const std::vector<std::vector<float>>& features,
            std::vector<std::pair<size_t, size_t>>& matches,
            std::vector<size_t>& unmatched_tracks,
            std::vector<size_t>& unmatched_dets
        );

        // 辅助函数：获取所有轨迹的预测框（用于匹配）
        std::vector<cv::Rect_<float>> _get_predicted_boxes() const;

        // =============== 私有成员变量 ===============
        std::vector<Track> tracks_;      // 当前所有活跃轨迹（包括 Tentative 和 Confirmed）
        int next_id_;                    // 下一个新轨迹的 ID（自增）

        // 跟踪超参数（可在构造时配置）
        float max_iou_distance_;         // IoU 匹配阈值
        int max_age_;                    // 轨迹最大未匹配帧数
        int n_init_;                     // 轨迹确认所需最小命中次数
        float max_cosine_distance_;      // 余弦距离阈值（越小越严格）

        // ReID 模型实例（使用智能指针自动管理内存）
        std::unique_ptr<MNNInfer> reid_model_;
};

#endif // DEEPSORTTRACKER_H