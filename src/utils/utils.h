#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>


// ==================== IoU 计算 ====================
// 输入：两个 OpenCV Rect_<float>（x, y, w, h）
// 输出：交并比 [0, 1]
inline float CalculateIoU(cv::Rect_<float> box1, cv::Rect_<float> box2) {
    // 计算交集
    float x_left = std::max(box1.x, box2.x);
    float y_top = std::max(box1.y, box2.y);
    float x_right = std::min(box1.x + box1.width, box2.x + box2.width);
    float y_bottom = std::min(box1.y + box1.height, box2.y + box2.height);

    if (x_right <= x_left || y_bottom <= y_top) {
        return 0.0f; // 无交集
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float box1_area = box1.width * box1.height;
    float box2_area = box2.width * box2.height;
    float union_area = box1_area + box2_area - intersection_area;

    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

// ==================== 余弦距离 ====================
// 输入：两个特征向量 f1, f2（如 ReID 特征）
// 输出：余弦距离 = 1 - 余弦相似度 ∈ [0, 2]
inline float CosineLoss(const std::vector<float>& f1, const std::vector<float>& f2) {
    if (f1.empty() || f2.empty() || f1.size() != f2.size()) {
        return 1.0f; // 无效输入，返回最大距离
    }

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < f1.size(); ++i) {
        dot += static_cast<double>(f1[i]) * f2[i];
        norm1 += static_cast<double>(f1[i]) * f1[i];
        norm2 += static_cast<double>(f2[i]) * f2[i];
    }

    if (norm1 <= 0.0 || norm2 <= 0.0) {
        return 1.0f;
    }

    double cosine_sim = dot / (std::sqrt(norm1) * std::sqrt(norm2));
    // 限制在 [-1, 1] 防止浮点误差
    cosine_sim = std::max(-1.0, std::min(1.0, cosine_sim));
    return static_cast<float>(1.0 - cosine_sim);
}

// ==================== 坐标转换：xyah → tlwh ====================
// xyah: [center_x, center_y, aspect_ratio, height]
// tlwh: [top_left_x, top_left_y, width, height]
inline std::vector<float> xyah_to_tlwh(const std::vector<float>& xyah) {
    if (xyah.size() != 4) return {0, 0, 0, 0};
    float cx = xyah[0], cy = xyah[1], a = xyah[2], h = xyah[3];
    float w = a * h;
    float x = cx - w / 2.0f;
    float y = cy - h / 2.0f;
    return {x, y, w, h};
}

// ==================== 坐标转换：tlwh → xyah ====================
// tlwh: [top_left_x, top_left_y, width, height]
// xyah: [center_x, center_y, aspect_ratio, height]
inline std::vector<float> tlwh_to_xyah(const std::vector<float>& tlwh) {
    if (tlwh.size() != 4) return {0, 0, 0, 0};
    float x = tlwh[0], y = tlwh[1], w = tlwh[2], h = tlwh[3];
    if (h <= 0) h = 1e-6f; // 防止除零
    float cx = x + w / 2.0f;
    float cy = y + h / 2.0f;
    float a = w / h;
    return {cx, cy, a, h};
}

// ==================== 匈牙利算法：双向最优匹配（适配 cv::Mat） ====================
// 输入：
//   - cost_matrix: 代价矩阵，类型 CV_32F，尺寸 [num_tracks x num_dets]
//   - gating_threshold: 门控阈值（<=0 表示不启用）
// 输出：
//   - matches: 匹配对 (track_idx, det_idx)
//   - unmatched_tracks, unmatched_dets: 未匹配索引
inline void HungarianAlgorithm(
    const cv::Mat& cost_matrix,
    float gating_threshold,
    std::vector<std::pair<size_t, size_t>>& matches,
    std::vector<size_t>& unmatched_tracks,
    std::vector<size_t>& unmatched_dets) {

    matches.clear();
    unmatched_tracks.clear();
    unmatched_dets.clear();

    if (cost_matrix.empty()) {
        return;
    }

    size_t num_tracks = cost_matrix.rows;
    size_t num_dets = cost_matrix.cols;

    if (num_tracks == 0 || num_dets == 0) {
        for (size_t i = 0; i < num_tracks; ++i) unmatched_tracks.push_back(i);
        for (size_t j = 0; j < num_dets; ++j) unmatched_dets.push_back(j);
        return;
    }

    // 贪心排序匹配（适用于 DeepSORT 小规模场景）
    std::vector<std::tuple<float, size_t, size_t>> candidates;
    for (size_t i = 0; i < num_tracks; ++i) {
        for (size_t j = 0; j < num_dets; ++j) {
            float cost_val = cost_matrix.at<float>((int)i, (int)j);
            if (gating_threshold <= 0 || cost_val <= gating_threshold) {
                candidates.emplace_back(cost_val, i, j);
            }
        }
    }

    // 按代价升序排序
    std::sort(candidates.begin(), candidates.end());

    std::vector<bool> track_used(num_tracks, false);
    std::vector<bool> det_used(num_dets, false);

    for (const auto& [cost_val, i, j] : candidates) {
        if (!track_used[i] && !det_used[j]) {
            matches.emplace_back(i, j);
            track_used[i] = true;
            det_used[j] = true;
        }
    }

    for (size_t i = 0; i < num_tracks; ++i) {
        if (!track_used[i]) unmatched_tracks.push_back(i);
    }
    for (size_t j = 0; j < num_dets; ++j) {
        if (!det_used[j]) unmatched_dets.push_back(j);
    }
}

#endif // UTILS_H