#include "DeepSortTracker.h"
#include <algorithm>
#include <numeric>

// ==================== Track ====================

Track::Track(int id, const cv::Rect_<float>& box, const std::vector<float>& feature, int n_init)
    : id(id), box(box), feature(feature), n_init_(n_init),
      time_since_update(0), hits(1), age(1), state(TrackState::Tentative) {
    std::vector<float> xyah = tlwh_to_xyah({box.x, box.y, box.width, box.height});
    Eigen::Vector4f z(xyah[0], xyah[1], xyah[2], xyah[3]);
    kalman.update(z);
}

void Track::predict() {
    Eigen::Vector4f pred = kalman.predict();
    std::vector<float> tlwh = xyah_to_tlwh({pred[0], pred[1], pred[2], pred[3]});
    box = cv::Rect_<float>(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
    age++;
    time_since_update++;
}

void Track::update(const cv::Rect_<float>& box, const std::vector<float>& feature) {
    this->box = box;
    this->feature = feature;
    std::vector<float> xyah = tlwh_to_xyah({box.x, box.y, box.width, box.height});
    Eigen::Vector4f z(xyah[0], xyah[1], xyah[2], xyah[3]);
    kalman.update(z);
    hits++;
    time_since_update = 0;
    if (state == TrackState::Tentative && hits >= n_init_) {
        state = TrackState::Confirmed;
    }
}

cv::Rect_<float> Track::to_tlwh() const {
    return box;
}

// ==================== DeepSortTracker ====================

DeepSortTracker::DeepSortTracker(
    const std::string& reid_model_path,
    float max_iou_distance,
    int max_age,
    int n_init,
    float max_cosine_distance
)
    : next_id_(1),
      max_iou_distance_(max_iou_distance),
      max_age_(max_age),
      n_init_(n_init),
      max_cosine_distance_(max_cosine_distance) {
    
    // 初始化 ReID 模型（使用你的 MNNInfer）
    float mean[3] = {0.485f, 0.456f, 0.406f}; // ImageNet mean
    float std[3]  = {0.229f, 0.224f, 0.225f}; // ImageNet std
    reid_model_ = std::make_unique<MNNInfer>(reid_model_path, mean, std);
    
    if (reid_model_->loadModel() != 0) {
        throw std::runtime_error("Failed to load ReID model!");
    }
}

std::vector<Track> DeepSortTracker::update(
    const cv::Mat& frame,
    const std::vector<cv::Rect_<float>>& detections) {

    // Step 1: 提取 ReID 特征
    std::vector<cv::Mat> crops;
    for (const auto& det : detections) {
        cv::Rect_<int> roi(
            static_cast<int>(det.x),
            static_cast<int>(det.y),
            static_cast<int>(det.width),
            static_cast<int>(det.height)
        );
        // 边界检查
        roi &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.area() <= 0) {
            crops.push_back(cv::Mat()); // 空图
        } else {
            crops.push_back(frame(roi).clone());
        }
    }

    std::vector<std::vector<float>> features;
    std::vector<std::vector<float>> outputs;

    if (reid_model_->runInference(crops, outputs) != 0 || outputs.empty()) {
        // 推理失败，用零向量填充
        features.resize(detections.size(), std::vector<float>(512, 0.0f)); // 注意：维度应匹配模型
    } else {
        // ✅ 直接赋值！outputs[i] 就是第 i 个检测的特征
        if (outputs.size() != detections.size()) {
            std::cerr << "⚠️ Output count mismatch! Expected " 
                    << detections.size() << ", got " << outputs.size() << std::endl;
            features.resize(detections.size(), std::vector<float>(512, 0.0f));
        } else {
            features = std::move(outputs); // ✅ 直接移动，无需拆分
        }
    }

    // Step 2: 预测所有轨迹
    for (auto& track : tracks_) {
        track.predict();
    }

    // Step 3: 匹配
    std::vector<std::pair<size_t, size_t>> matches;
    std::vector<size_t> unmatched_tracks, unmatched_dets;
    _match(detections, features, matches, unmatched_tracks, unmatched_dets);

    // Step 4: 更新匹配的轨迹
    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> det_used(detections.size(), false);

    for (const auto& [t_idx, d_idx] : matches) {
        tracks_[t_idx].update(detections[d_idx], features[d_idx]);
        track_used[t_idx] = true;
        det_used[d_idx] = true;
    }

    // Step 5: 处理未匹配轨迹
    std::vector<Track> new_tracks;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!track_used[i]) {
            tracks_[i].time_since_update++;
            if (tracks_[i].time_since_update <= max_age_) {
                if (tracks_[i].state == TrackState::Confirmed || tracks_[i].hits >= n_init_) {
                    new_tracks.push_back(std::move(tracks_[i]));
                }
                // Tentative 未命中直接丢弃
            }
        } else {
            new_tracks.push_back(std::move(tracks_[i]));
        }
    }

    // Step 6: 创建新轨迹
    for (size_t j = 0; j < detections.size(); ++j) {
        if (!det_used[j]) {
            Track new_track(next_id_++, detections[j], features[j], n_init_);
            if (n_init_ == 1) {
                new_track.state = TrackState::Confirmed;
            }
            new_tracks.push_back(std::move(new_track));
        }
    }

    tracks_ = std::move(new_tracks);

    // 返回 confirmed 轨迹
    std::vector<Track> results;
    for (const auto& track : tracks_) {
        if (track.state == TrackState::Confirmed) {
            results.push_back(track);
        }
    }
    return results;
}

void DeepSortTracker::_match(
    const std::vector<cv::Rect_<float>>& detections,
    const std::vector<std::vector<float>>& features,
    std::vector<std::pair<size_t, size_t>>& matches,
    std::vector<size_t>& unmatched_tracks,
    std::vector<size_t>& unmatched_dets) {

    matches.clear();
    unmatched_tracks.clear();
    unmatched_dets.clear();

    if (tracks_.empty() || detections.empty()) {
        for (size_t i = 0; i < tracks_.size(); ++i) unmatched_tracks.push_back(i);
        for (size_t j = 0; j < detections.size(); ++j) unmatched_dets.push_back(j);
        return;
    }

    // 构建代价矩阵 (tracks x detections)
    size_t num_tracks = tracks_.size();
    size_t num_dets = detections.size();
    cv::Mat cost_matrix = cv::Mat::zeros((int)num_tracks, (int)num_dets, CV_32F);

    for (size_t i = 0; i < num_tracks; ++i) {
        for (size_t j = 0; j < num_dets; ++j) {
            // 计算余弦距离
            float cos_dist = CosineLoss(tracks_[i].feature, features[j]);
            // 计算 1 - IoU
            float iou_dist = 1.0f - CalculateIoU(tracks_[i].box, detections[j]);
            // 融合策略：可调整权重
            cost_matrix.at<float>((int)i, (int)j) = iou_dist;
        }
    }

    // ✅ 使用通用匈牙利匹配函数（带门控）
    HungarianAlgorithm(
        cost_matrix,
        max_cosine_distance_,  // 门控阈值（也可用独立的匹配阈值）
        matches,
        unmatched_tracks,
        unmatched_dets
    );
}

std::vector<cv::Rect_<float>> DeepSortTracker::_get_predicted_boxes() const {
    std::vector<cv::Rect_<float>> boxes;
    for (const auto& track : tracks_) {
        boxes.push_back(track.box);
    }
    return boxes;
}