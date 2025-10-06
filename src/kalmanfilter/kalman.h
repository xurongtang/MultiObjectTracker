#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>

class KalmanFilter {
    public:
        // 构造函数：可选传入超参数，使用默认值（DeepSORT 推荐值）
        KalmanFilter(
            float q_pos = 1.0f / 20.0f,
            float q_vel = 1.0f / 160.0f,
            float r = 0.05f,
            float init_P = 1000.0f
        );

        ~KalmanFilter() = default;

        Eigen::Vector4f predict();
        void update(const Eigen::Vector4f& z);

        const Eigen::VectorXf& getState() const { return x_; }
        const Eigen::MatrixXf& getCovariance() const { return P_; }

    private:
        // === 超参数（全部在 private 中定义）===
        float q_pos_;      // 位置过程噪声标准差
        float q_vel_;      // 速度过程噪声标准差
        float r_;          // 观测噪声标准差
        float init_P_;     // 初始状态协方差（标量，用于对角初始化）

        // === Kalman 滤波器内部变量 ===
        Eigen::MatrixXf F_; // 8x8   状态转移矩阵
        Eigen::MatrixXf H_; // 4x8    观测矩阵
        Eigen::MatrixXf Q_; // 8x8   位置过程噪声协方差
        Eigen::MatrixXf R_; // 4x4    观测噪声协方差
        Eigen::VectorXf x_; // 8x1   状态向量
        Eigen::MatrixXf P_; // 8x8   状态协方差
};

#endif