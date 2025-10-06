#include "kalman.h"

KalmanFilter::KalmanFilter(
    float q_pos,
    float q_vel,
    float r,
    float init_P
)
    : q_pos_(q_pos),
      q_vel_(q_vel),
      r_(r),
      init_P_(init_P) {
    
    // 初始化状态向量 (8维)
    x_ = Eigen::VectorXf::Zero(8);

    // 初始化协方差矩阵 P (8x8)
    P_ = Eigen::MatrixXf::Identity(8, 8) * init_P_;

    // 状态转移矩阵 F (dt = 1)
    F_ = Eigen::MatrixXf::Identity(8, 8);
    F_.block<4, 4>(0, 4) = Eigen::MatrixXf::Identity(4, 4);

    // 观测矩阵 H
    H_ = Eigen::MatrixXf::Zero(4, 8);
    H_.block<4, 4>(0, 0) = Eigen::MatrixXf::Identity(4, 4);

    // 构建过程噪声协方差 Q
    Q_ = Eigen::MatrixXf::Zero(8, 8);
    float q_pos2 = q_pos_ * q_pos_;
    float q_vel2 = q_vel_ * q_vel_;
    Q_.diagonal() << q_pos2, q_pos2, q_pos2, q_pos2,
                     q_vel2, q_vel2, q_vel2, q_vel2;

    // 构建观测噪声协方差 R
    float r2 = r_ * r_;
    R_ = Eigen::MatrixXf::Identity(4, 4) * r2;
}

Eigen::Vector4f KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
    return x_.head<4>();
}

void KalmanFilter::update(const Eigen::Vector4f& z) {
    Eigen::MatrixXf y = z - H_ * x_; // 残差
    Eigen::MatrixXf S = H_ * P_ * H_.transpose() + R_; // 创新协方差
    Eigen::MatrixXf K = P_ * H_.transpose() * S.inverse(); // 卡尔曼增益

    // 更新状态
    x_ = x_ + K * y;

    // ✅ 使用 Joseph form 更新协方差（数值稳定！）
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(8, 8);
    P_ = (I - K * H_) * P_ * (I - K * H_).transpose();
    P_ += K * R_ * K.transpose();

    // 可选：强制对称（应对浮点误差）
    P_ = (P_ + P_.transpose()) * 0.5f;
}