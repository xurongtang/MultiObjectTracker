#include "kalmanfilter/kalman.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

int main() {
    std::cout << "=== KalmanFilter 单轨迹测试 ===\n\n";

    // 创建 Kalman 滤波器（使用默认 DeepSORT 超参数）
    KalmanFilter kf;

    // 模拟一条真实轨迹：匀速运动的目标
    // 真实状态: [u, v, gamma, h] = [100 + t*2, 200 + t*1, 0.5, 100]
    std::vector<Eigen::Vector4f> ground_truth;
    for (int t = 0; t < 10; ++t) {
        float u = 100.0f + t * 2.0f;
        float v = 200.0f + t * 1.0f;
        float gamma = 0.5f;
        float h = 100.0f;
        ground_truth.push_back(Eigen::Vector4f(u, v, gamma, h));
    }

    // 模拟检测：前3帧有检测，中间3帧丢失（遮挡），后4帧恢复
    std::vector<bool> has_detection = {
        true, true, true,   // 帧 0-2：正常
        false, false, false, // 帧 3-5：遮挡，无检测
        true, true, true, true // 帧 6-9：恢复
    };

    // 添加轻微噪声到检测（模拟检测器误差）
    auto add_noise = [](const Eigen::Vector4f& z) {
        Eigen::Vector4f noisy = z;
        noisy[0] += (rand() % 10 - 5) * 0.1f; // u ±0.5
        noisy[1] += (rand() % 10 - 5) * 0.1f; // v ±0.5
        return noisy;
    };

    std::cout << "帧 | 状态       | 预测 [u,v,γ,h]     | 观测 [u,v,γ,h]     | P_diag(u,v,γ,h)\n";
    std::cout << "--------------------------------------------------------------------------\n";

    for (int t = 0; t < 10; ++t) {
        // if (t == 0) {
        //     std::cout << "P_ after update:\n" << kf.getCovariance() << "\n";
        // }
        // Step 1: 预测
        Eigen::Vector4f prediction = kf.predict();

        // Step 2: 判断是否有检测
        Eigen::Vector4f measurement;
        bool matched = false;
        std::string status = "未匹配";
        if (has_detection[t]) {
            measurement = add_noise(ground_truth[t]);
            kf.update(measurement);
            matched = true;
            status = "已匹配";
        }

        // 打印结果
        std::cout << std::setw(2) << t << " | "
                  << std::setw(10) << status << " | "
                  << "[" << std::fixed << std::setprecision(1)
                  << prediction[0] << ", " << prediction[1] << ", "
                  << prediction[2] << ", " << prediction[3] << "] | ";

        if (matched) {
            std::cout << "[" << measurement[0] << ", " << measurement[1] << ", "
                      << measurement[2] << ", " << measurement[3] << "] | ";
        } else {
            std::cout << "                -                | ";
        }

        // 打印协方差对角线（前4维），用科学计数法或更高精度
        std::cout << "[";
        for (int i = 0; i < 4; ++i) {
            std::cout << std::scientific << std::setprecision(2) << kf.getCovariance()(i, i);
            if (i < 3) std::cout << ", ";
        }
        std::cout << "]\n";

        // 验证：匹配后预测应接近观测
        if (t == 2 && matched) {
            assert(std::abs(prediction[0] - measurement[0]) < 10.0f);
            assert(std::abs(prediction[1] - measurement[1]) < 10.0f);
        }
    }

    std::cout << "\n✅ 测试完成！观察以下行为是否符合预期：\n";
    std::cout << "1. 初始几帧：预测逐渐贴近观测\n";
    std::cout << "2. 帧 3-5（未匹配）：预测继续外推，协方差 P_diag 显著增大\n";
    std::cout << "3. 帧 6（恢复匹配）：滤波器快速收敛回真实轨迹\n";

    return 0;
}