#include "InferMNN/mnnInfer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // 1. 解析命令行参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.mnn> <input_image.jpg>\n";
        std::cerr << "Example: " << argv[0] << " ReID/ReID.mnn test.jpg\n";
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    // 2. 创建推理器并加载模型
    MNNInfer infer(modelPath);
    if (infer.loadModel() != 0) {
        std::cerr << "Failed to load model: " << modelPath << std::endl;
        return -1;
    }

    // 3. 读取测试图像
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return -1;
    }

    std::cout << "Input image size: " << img.cols << " x " << img.rows << std::endl;

    // 4. 准备输入（支持 batch，这里 batch=1）
    std::vector<cv::Mat> inputs = {img};
    std::vector<std::vector<float>> outputs;

    // 5. 执行推理
    if (infer.runInference(inputs, outputs) != 0) {
        std::cerr << "Inference failed!" << std::endl;
        return -1;
    }

    // 6. 打印输出结果
    std::cout << "\n✅ Inference succeeded!\n";
    std::cout << "Number of output tensors: " << outputs.size() << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "\n--- Output[" << i << "] ---\n";
        std::cout << "Size: " << outputs[i].size() << " elements\n";

        // 打印前10个值（避免输出太长）
        int printCount = std::min(10, (int)outputs[i].size());
        std::cout << "First " << printCount << " values: ";
        for (int j = 0; j < printCount; ++j) {
            std::cout << outputs[i][j] << " ";
        }
        std::cout << "\n";

        // 如果是 ReID 特征（如 128 维），可计算 L2 norm
        if (outputs[i].size() == 128 || outputs[i].size() == 256 || outputs[i].size() == 512) {
            float norm = 0.0f;
            for (float v : outputs[i]) norm += v * v;
            norm = std::sqrt(norm);
            std::cout << "L2 Norm: " << norm << " (should be ~1.0 if normalized)\n";
        }
    }

    return 0;
}