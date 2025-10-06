// MNNInfer.cpp (修复版)
#include "mnnInfer.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

MNNInfer::MNNInfer(std::string modelPath,float mean_[3],float std_[3])
    : m_modelPath(modelPath) {
        for(int i = 0; i < 3; i++)
        {
            mnn_mean[i] = mean_[i];
            mnn_std[i] = std_[i];
        }
    }

MNNInfer::~MNNInfer() {
    if (m_session) {
        m_net->releaseSession(m_session);
    }
}

int MNNInfer::loadModel() {
    m_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(m_modelPath.c_str()));
    if (!m_net) {
        std::cerr << "❌ Failed to load MNN model: " << m_modelPath << std::endl;
        return -1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;

    m_session = m_net->createSession(config);
    if (!m_session) {
        std::cerr << "❌ Failed to create MNN session." << std::endl;
        return -1;
    }

    // 获取输入张量
    auto inputTensors = m_net->getSessionInputAll(m_session);
    if (inputTensors.empty()) {
        std::cerr << "❌ No input tensor found!" << std::endl;
        return -1;
    }
    std::string inputName = inputTensors.begin()->first;
    m_inputTensor = inputTensors.begin()->second;
    
    if (!m_inputTensor) {
        std::cerr << "❌ Failed to get input tensor." << std::endl;
        return -1;
    }

    // 打印输入信息
    auto shape = m_inputTensor->shape();
    std::cout << "✅ Model loaded. Input shape (NCHW): ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

int MNNInfer::runInference(std::vector<cv::Mat> &inputs, std::vector<std::vector<float>> &outputs) {
    if (!m_session || !m_inputTensor) {
        std::cerr << "❌ Model not loaded!" << std::endl;
        return -1;
    }
    if (inputs.empty()) {
        std::cerr << "❌ Input images is empty!" << std::endl;
        return -1;
    }

    auto shape = m_inputTensor->shape();
    int model_batch = shape[0];   // 通常是 1
    int channel = shape[1];
    int height = shape[2];
    int width = shape[3];

    // std::cout << "📊 Model input shape: " << shape[0] << "x" << shape[1] 
    //           << "x" << shape[2] << "x" << shape[3] << std::endl;

    // ✅ 关键修改：不要检查 batch size，而是循环处理每个输入
    // 即使 model_batch=1，我们也逐个推理

    outputs.clear();
    output_shapes.clear();

    // 获取输出 tensor 信息（假设单输出）
    auto outputNames = m_net->getSessionOutputAll(m_session);
    if (outputNames.empty()) {
        std::cerr << "❌ No output tensor!" << std::endl;
        return -1;
    }
    const std::string& outName = outputNames.begin()->first;

    // 预处理配置
    MNN::CV::ImageProcess::Config config;
    config.filterType = MNN::CV::BILINEAR;
    config.sourceFormat = MNN::CV::BGR;
    config.destFormat = MNN::CV::RGB;
    for (int i = 0; i < 3; ++i) {
        config.mean[i]   = mnn_mean[i];
        config.normal[i] = 1.0f / (mnn_std[i] * 255.0f);
    }
    auto process = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(config));

    // ✅ 逐个处理每个输入图像
    for (size_t i = 0; i < inputs.size(); ++i) {
        cv::Mat& img = inputs[i];
        if (img.empty()) {
            // 空图：填充零特征
            auto outTensor = m_net->getSessionOutput(m_session, outName.c_str());
            size_t feat_dim = 1;
            for (auto s : outTensor->shape()) feat_dim *= s;
            outputs.push_back(std::vector<float>(feat_dim, 0.0f));
            continue;
        }

        // 调整尺寸
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(width, height)); // 注意：Size(宽, 高)

        // 准备输入（NCHW，batch=1）
        MNN::Tensor inputUser(m_inputTensor, MNN::Tensor::CAFFE);
        process->convert(
            resized.data, width, height, width * 3,
            inputUser.host<float>(), width, height
        );
        m_inputTensor->copyFromHostTensor(&inputUser);

        // 推理
        m_net->runSession(m_session);

        // 获取输出
        auto outputTensor = m_net->getSessionOutput(m_session, outName.c_str());
        MNN::Tensor outputUser(outputTensor, MNN::Tensor::CAFFE);
        outputTensor->copyToHostTensor(&outputUser);

        auto outShape = outputTensor->shape();
        output_shapes.push_back({outName, outShape});

        size_t total = 1;
        for (auto s : outputTensor->shape()) 
            total *= s;
        std::vector<float> feat(outputUser.host<float>(), outputUser.host<float>() + total);
        outputs.push_back(std::move(feat));
    }

    return 0;
}