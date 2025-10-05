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
    int batch = shape[0];
    int channel = shape[1];
    int height = shape[2];
    int width = shape[3];

    // std::cout << "📊 Input image size: " << inputs[0].cols << " x " << inputs[0].rows << std::endl;
    // std::cout << "📊 Model expects: " << width << "x" << height << " (WxH)" << std::endl;

    if (static_cast<int>(inputs.size()) > batch) {
        std::cerr << "❌ Input batch size exceeds model capacity!" << std::endl;
        return -1;
    }

    // ✅ 关键修复：使用 CAFFE (NCHW) 格式
    MNN::Tensor inputUser(m_inputTensor, MNN::Tensor::CAFFE); // NCHW
    auto inputPtr = inputUser.host<float>();

    // 配置预处理（归一化到 [0,1]，BGR→RGB）
    MNN::CV::ImageProcess::Config config;
    config.filterType = MNN::CV::BILINEAR;
    config.sourceFormat = MNN::CV::BGR;   // OpenCV 默认 BGR
    config.destFormat = MNN::CV::RGB;     // 模型需要 RGB

    for (int i = 0; i < 3; ++i) {
        config.mean[i]   = mnn_mean[i];
        config.normal[i] = 1.0f / (mnn_std[i] * 255.0f);
    }
    
    auto process = std::shared_ptr<MNN::CV::ImageProcess>(MNN::CV::ImageProcess::create(config));

    for (size_t i = 0; i < inputs.size(); ++i) {
        cv::Mat& img = inputs[i];
        if (img.empty()) continue;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(width, height)); // 注意：Size(width, height)

        // ✅ 关键：使用 CAFFE 格式，convert 会自动输出 NCHW
        float* batchPtr = inputPtr + i * channel * height * width;
        process->convert(
            resized.data,    // BGR input
            width, height,   // 输入宽高
            width * 3,       // 输入 stride (BGR)
            batchPtr,        // 输出指针 (NCHW)
            width, height    // 输出尺寸
        );
    }

    m_inputTensor->copyFromHostTensor(&inputUser);
    m_net->runSession(m_session);

    // 获取并打印输出
    outputs.clear();
    output_shapes.clear();
    
    auto outputNames = m_net->getSessionOutputAll(m_session);
    // std::cout << "📤 Number of output tensors: " << outputNames.size() << std::endl;

    for (const auto& pair : outputNames) {
        const std::string& name = pair.first;
        auto outputTensor = m_net->getSessionOutput(m_session, name.c_str());
        
        // 打印输出形状
        auto outShape = outputTensor->shape();
        output_shapes.push_back({name, outShape});
        size_t total = 1;
        // std::cout << "--- Output[" << name << "] ---\nShape: ";
        for (auto s : outShape) {
            // std::cout << s << " ";
            total *= s;
        }
        // std::cout << "\nTotal elements: " << total << std::endl;

        // 拷贝到 host
        MNN::Tensor outputUser(outputTensor, MNN::Tensor::CAFFE);
        outputTensor->copyToHostTensor(&outputUser);

        std::vector<float> outVec(outputUser.host<float>(), outputUser.host<float>() + total);
        outputs.push_back(std::move(outVec));

        // 打印前10个值
        // std::cout << "First 10 values: ";
        // for (int j = 0; j < std::min(10, (int)outVec.size()); ++j) {
        //     std::cout << outVec[j] << " ";
        // }
        // std::cout << "\n" << std::endl;
    }

    return 0;
}