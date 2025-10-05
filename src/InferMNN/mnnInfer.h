// MNNInfer.h
#ifndef MNN_INFER_H
#define MNN_INFER_H

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

class MNNInfer 
{
    public:
        MNNInfer(std::string modelPath); 
        ~MNNInfer();

    public:
        int loadModel();
        int runInference(std::vector<cv::Mat> &inputs, std::vector<std::vector<float>> &outputs);

    private:
        std::string m_modelPath;
        std::shared_ptr<MNN::Interpreter> m_net;
        MNN::Session* m_session = nullptr;
        MNN::Tensor* m_inputTensor = nullptr;
};

#endif // MNN_INFER_H