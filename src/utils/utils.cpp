#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// =================================================//
// ===============函数及结构体声明开始===============//

// IoU计算
int CalculateIoU(cv::Rect_<float> box1, cv::Rect_<float> box2);

// 余弦损失度
float CosineLoss(cv::Mat& vec1, cv::Mat& vec2);

// 

//

//


// ===============函数及结构体声明结束===============//
// =================================================//


// ===============函数及结构体实现===============//