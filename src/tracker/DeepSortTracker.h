#include <iostream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

class DeepSortTracker
{
    public:
        DeepSortTracker();

    private:
        void calculate_iou();
}