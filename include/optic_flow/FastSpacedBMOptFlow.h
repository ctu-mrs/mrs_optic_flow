#ifndef FASTBMOF_H_
#define FASTBMOF_H_
#include <opencv2/core/core.hpp>

void FastSpacedBMOptFlow(cv::InputArray _imPrev, cv::InputArray _imCurr,
                         cv::OutputArray _imOutX, cv::OutputArray _imOutY,
                         int blockSize,
                         int blockStep,
                         int scanRadius,
                         double cx, double cy,double fx,
                         double fy, double k1, double k2, double k3, double p1, double p2,
                         signed char &outX,
                         signed char &outY);

void ResetCudaDevice();

#endif  // FASTBMOF_H_
