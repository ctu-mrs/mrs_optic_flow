#ifndef SCALEROTATIONESTIMATOR_H
#define SCALEROTATIONESTIMATOR_H

#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <cmath>
#include <image_transport/image_transport.h>

class scaleRotationEstimator {

public:
  scaleRotationEstimator(int resolution, double m, bool i_storeVideo, std::string *videoPath, int videoFPS);  // accepts only rectangular images
  cv::Point2d processImage(cv::Mat imCurr, bool gui, bool debug);

private:
  cv::Mat prevIm_F32;
  cv::Mat tempIm;
  cv::Mat tempIm_F32;
  cv::Point2f center;
  double   optimM, Ky;
  bool     first;
  IplImage ipl_ta, ipl_tb;
  bool     storeVideo;

  cv::VideoWriter outputVideo;

  cv::Mat magLL_prev;
  cv::Mat magLL;
  int     resolution;
};

#endif  // SCALEROTATIONESTIMATOR_H
