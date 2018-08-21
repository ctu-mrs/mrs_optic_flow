#ifndef FFTMETHOD_H
#define FFTMETHOD_H

//#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"
#include <iostream>
#include <string.h>
#include "optic_flow/utilityFunctions.h"

class FftMethod : public OpticFlowCalc {
private:
  int frameSize;
  int samplePointSize;

  int imCenterX, imCenterY;  // center of original image
  int xi, yi;                // frame corner coordinates

  std::vector<cv::Point2f> speeds;

  int sqNum;

  cv::Point2d shift;

  bool first;
  bool raw_enable;
  bool rot_corr_enable;
  bool tilt_corr_enable;

  bool storeVideo;

  cv::VideoWriter outputVideo;

public:
  FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
            bool i_tilt_corr_enable, std::string *videoPath, int videoFPS);

  std::vector<cv::Point2f> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr);
};

#endif  // FFTMETHOD_H
