#ifndef FFTMETHOD_H
#define FFTMETHOD_H

//#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <image_transport/image_transport.h>
#include "optic_flow/OpticFlowCalc.h"
#include <iostream>
#include <string.h>
#include "optic_flow/utilityFunctions.h"

class FftMethod : public OpticFlowCalc {
private:
    cv::UMat usrc1, usrc2;
    cv::UMat window1, window2;
    cv::UMat FFT1, FFT2, P, Pm, C;

  int frameSize;
  int samplePointSize;

  int imCenterX, imCenterY;  // center of original image
  int xi, yi;                // frame corner coordinates
  double fx,fy;

  std::vector<cv::Point2d> speeds;

  int sqNum;

  cv::Point2d shift;
  cv::Point2d shift_raw;

  bool first;
  bool raw_enable;
  bool rot_corr_enable;
  bool tilt_corr_enable;

  bool storeVideo;

  cv::VideoWriter outputVideo;

  std::vector<cv::Point2d> phaseCorrelateField(cv::Mat &_src1, cv::Mat &_src2, unsigned int X,unsigned int Y,
                                     CV_OUT double* response = 0);

public:
  FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
            bool i_tilt_corr_enable, std::string *videoPath, int videoFPS);

  std::vector<cv::Point2d> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center, cv::Point2d tiltCorr_dynamic, std::vector<cv::Point2d> &raw_output, double i_fx, double i_fy);
};

#endif  // FFTMETHOD_H
