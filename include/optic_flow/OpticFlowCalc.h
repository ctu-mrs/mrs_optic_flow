#ifndef OPTICFLOWCALC_H
#define OPTICFLOWCALC_H

#include <cv_bridge/cv_bridge.h>

class OpticFlowCalc {

public:
  virtual std::vector<cv::Point2f> processImage([[maybe_unused]] cv::Mat imCurr, [[maybe_unused]] bool gui, [[maybe_unused]] bool debug,
                                                [[maybe_unused]] cv::Point midPoint_t, [[maybe_unused]] double yaw_angle,
                                                [[maybe_unused]] cv::Point2d tiltCorr) = 0;

  void setImPrev(cv::Mat imPrev_t) {
    imPrev = imPrev_t;
  }

protected:
  cv::Mat     imPrev, imCurr, imView;
  cv::Point2i midPoint;
  double      max_px_speed_sq;
};

#endif  // OPTICFLOWCALC_H
