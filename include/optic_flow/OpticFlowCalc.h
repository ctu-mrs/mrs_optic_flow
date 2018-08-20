#ifndef OPTICFLOWCALC_H
#define OPTICFLOWCALC_H

#include <cv_bridge/cv_bridge.h>

class OpticFlowCalc {

public:
  virtual std::vector<cv::Point2f> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr) {
  }

  void setImPrev(cv::Mat imPrev_t) {
    imPrev = imPrev_t;
  }

protected:
  cv::Mat     imPrev, imCurr, imView;
  cv::Point2i midPoint;
  double      max_px_speed_sq;
};

#endif  // OPTICFLOWCALC_H
