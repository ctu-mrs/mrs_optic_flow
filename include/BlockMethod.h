#ifndef BLOCKMETHOD_H
#define BLOCKMETHOD_H

#include <opencv2/highgui/highgui.hpp>
#include <OpticFlowCalc.h>

class BlockMethod : public OpticFlowCalc {
private:
  cv::Mat imOrigScaled;
  // cv::Mat imCurr;
  cv::Mat imDiff;
  cv::Mat imPrev;

  cv::Mat imView;
  cv::Mat absDiffsMat;
  cv::Mat absDiffsMatSubpix;

  cv::Point2i midPoint;

  int ScaleFactor;

  int samplePointSize;

  int frameSize;
  int maxSamplesSide;

  int scanRadius;
  int scanDiameter;
  int scanCount;
  int stepSize;

  double currentRange;

  int *xHist;
  int *yHist;

  cv::Point2d Refine(cv::Mat imCurr, cv::Mat imPrev, cv::Point2i fullpixFlow, int passes);

public:
  BlockMethod(int frameSize, int samplePointSize, int scanRadius, int scanDiameter, int scanCount, int stepSize);

  std::vector<cv::Point2d> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr);
};

#endif  // BLOCKMETHOD_H
