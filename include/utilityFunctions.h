#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H

#include <math.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

struct StatData
{
  uint   num;
  double mean;
  double stdDev;

  double meanX;
  double stdDevX;

  double meanY;
  double stdDevY;
};

struct SpeedBox
{
  ros::Time   time;
  cv::Point2d speed;
  cv::Point2d odometry_speed;
};

void                     rotate2d(double &x, double &y, double alpha);
void                     rotate2d(cv::Point2d &pt, double alpha);
cv::Point2d              pointMean(std::vector<cv::Point2d> pts);
double                   getDistSq(cv::Point2d p1, cv::Point2d p2);
double                   getNormSq(cv::Point2d p1);
cv::Point2d              twoPointMean(cv::Point2d p1, cv::Point2d p2);
double                   calcMean(std::vector<double> pts);
cv::Point2d              allsacMean(std::vector<cv::Point2d> pts, double thresholdRadius_sq, int *chosen);
double                   allsacMean(std::vector<double> pts, double thresholdRadius, int *chosen);
std::vector<cv::Point2d> multiplyAllPts(std::vector<cv::Point2d> &v, double mulx, double muly, bool affect_input = true);
void                     multiplyAllPts(std::vector<double> &v, double mul);
void                     rotateAllPts(std::vector<cv::Point2d> &v, double alpha);
void                     addToAll(std::vector<cv::Point2d> &v, double adx, double ady);
cv::Point2d              ransacMean(std::vector<cv::Point2d> pts, int numOfChosen, double thresholdRadius_sq, int numOfIterations);
std::vector<cv::Point2d> getOnlyInAbsBound(std::vector<cv::Point2d> v, double up);
std::vector<double>      getOnlyInAbsBound(std::vector<double> v, double up);
std::vector<cv::Point2d> removeNanPoints(std::vector<cv::Point2d> v);
std::vector<double>      removeNanPoints(std::vector<double> v);
std::vector<cv::Point2d> getOnlyInRadiusFromExpected(cv::Point2d expected, std::vector<cv::Point2d> v, double rad);
double                   absf(double x);
double                   absd(double x);
StatData                 analyzeSpeeds(ros::Time fromTime, std::vector<SpeedBox> speeds);
std::vector<cv::Point2d> estimateTranRotVvel(std::vector<cv::Point2d> vectors, double a, double fx, double fy, double range, double allsac_radius,
                                             double duration, double max_vert_speed, double max_yaw_speed);

#endif  // UTILITYFUNCTIONS_H
