#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H
#include <math.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>

struct StatData{
    uint num;
    float mean;
    float stdDev;

    float meanX;
    float stdDevX;

    float meanY;
    float stdDevY;
};

struct SpeedBox
{
    ros::Time time;
    cv::Point2f speed;
    cv::Point2f odomSpeed;
};


void rotate2d(double &x, double &y, double alpha);
void rotate2d(cv::Point2f &pt, double alpha);
cv::Point2f pointMean(std::vector<cv::Point2f> pts);
float getDistSq(cv::Point2f p1,cv::Point2f p2);
float getNormSq(cv::Point2f p1);
cv::Point2f twoPointMean(cv::Point2f p1, cv::Point2f p2);
double calcMean(std::vector<double> pts);
cv::Point2f allsacMean(std::vector<cv::Point2f> pts, float thresholdRadius_sq, int *chosen);
double allsacMean(std::vector<double> pts, float thresholdRadius, int *chosen);
void multiplyAllPts(std::vector<cv::Point2f> &v, float mulx, float muly);
void multiplyAllPts(std::vector<double> &v,float mul);
void rotateAllPts(std::vector<cv::Point2f> &v,double alpha);
void addToAll(std::vector<cv::Point2f> &v,float adx, float ady);
cv::Point2f ransacMean(std::vector<cv::Point2f> pts, int numOfChosen, float thresholdRadius_sq, int numOfIterations);
std::vector<cv::Point2f> getOnlyInAbsBound(std::vector<cv::Point2f> v, float up);
std::vector<double> getOnlyInAbsBound(std::vector<double> v,double up);
std::vector<cv::Point2f> removeNanPoints(std::vector<cv::Point2f> v);
std::vector<double> removeNanPoints(std::vector<double> v);
std::vector<cv::Point2f> getOnlyInRadiusFromTruth(cv::Point2f truth,std::vector<cv::Point2f> v,float rad);
float absf(float x);
double absd(double x);
StatData analyzeSpeeds(ros::Time fromTime, std::vector<SpeedBox> speeds);
std::vector<cv::Point2f> estimateTranRotVvel(std::vector<cv::Point2f> vectors, double a, double fx, double fy, double range, double allsac_radius, double duration, double max_vert_speed, double max_yaw_speed);


#endif // UTILITYFUNCTIONS_H
