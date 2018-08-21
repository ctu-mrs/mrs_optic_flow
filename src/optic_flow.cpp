//#define OPENCL_ENABLE

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <string>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Range.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32.h>

using namespace std;

//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>

#include "optic_flow/OpticFlowCalc.h"
#include "optic_flow/BlockMethod.h"
#include "optic_flow/FftMethod.h"
#include "optic_flow/utilityFunctions.h"
#include "optic_flow/scaleRotationEstimator.h"

#ifdef OPENCL_ENABLE

#include "optic_flow/FastSpacedBMMethod_OCL.h"
#include "optic_flow/FastSpacedBMOptFlow.h"
#include <opencv2/gpu/gpu.hpp>

#endif

namespace enc = sensor_msgs::image_encodings;

namespace optic_flow
{

//{ class OpticFlow

struct PointValue
{
  int         value;
  cv::Point2i location;
};

class OpticFlow : public nodelet::Nodelet {

public:
  virtual void    onInit();
  ros::NodeHandle nh_;

private:
  void callbackRangefinder(const sensor_msgs::Range& range_msg);
  void callbackImu(const sensor_msgs::Imu imu_msg);
  void callbackOdometry(const nav_msgs::Odometry odom_msg);
  void callbackImageCompressed(const sensor_msgs::CompressedImageConstPtr& image_msg);
  void callbackImageRaw(const sensor_msgs::ImageConstPtr& image_msg);
  void callbackCameraInfo(const sensor_msgs::CameraInfo cam_info);

private:
  void processImage(const cv_bridge::CvImagePtr image);

private:
  ros::Timer cam_init_timer;
  void       camInitTimer(const ros::TimerEvent& event);

private:
  bool first;

  ros::Time RangeRecTime;

  ros::Subscriber ImageSubscriber;
  ros::Subscriber RangeSubscriber;
  ros::Publisher  VelocityPublisher;
  ros::Publisher  VelocitySDPublisher;
  ros::Publisher  VelocityRawPublisher;
  ros::Publisher  MaxAllowedVelocityPublisher;
  ros::Publisher  TiltCorrectionPublisher;
  ros::Publisher  AllsacChosenPublisher;

  ros::Subscriber CamInfoSubscriber;
  ros::Subscriber TiltSubscriber;
  ros::Subscriber ImuSubscriber;

  bool got_camera_info      = false;
  bool got_raw_image        = false;
  bool got_compressed_image = false;

  cv::Mat imOrigScaled;
  cv::Mat imCurr;
  cv::Mat imPrev;

  double vxm, vym, vam;

  ros::Time      begin;
  ros::Duration  dur;
  OpticFlowCalc* processClass;

  double roll, pitch, yaw;

  /*double ypr_time;
    double roll_old, pitch_old, yaw_old;
    double ypr_old_time;

    double roll_im_old, pitch_im_old, yaw_im_old;
    double roll_dif, pitch_dif, yaw_dif;*/

  // scale to altitude..
  double d1, t12;

  // Input arguments
  bool DEBUG;
  bool silent_debug;
  bool storeVideo;
  // std::vector<double> camRot;
  double gamma;  // rotation of camera in the helicopter frame (positive)

  int expectedWidth;
  int ScaleFactor;

  int frameSize;
  int samplePointSize;

  int scanRadius;
  int scanDiameter;
  int scanCount;
  int stepSize;

  double cx, cy, fx, fy, s;
  double k1, k2, p1, p2, k3;
  bool   negativeCamInfo;

  bool gui;
  int  method;

  int numberOfBins;

  int         RansacNumOfChosen;
  int         RansacNumOfIter;
  float       RansacThresholdRadSq;
  std::string filterMethod;

  bool                    rotation_correction_enable, tilt_correction_enable, raw_enable;
  std::string             ang_rate_source;
  bool                    scaleRot_enable;
  double                  scaleRot_mag;
  std::string             scale_rot_output;
  scaleRotationEstimator* srEstimator;
  std::string             d3d_method;

  // Ranger & odom vars
  double trueRange;

  double max_freq;
  double max_px_speed_t;
  double maxSpeed;
  double maxAccel;
  double maxVertSpeed;
  double maxYawSpeed;

  bool applyAbsBounding;
  bool applyRelBounding;

  cv::Point3d angular_velocity;

  cv::Point2f odomSpeed;
  ros::Time   odomSpeedTime;
  float       speed_noise;

  std::vector<SpeedBox> lastSpeeds;
  int                   lastSpeedsSize;
  double                analyseDuration;

private:
  bool is_initialized = false;
};

//}

//{ onInit()

void OpticFlow::onInit() {

  ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  first = true;

  // LOAD PARAMETERS
  nh_.param("DEBUG", DEBUG, bool(false));

  nh_.param("method", method, int(0));

  if (method < 3 || method > 5) {
    ROS_ERROR("[OpticFlow]: No such OpticFlow calculation method. Available: 3 = BM on CPU, 4 = FFT on CPU, 5 = BM on GPU via OpenCL");
  }

  // optic flow parameters
  nh_.param("ScanRadius", scanRadius, int(8));
  nh_.param("FrameSize", frameSize, int(64));
  nh_.param("SamplePointSize", samplePointSize, int(8));
  nh_.param("NumberOfBins", numberOfBins, int(20));

  nh_.param("StepSize", stepSize, int(0));

  nh_.param("gui", gui, bool(false));

  nh_.param("applyAbsBounding", applyAbsBounding, bool(true));
  nh_.param("applyRelBounding", applyRelBounding, bool(false));

  bool ImgCompressed;

  nh_.param("CameraImageCompressed", ImgCompressed, bool(false));

  nh_.param("ScaleFactor", ScaleFactor, int(1));

  nh_.param("RansacNumOfChosen", RansacNumOfChosen, int(2));
  nh_.param("RansacNumOfIter", RansacNumOfIter, int(5));

  float RansacThresholdRad;

  nh_.param("RansacThresholdRad", RansacThresholdRad, float(4));
  RansacThresholdRadSq = pow(RansacThresholdRad, 2);

  nh_.param("filterMethod", filterMethod, std::string("no parameter"));

  nh_.param("lastSpeedsSize", lastSpeedsSize, int(60));
  nh_.param("analyseDuration", analyseDuration, double(1));
  nh_.param("silentDebug", silent_debug, bool(false));

  nh_.param("rotation_correction_enable", rotation_correction_enable, bool(true));
  nh_.param("tilt_correction_enable", tilt_correction_enable, bool(true));
  nh_.param("ang_rate_source", ang_rate_source, std::string("no parameter"));
  nh_.param("raw_enable", raw_enable, bool(false));

  nh_.param("scale_rot_enable", scaleRot_enable, bool(false));
  nh_.param("scale_rot_mag", scaleRot_mag, double(40));
  nh_.param("scale_rot_output", scale_rot_output, std::string("no parameter"));
  nh_.param("d3d_method", d3d_method, std::string("no parameter"));

  if (scaleRot_enable && d3d_method.compare("advanced") != 0 && d3d_method.compare("logpol") != 0) {
    ROS_ERROR("[OpticFlow]: Wrong parameter 3d_method. Possible values: logpol, advanced. Entered: %s", d3d_method.c_str());
    exit(2);
  }

  nh_.param("maxFPS", max_freq, double(500));

  nh_.param("storeVideo", storeVideo, bool(false));
  std::string videoPath;
  nh_.param("videoPath", videoPath, std::string("no parameter"));
  if (storeVideo) {
    ROS_INFO("[OpticFlow]: Video path: %s", videoPath.c_str());
  }

  int videoFPS;
  nh_.param("videoFPS", videoFPS, int(30));

  if (filterMethod.compare("ransac") && RansacNumOfChosen != 2) {
    ROS_WARN("[OpticFlow]: When Allsac is enabled, the RansacNumOfChosen can be only 2.");
  }

  if (RAND_MAX < 100) {
    ROS_WARN("[OpticFlow]: Why is RAND_MAX set to only %d? Ransac in OpticFlow won't work properly!", RAND_MAX);
  }

  nh_.getParam("image_width", expectedWidth);

  if ((frameSize % 2) == 1) {
    frameSize--;
  }
  scanDiameter = (2 * scanRadius + 1);
  scanCount    = (scanDiameter * scanDiameter);

  // nh_.getParam("camera_rotation_matrix/data", camRot);
  nh_.getParam("alpha", gamma);

  nh_.getParam("max_px_speed", max_px_speed_t);
  nh_.getParam("max_horiz_speed", maxSpeed);
  nh_.getParam("max_vert_speed", maxVertSpeed);
  nh_.getParam("max_yaw_speed", maxYawSpeed);
  nh_.getParam("max_acceleration", maxAccel);
  nh_.getParam("speed_noise", speed_noise);

  ROS_INFO(
      "Loaded physical constraints:\n - maximal optic flow: %f\n - maximal horizontal speed: %f\n - maximal vertical speed: %f\n - maximal acceleration: "
      "%f\n - speed noise: %f\n - max yaw velocity: %f\n",
      max_px_speed_t, maxSpeed, maxVertSpeed, maxAccel, speed_noise, maxYawSpeed);

  CamInfoSubscriber = nh_.subscribe("camera_info", 1, &OpticFlow::callbackCameraInfo, this);
  got_camera_info   = false;
  negativeCamInfo   = false;

  switch (method) {
    case 3: {
      processClass = new BlockMethod(frameSize, samplePointSize, scanRadius, scanDiameter, scanCount, stepSize);
      break;
    }
    case 4: {
      processClass = new FftMethod(frameSize, samplePointSize, max_px_speed_t, storeVideo, raw_enable, rotation_correction_enable, tilt_correction_enable,
                                   &videoPath, videoFPS);
      break;
    }

#ifdef OPENCL_ENABLE
    case 5: {
      processClass = new FastSpacedBMMethod(samplePointSize, scanRadius, stepSize, cx, cy, fx, fy, k1, k2, k3, p1, p2, storeVideo, &videoPath);
      break;
    }
#endif
  }

  imPrev = cv::Mat(frameSize, frameSize, CV_8UC1);
  imPrev = cv::Scalar(0);
  processClass->setImPrev(imPrev);

  begin = ros::Time::now();

  // prepare scale rotation estimator
  if (scaleRot_enable && d3d_method.compare("logpol") == 0) {
    if (scale_rot_output.compare("velocity") != 0) {
      if (scale_rot_output.compare("altitude") != 0) {
        ROS_ERROR("[OpticFlow]: Wrong parameter scale_rot_output - possible choices: velocity, altitude. Entered: %s", scale_rot_output.c_str());
        exit(2);
      }
    }
    std::string sr_name = videoPath.append("_scale_rot.avi");
    srEstimator         = new scaleRotationEstimator(frameSize, scaleRot_mag, storeVideo, &sr_name, videoFPS);
  }

  // image_transport::ImageTransport iTran(node);
  //
  VelocityPublisher   = nh_.advertise<geometry_msgs::TwistStamped>("velocity", 1);
  VelocitySDPublisher = nh_.advertise<geometry_msgs::Vector3>("velocity_stddev", 1);
  if (raw_enable) {
    VelocityRawPublisher = nh_.advertise<geometry_msgs::TwistStamped>("velocity_raw", 1);
  }
  MaxAllowedVelocityPublisher = nh_.advertise<std_msgs::Float32>("max_velocity", 1);

  TiltCorrectionPublisher = nh_.advertise<geometry_msgs::Vector3>("tilt_correction", 1);  // just for testing
  AllsacChosenPublisher   = nh_.advertise<geometry_msgs::Vector3>("allsac_chosen", 1);    // just for testing

  // Camera info subscriber
  RangeSubscriber = nh_.subscribe("ranger", 1, &OpticFlow::callbackRangefinder, this);
  TiltSubscriber  = nh_.subscribe("odometry", 1, &OpticFlow::callbackOdometry, this);

  if (ImgCompressed) {
    ImageSubscriber = nh_.subscribe("camera", 1, &OpticFlow::callbackImageCompressed, this);
  } else {
    ImageSubscriber = nh_.subscribe("camera", 1, &OpticFlow::callbackImageRaw, this);
  }

  if (ang_rate_source.compare("imu") == 0) {
    ImuSubscriber = nh_.subscribe("imu", 1, &OpticFlow::callbackImu, this);
  } else {
    if (ang_rate_source.compare("odometry") != 0) {
      ROS_ERROR("[OpticFlow]: Wrong parameter ang_rate_source - possible choices: imu, odometry. Entered: %s", ang_rate_source.c_str());
      exit(2);
    }
  }

  // --------------------------------------------------------------
  // |                           timers                           |
  // --------------------------------------------------------------

  cam_init_timer = nh_.createTimer(ros::Rate(10), &OpticFlow::camInitTimer, this);

  is_initialized = true;
}

//}

// --------------------------------------------------------------
// |                           timers                           |
// --------------------------------------------------------------

//{ camInitTimer()

void OpticFlow::camInitTimer(const ros::TimerEvent& event) {

  ros::Time camera_info_timeout;

  if (!got_raw_image && !got_compressed_image) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera");
    camera_info_timeout = ros::Time::now();
    return;
  }

  if (!got_camera_info && (ros::Time::now() - camera_info_timeout).toSec() < 5.0) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera info");
    return;
  }

  if (!got_camera_info || negativeCamInfo) {

    ROS_WARN(
        "[OpticFlow]: missing camera calibration parameters! (nothing on camera_info topic/wrong calibration matricies). Loading default parameters");
    std::vector<double> camMat;
    nh_.getParam("camera_matrix/data", camMat);
    fx = camMat[0];
    cx = camMat[2];
    fy = camMat[4];
    cy = camMat[5];
    std::vector<double> distCoeffs;
    nh_.getParam("distortion_coefficients/data", distCoeffs);
    k1              = distCoeffs[0];
    k2              = distCoeffs[1];
    k3              = distCoeffs[4];
    p1              = distCoeffs[2];
    p2              = distCoeffs[3];
    got_camera_info = true;

  } else {
    ROS_INFO("[OpticFlow]: camera parameters loaded");
  }

  cam_init_timer.stop();
}

//}

// --------------------------------------------------------------
// |                          callbacks                         |
// --------------------------------------------------------------

//{ callbackRangefinder()

void OpticFlow::callbackRangefinder(const sensor_msgs::Range& range_msg) {

  if (!is_initialized)
    return;

  if (absf(range_msg.range) < 0.001) {
    return;
  }

  trueRange = range_msg.range;
}

//}

//{ callbackImu()

void OpticFlow::callbackImu(const sensor_msgs::Imu imu_msg) {

  if (!is_initialized)
    return;

  // angular rate source is imu aka gyro
  if (ang_rate_source.compare("imu") == 0) {
    angular_velocity = cv::Point3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z);
  }
}

//}

//{ callbackOdometry()

void OpticFlow::callbackOdometry(const nav_msgs::Odometry odom_msg) {

  if (!is_initialized)
    return;

  ROS_INFO_ONCE("[OpticFlow]: Receiving odometry");

  // roll_old = roll; pitch_old = pitch; yaw_old = yaw;
  // ypr_old_time = ypr_time;

  tf::Quaternion bt;
  tf::quaternionMsgToTF(odom_msg.pose.pose.orientation, bt);
  tf::Matrix3x3(bt).getRPY(roll, pitch, yaw);
  // ypr_time = odom_msg.header.stamp.toSec();

  // angular rate source is odometry
  if (ang_rate_source.compare("odometry") == 0) {
    angular_velocity = cv::Point3d(odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z);
  }

  odomSpeed     = cv::Point2f(odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y);
  odomSpeedTime = ros::Time::now();
}

//}

//{ callbackImageCompressed()

void OpticFlow::callbackImageCompressed(const sensor_msgs::CompressedImageConstPtr& image_msg) {

  if (!is_initialized)
    return;

  got_compressed_image = true;

  ros::Time nowTime = image_msg->header.stamp;

  if (!first && (nowTime - begin).toSec() < 1 / max_freq) {

    if (DEBUG) {
      ROS_INFO("[OpticFlow]: MAX frequency overrun (%f). Skipping...", (nowTime - begin).toSec());
    }

    return;
  }

  dur   = nowTime - begin;
  begin = nowTime;

  if (DEBUG) {
    ROS_INFO("[OpticFlow]: freq = %fHz", 1.0 / dur.toSec());
  }

  /* double odom_time_diff = image_msg->header.stamp.toSec() - ypr_time; */
  /* double odom_period    = ypr_time - ypr_old_time; */

  /* double yaw_curr = odom_time_diff * ((yaw - yaw_old) / (odom_period)) + yaw; */
  /* yaw_dif         = yaw_curr - yaw_im_old; */
  /* yaw_im_old      = yaw_curr; */

  /* double pitch_curr = odom_time_diff * ((pitch - pitch_old) / (odom_period)) + pitch; */
  /* pitch_dif         = pitch_curr - pitch_im_old; */
  /* ROS_INFO("[OpticFlow]: Pitch odom: %f pitch extra: %f pitch diff %f time diff: %f odom extra time: %f", yaw, yaw_curr, pitch_dif, odom_time_diff, */
  /*          odom_period); */
  /* pitch_im_old = pitch_curr; */

  /* double roll_curr = odom_time_diff * ((roll - roll_old) / (odom_period)) + roll; */
  /* roll_dif         = roll_curr - roll_im_old; */
  /* roll_im_old      = roll_curr; */

  cv_bridge::CvImagePtr image;
  image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
  processImage(image);
}

//}

//{ callbackImageRaw()

void OpticFlow::callbackImageRaw(const sensor_msgs::ImageConstPtr& image_msg) {

  if (!is_initialized)
    return;

  got_raw_image = true;

  ros::Time nowTime = image_msg->header.stamp;

  if (!first && (nowTime - begin).toSec() < 1 / max_freq) {
    if (DEBUG) {
      ROS_INFO("[OpticFlow]: MAX frequency overrun (%f). Skipping...", (nowTime - begin).toSec());
    }
    return;
  }

  dur   = nowTime - begin;
  begin = nowTime;
  if (DEBUG) {
    ROS_INFO("[OpticFlow]: freq = %fHz", 1.0 / dur.toSec());
  }

  cv_bridge::CvImagePtr image;
  image = cv_bridge::toCvCopy(image_msg, enc::BGR8);
  processImage(image);
}

//}

//{ callbackCameraInfo()

void OpticFlow::callbackCameraInfo(const sensor_msgs::CameraInfo cam_info) {

  if (!is_initialized)
    return;

  // TODO: deal with binning
  got_camera_info = true;

  if (cam_info.binning_x != 0) {
    ROS_WARN("[OpticFlow]: TODO : deal with binning when loading camera parameters.");
  }

  // check if the matricies have any data
  if (cam_info.K.size() < 6 || cam_info.D.size() < 5) {

    ROS_WARN("[OpticFlow]: Camera info has wrong calibration matricies.");
    negativeCamInfo = true;

  } else {

    fx = cam_info.K.at(0);
    fy = cam_info.K.at(4);
    cx = cam_info.K.at(2);
    cy = cam_info.K.at(5);
    k1 = cam_info.D.at(0);
    k2 = cam_info.D.at(1);
    p1 = cam_info.D.at(2);
    p2 = cam_info.D.at(3);
    k3 = cam_info.D.at(4);

    if (DEBUG) {
      ROS_INFO("[OpticFlow]: Camera params: %f %f %f %f %f %f %f %f %f", fx, fy, cx, cy, k1, k2, p1, p2, k3);
    }
  }
  CamInfoSubscriber.shutdown();
}

//}

// --------------------------------------------------------------
// |                       custom methods                       |
// --------------------------------------------------------------

//{ processImage()

void OpticFlow::processImage(const cv_bridge::CvImagePtr image) {

  // First things first
  if (first) {

    // not needed, because we are subscribed to camera_info topic
    /* if (ScaleFactor == 1) { */
    /*   int parameScale = image->image.cols / expectedWidth; */
    /*   fx              = fx * parameScale; */
    /*   cx              = cx * parameScale; */
    /*   fy              = fy * parameScale; */
    /*   cy              = cy * parameScale; */
    /*   k1              = k1 * parameScale; */
    /*   k2              = k2 * parameScale; */
    /*   k3              = k3 * parameScale; */
    /*   p1              = p1 * parameScale; */
    /*   p2              = p2 * parameScale; */
    /* } */

    ROS_INFO("[OpticFlow]: Source img: %dx%d", image->image.cols, image->image.rows);
    ROS_INFO("[OpticFlow]: Camera params: fx: %f, fy: %f \t cx: %f, cy: %f \t k1: %f, k2: %f \t p1: %f, p2: %f, p3: %f", fx, fy, cx, cy, k1, k2, p1, p2, k3);

    first = false;
  }

  if (!got_camera_info) {
    ROS_WARN("[OpticFlow]: Camera info didn't arrive yet! We don't have focus lenght coefficients. Can't publish optic flow.");
    return;
  }

  // Frequency control
  // Scaling
  if (ScaleFactor != 1) {
    cv::resize(image->image, imOrigScaled, cv::Size(image->image.size().width / ScaleFactor, image->image.size().height / ScaleFactor));
  } else {
    imOrigScaled = image->image.clone();
  }

  // Cropping
  int         imCenterX = imOrigScaled.size().width / 2;
  int         imCenterY = imOrigScaled.size().height / 2;
  int         xi        = imCenterX - (frameSize / 2);
  int         yi        = imCenterY - (frameSize / 2);
  cv::Rect    frameRect = cv::Rect(xi, yi, frameSize, frameSize);
  cv::Point2i midPoint  = cv::Point2i((frameSize / 2), (frameSize / 2));

  //  Converting color
  cv::cvtColor(imOrigScaled(frameRect), imCurr, CV_RGB2GRAY);

  // Calculate angular rate correction
  cv::Point2d tiltCorr = cv::Point2d(0, 0);

  if (tilt_correction_enable) {

    // do tilt correction (in pixels)
    tiltCorr.x = -angular_velocity.x * fx * dur.toSec();  // version 4
    tiltCorr.y = angular_velocity.y * fy * dur.toSec();

    // double xTiltCorr =  - fx * sqrt(2 - 2*cos(angular_velocity.x * dur.toSec())) * angular_velocity.x/abs(angular_velocity.x); // version 5
    // double yTiltCorr =  fy * sqrt(2 - 2*cos(angular_velocity.y * dur.toSec())) * angular_velocity.y/abs(angular_velocity.y);

    geometry_msgs::Vector3 tiltCorrOut;
    tiltCorrOut.x = tiltCorr.x;  // (tan(angular_velocity.y*dur.toSec())*trueRange)/dur.toSec();
    tiltCorrOut.y = tiltCorr.y;  // (tan(angular_velocity.x*dur.toSec())*trueRange)/dur.toSec();
    tiltCorrOut.z = 0;
    TiltCorrectionPublisher.publish(tiltCorrOut);
  }

  // Estimate scale and rotation (if enabled)
  cv::Point2d scaleRot = cv::Point2d(0, 0);

  if (scaleRot_enable && d3d_method.compare("logpol") == 0) {

    scaleRot = srEstimator->processImage(imCurr, gui, DEBUG);
    scaleRot.y /= dur.toSec();

    // Acceleration to altitude
    /*
       double d2 = scaleRot.x;
       double t23 = dur.toSec();

       double A = d1*(d2-1)/t23 - (d1-1)/t12;

       double ta = (t12 + t23)/2;
       double accZ = imu_last_msg.linear_acceleration.z - 9.80665;
    //ROS_INFO("[OpticFlow]: accZ: %f",accZ);
    scaleRot.x = (accZ*ta)/A;

    d1 = d2;
    t12 = t23;*/

    if (scale_rot_output.compare("altitude") == 0) {

      // Altitude from velocity

      if (abs(scaleRot.x - 1) > 0.01) {
        scaleRot.x = 0;  //(Zvelocity*dur.toSec())/(scaleRot.x - 1);
      } else {
        // ROS_INFO("[OpticFlow]: Scale too small: %f",scaleRot.x);
        scaleRot.x = 0;
      }

    } else {
      // Velocity from altitude
      scaleRot.x = ((scaleRot.x - 1) / trueRange) / dur.toSec();
    }
  }

  // Call the method function
  std::vector<cv::Point2f> speeds = processClass->processImage(imCurr, gui, DEBUG, midPoint, angular_velocity.z * dur.toSec(), tiltCorr);

  // Check for wrong values
  /*speeds = removeNanPoints(speeds);
    if(speeds.size() <= 0){
    ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
    return;
    }*/

  // RAW velocity without tilt corrections
  if (raw_enable) {

    if (method == 4 && (speeds.size() % 2 != 0)) {
      ROS_WARN("[OpticFlow]: Raw enabled and the processing function returned odd number of points. If this is not normal, disable raw veolcity.");
      return;
    }

    std::vector<cv::Point2f> speeds_no_rot_corr;
    // extract uncorrected (rot) from FFT method
    if (method == 4) {
      std::vector<cv::Point2f> speeds_rot_corr;
      for (int i = 0; i < speeds.size(); i += 2) {
        speeds_no_rot_corr.push_back(speeds[i]);
        speeds_rot_corr.push_back(speeds[i + 1]);
      }
      speeds = speeds_rot_corr;
    } else {
      speeds_no_rot_corr = speeds;
    }

    speeds_no_rot_corr = removeNanPoints(speeds_no_rot_corr);

    multiplyAllPts(speeds_no_rot_corr, -trueRange / (fx * dur.toSec()), trueRange / (fy * dur.toSec()));
    double phi = -1.570796326794897;
    rotateAllPts(speeds_no_rot_corr, phi);
    rotateAllPts(speeds_no_rot_corr, yaw);
    if (applyAbsBounding) {
      speeds_no_rot_corr = getOnlyInAbsBound(speeds_no_rot_corr, maxSpeed);  // bound according to max speed
    }

    cv::Point2f out;
    if (filterMethod.compare("average") == 0) {
      out = pointMean(speeds_no_rot_corr);
    } else if (filterMethod.compare("allsac") == 0) {
      int chosen;
      out = allsacMean(speeds_no_rot_corr, RansacThresholdRadSq, &chosen);
    } else if (filterMethod.compare("ransac") == 0) {
      out = ransacMean(speeds_no_rot_corr, RansacNumOfChosen, RansacThresholdRadSq, RansacNumOfIter);
    } else {
      ROS_ERROR("[OpticFlow]: Entered filtering method (filterMethod) does not match to any of these: average,ransac,allsac.");
    }

    geometry_msgs::TwistStamped velocity;

    velocity.header.frame_id = "local_origin";
    velocity.header.stamp    = ros::Time::now();

    velocity.twist.linear.x  = out.x;
    velocity.twist.linear.y  = out.y;
    velocity.twist.linear.z  = scaleRot.x;
    velocity.twist.angular.z = scaleRot.y;

    VelocityRawPublisher.publish(velocity);
  }

  // tilt correction! (FFT has it inside the processing function...)
  if (tilt_correction_enable && method != 4) {
    addToAll(speeds, tiltCorr.x, tiltCorr.y);
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: using angular velocity tilt correction");
  }

  // Advanced 3D positioning
  if (scaleRot_enable && d3d_method.compare("advanced") == 0) {
    if (speeds.size() != 9) {
      ROS_ERROR("[OpticFlow]: Speeds have a bad size for advanced 3D positioning!");
      return;
    }

    if (scale_rot_output.compare("altitude") == 0) {
      ROS_ERROR("[OpticFlow]: Cannot output altitude with 3D positioning, just vertical velocity!");
      return;
    }

    if (filterMethod.compare("average") == 0) {
      ROS_ERROR("[OpticFlow]: Cannot do averaging with advanced 3D positioning, just allsac!");
      return;
    }

    std::vector<cv::Point2f> trvv =
        estimateTranRotVvel(speeds, (double)samplePointSize, fx, fy, trueRange, RansacThresholdRadSq, dur.toSec(), maxVertSpeed, maxYawSpeed);
    speeds.clear();
    speeds.push_back(trvv[0]);  // translation in px
    scaleRot.x = trvv[1].y;     // rotation in rad/s
    scaleRot.y = trvv[1].x;     // vertical velocity
  }

  speeds = removeNanPoints(speeds);
  if (speeds.size() <= 0) {
    ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
    return;
  }

  /* for (int i=0;i<speeds.size();i++) */
  /*   ROS_INFO("[OpticFlow]: x:%f, y:%f",speeds[i].x,speeds[i].y); */

  /* ROS_INFO("[OpticFlow]: %2.2f %2.2f", -trueRange/(fx*dur.toSec()), trueRange/(fy*dur.toSec())); */

  // Calculate real velocity
  multiplyAllPts(speeds, -trueRange / (fx * dur.toSec()), trueRange / (fy * dur.toSec()));

  // ROTATION

  // camera rotation (within the construction) correction
  double phi = -1.570796326794897;

  /* for (int i=0;i<speeds.size();i++) */
  /*   ROS_INFO("[OpticFlow]: BBBx:%f, y:%f",speeds[i].x,speeds[i].y); */

  // rotate2d(vxm,vym,phi);
  rotateAllPts(speeds, phi);

  // transform to global system
  rotateAllPts(speeds, yaw);

  /* for (int i=0;i<speeds.size();i++) */
  /*   ROS_INFO("[OpticFlow]: x:%f, y:%f",speeds[i].x,speeds[i].y); */

  // Print output
  if (DEBUG) {
    ROS_INFO_THROTTLE(0.1, "[OpticFlow]: After recalc.");
    for (uint i = 0; i < speeds.size(); i++) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: %d -> vxr = %f; vyr=%f", i, speeds[i].x, speeds[i].y);
    }
  }

  // FILTERING
  // Bound speeds
  ros::Duration timeDif               = ros::Time::now() - odomSpeedTime;
  float         max_sp_dif_from_accel = maxAccel * timeDif.toSec() + speed_noise;  // speed_noise is always significantly hihger

  // Backup speeds for silent debug
  std::vector<cv::Point2f> bck_speeds;
  if (silent_debug) {
    bck_speeds = speeds;
  }

  int af_abs = 0;
  int af_acc = 0;

  if (applyAbsBounding) {
    // Bounding of speeds, if enabled
    if (DEBUG)
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds before bound #%lu", speeds.size());

    speeds = getOnlyInAbsBound(speeds, maxSpeed);  // bound according to max speed
    if (silent_debug) {
      af_abs = speeds.size();
    }
    if (DEBUG) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after speed bound #%lu, max speed: %f", speeds.size(), maxSpeed);
    }
  }
  if (applyRelBounding) {

    speeds = getOnlyInRadiusFromTruth(odomSpeed, speeds, max_sp_dif_from_accel);

    if (silent_debug) {
      af_acc = speeds.size();
    }

    if (DEBUG) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after acceleration bound #%lu, max speed from acc: %f", speeds.size(), max_sp_dif_from_accel);
    }

    if (speeds.size() < 1) {
      ROS_WARN("[OpticFlow]: No speeds after bounding, can't publish!");

      if (silent_debug) {
        for (uint i = 0; i < bck_speeds.size(); i++) {
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: %d -> vx = %f; vy=%f; v=%f; dist from odom=%f", i, bck_speeds[i].x, bck_speeds[i].y,
                            sqrt(getNormSq(bck_speeds[i])), sqrt(getDistSq(bck_speeds[i], odomSpeed)));
        }
        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Absolute max: %f, Odometry: vx = %f, vy = %f, v = %f, Max odom distance: %f", maxSpeed, odomSpeed.x, odomSpeed.y,
                          sqrt(getNormSq(odomSpeed)), max_sp_dif_from_accel);
        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: After absoulute bound: #%d, after accel: #%d", af_abs, af_acc);
      }

      return;
    }

  } else if (DEBUG) {
    ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Bounding of speeds not enabled.");
  }

  // Do the Allsac/Ransac/Averaging
  cv::Point2f out;
  if (filterMethod.compare("average") == 0) {
    out = pointMean(speeds);
  } else if (filterMethod.compare("allsac") == 0) {
    int chosen;
    out = allsacMean(speeds, RansacThresholdRadSq, &chosen);
    geometry_msgs::Vector3 allsacChosen;
    allsacChosen.x = chosen;
    allsacChosen.y = 0;
    allsacChosen.z = 0;
    AllsacChosenPublisher.publish(allsacChosen);
  } else if (filterMethod.compare("ransac") == 0) {
    out = ransacMean(speeds, RansacNumOfChosen, RansacThresholdRadSq, RansacNumOfIter);
  } else {
    ROS_ERROR("[OpticFlow]: Entered filtering method (filterMethod) does not match to any of these: average,ransac,allsac.");
  }

  vam = sqrt(getNormSq(out));
  vxm = out.x;
  vym = out.y;

  // | -------------------- publish velocity -------------------- |
  geometry_msgs::TwistStamped velocity;

  velocity.header.frame_id = "local_origin";
  velocity.header.stamp    = ros::Time::now();

  velocity.twist.linear.x  = vxm;
  velocity.twist.linear.y  = vym;
  velocity.twist.linear.z  = scaleRot.x;
  velocity.twist.angular.z = -scaleRot.y;

  if (DEBUG) {
    ROS_INFO_THROTTLE(0.1, "[OpticFlow]: vxm = %f; vym=%f; vam=%f; range=%f; yaw=%f", vxm, vym, vam, trueRange, yaw);
  }

  // Add speedbox to lastspeeds array
  SpeedBox sb;
  sb.time      = ros::Time::now();
  sb.speed     = out;
  sb.odomSpeed = odomSpeed;

  if (int(lastSpeeds.size()) >= lastSpeedsSize) {
    lastSpeeds.erase(lastSpeeds.begin());
  }

  lastSpeeds.push_back(sb);

  // Create statistical data
  ros::Time fromTime = sb.time - ros::Duration(analyseDuration);
  StatData  sd       = analyzeSpeeds(fromTime, lastSpeeds);

  velocity.twist.angular.x = yaw;  // PUBLISING YAW as something else...
  VelocityPublisher.publish(velocity);

  geometry_msgs::Vector3 v3;
  v3.x = sd.stdDevX;
  v3.y = sd.stdDevY;
  v3.z = sd.stdDev;
  VelocitySDPublisher.publish(v3);

  if (method == 5) {
    std_msgs::Float32 maxVel;
    maxVel.data = scanRadius * trueRange / (dur.toSec() * std::max(fx, fy));
    MaxAllowedVelocityPublisher.publish(maxVel);
  }
}

//}

}  // namespace optic_flow

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(optic_flow::OpticFlow, nodelet::Nodelet)
