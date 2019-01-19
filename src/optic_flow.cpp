
/* includes //{ */

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <string>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <mrs_msgs/Float64Stamped.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <std_msgs/UInt32MultiArray.h>
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

/* #ifdef OPENCL_ENABLE */

/* #include "optic_flow/FastSpacedBMMethod_OCL.h" */
/* #include "optic_flow/FastSpacedBMOptFlow.h" */
/* #include <opencv2/gpu/gpu.hpp> */

/* #endif */

#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/Profiler.h>
#include <mutex>

//}

namespace enc = sensor_msgs::image_encodings;

#define STRING_EQUAL 0

namespace optic_flow
{

/* //{ class OpticFlow */

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
  void callbackHeight(const mrs_msgs::Float64StampedConstPtr& msg);
  void callbackImu(const sensor_msgs::ImuConstPtr& msg);
  void callbackOdometry(const nav_msgs::OdometryConstPtr& msg);
  void callbackImage(const sensor_msgs::ImageConstPtr& msg);
  void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);

private:
  void processImage(const cv_bridge::CvImagePtr image);

private:
  ros::Timer cam_init_timer;
  void       camInitTimer(const ros::TimerEvent& event);

private:
  double max_processing_rate_;
  double max_pixel_speed_;
  double max_horizontal_speed_;
  double max_horizontal_acceleration_;
  double max_vertical_speed_;
  double max_yaw_rate_;

private:
  ros::Subscriber subscriber_image;
  ros::Subscriber subscriber_uav_height;
  ros::Subscriber subscriber_camera_info;
  ros::Subscriber subscriber_odometry;
  ros::Subscriber subscriber_imu;

  ros::Publisher publisher_velocity;
  ros::Publisher publisher_velocity_std;
  ros::Publisher publisher_points_raw;
  ros::Publisher publisher_max_allowed_velocity;
  ros::Publisher publisher_tilt_correction;
  ros::Publisher publisher_chosen_allsac;

  bool got_camera_info = false;
  bool got_image       = false;
  bool got_height      = false;
  bool got_imu         = false;
  bool got_odometry    = false;

  ros::Time camera_info_timeout;

private:
  double     uav_height;
  std::mutex mutex_uav_height;

private:
  cv::Point3d angular_rate;
  std::mutex  mutex_angular_rate;

private:
  double      odometry_roll, odometry_pitch, odometry_yaw;
  cv::Point2f odometry_speed;
  ros::Time   odometry_stamp;
  std::mutex  mutex_odometry;

private:
  std::vector<double> fallback_camera_data;
  std::vector<double> fallback_distortion_coeffs;

private:
  bool first_image = true;

  cv::Mat image_scaled;
  cv::Mat imCurr;
  cv::Mat imPrev;

  double vxm, vym, vam;

  ros::Time      begin;
  ros::Duration  dur;
  OpticFlowCalc* processClass;

  /*double ypr_time;
    double roll_old, pitch_old, yaw_old;
    double ypr_old_time;

    double roll_im_old, pitch_im_old, yaw_im_old;
    double roll_dif, pitch_dif, yaw_dif;*/

  // scale to altitude..
  double d1, t12;

  // Input arguments
  bool debug_;
  bool gui_;
  int  method_;
  bool silent_debug_;
  bool store_video_;

  double camera_yaw_offset_;

  // optic flow parameters
  int scan_radius_;
  int scanDiameter;
  int scanCount;
  int step_size_;

  int scale_factor_;

  int frame_size_;
  int sample_point_size_;
  int sample_point_count_;
  int sample_point_count_sqrt_;

  double calibration_coeff_x_;
  double calibration_coeff_y_;

  double cx, cy, fx, fy, s;
  double k1, k2, p1, p2, k3;

  int         ransac_num_of_chosen_;
  int         ransac_num_of_iter_;
  double       RansacThresholdRadSq;
  std::string filter_method_;

  bool                    rotation_correction_, tilt_correction_, raw_enabled_;
  std::string             ang_rate_source_;
  bool                    scale_rotation;
  double                  scale_rotation_magnitude_;
  std::string             scale_rot_output_;
  scaleRotationEstimator* scale_rotation_estimator;
  std::string             d3d_method_;

  bool apply_abs_bounding_;
  bool apply_rel_bouding_;

  float speed_noise;

  std::vector<SpeedBox> lastSpeeds;
  int                   last_speeds_size_;
  double                analyze_duration_;

private:
  mrs_lib::Profiler* profiler;
  bool profiler_enabled_ = false;

private:
  bool is_initialized = false;
};

//}

/* //{ onInit() */

void OpticFlow::onInit() {

  ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  mrs_lib::ParamLoader param_loader(nh_, "OpticFlow");

  // | -------------------- basic node params ------------------- |

  param_loader.load_param("enable_profiler", profiler_enabled_);
  param_loader.load_param("debug", debug_);
  param_loader.load_param("gui", gui_);
  param_loader.load_param("silent_debug", silent_debug_);

  // | --------------------- general params --------------------- |

  param_loader.load_param("ang_rate_source", ang_rate_source_);
  param_loader.load_param("raw_output", raw_enabled_);
  param_loader.load_param("camera_yaw_offset", camera_yaw_offset_);
  param_loader.load_param("analyze_duration", analyze_duration_);

  param_loader.load_param("scale_rotation", scale_rotation);
  param_loader.load_param("scale_rot_magnitude", scale_rotation_magnitude_);
  param_loader.load_param("scale_rot_output", scale_rot_output_);
  param_loader.load_param("d3d_method", d3d_method_);

  int videoFPS = param_loader.load_param2<int>("video_fps");

  // | -------------------- optic flow params ------------------- |

  param_loader.load_param("optic_flow/max_processing_rate", max_processing_rate_);
  param_loader.load_param("optic_flow/method", method_);
  param_loader.load_param("optic_flow/scan_radius", scan_radius_);
  param_loader.load_param("optic_flow/step_size", step_size_);
  param_loader.load_param("optic_flow/frame_size", frame_size_);
  param_loader.load_param("optic_flow/sample_point_size", sample_point_size_);
  sample_point_count_sqrt_ = frame_size_/sample_point_size_;
  sample_point_count_ = sample_point_count_sqrt_;


  param_loader.load_param("optic_flow/filter_method", filter_method_);

  param_loader.load_param("optic_flow/apply_abs_bouding", apply_abs_bounding_);
  param_loader.load_param("optic_flow/apply_rel_bouding", apply_rel_bouding_);

  {
    double calibration_coeff_both; // use this as a backup value in case calibrations for separate axes are not available
    param_loader.load_param("optic_flow/calibration/both_velocity_correction_ratio", calibration_coeff_both, 1.0);
    param_loader.load_param("optic_flow/calibration/x_velocity_correction_ratio", calibration_coeff_x_, calibration_coeff_both);
    param_loader.load_param("optic_flow/calibration/y_velocity_correction_ratio", calibration_coeff_y_, calibration_coeff_both);
  }

  param_loader.load_param("optic_flow/scale_factor", scale_factor_);

  param_loader.load_param("optic_flow/ransac/num_of_chosen", ransac_num_of_chosen_);
  param_loader.load_param("optic_flow/ransac/num_of_iter", ransac_num_of_iter_);
  RansacThresholdRadSq = pow(param_loader.load_param2<double>("optic_flow/ransac/threshold_rad"), 2);

  param_loader.load_param("optic_flow/rotation_correction", rotation_correction_);
  param_loader.load_param("optic_flow/tilt_correction", tilt_correction_);

  // method check
  if (method_ < 3 || method_ > 5) {
    ROS_ERROR("[OpticFlow]: No such OpticFlow calculation method. Available: 3 = BM on CPU, 4 = FFT on CPU, 5 = BM on GPU via OpenCL");
  }

  // | ------------------------ filtering ----------------------- |

  param_loader.load_param("optic_flow/filtering/last_speeds_size", last_speeds_size_);

  param_loader.load_param("store_video", store_video_);
  std::string video_path_ = param_loader.load_param2<std::string>("video_path");

  // | ------------------ physical constraints ------------------ |
  param_loader.load_param("constraints/max_pixel_speed", max_pixel_speed_);
  param_loader.load_param("constraints/max_horizontal_speed", max_horizontal_speed_);
  param_loader.load_param("constraints/max_horizontal_acceleration", max_horizontal_acceleration_);
  param_loader.load_param("constraints/max_vertical_speed", max_vertical_speed_);
  param_loader.load_param("constraints/max_yaw_rate", max_yaw_rate_);
  param_loader.load_param("constraints/speed_noise", speed_noise);

  // | --------------- fallback camera parameters --------------- |
  param_loader.load_param("camera_matrix/data", fallback_camera_data);
  param_loader.load_param("distortion_coefficients/data", fallback_distortion_coeffs);

  // --------------------------------------------------------------
  // |                    end of loading params                   |
  // --------------------------------------------------------------

  if (gui_) {
    cv::namedWindow("optic_flow", cv::WINDOW_FREERATIO);
  }

  if (scale_rotation && d3d_method_.compare("advanced") != 0 && d3d_method_.compare("logpol") != 0) {
    ROS_ERROR("[OpticFlow]: Wrong parameter 3d_method. Possible values: logpol, advanced. Entered: %s", d3d_method_.c_str());
    ros::shutdown();
  }

  if (filter_method_.compare("ransac") && ransac_num_of_chosen_ != 2) {
    ROS_WARN("[OpticFlow]: When Allsac is enabled, the ransac_num_of_chosen_ can be only 2.");
  }

  if (store_video_) {
    ROS_INFO("[OpticFlow]: Video path: %s", video_path_.c_str());
  }

  if (RAND_MAX < 100) {
    ROS_WARN("[OpticFlow]: Why is RAND_MAX set to only %d? Ransac in OpticFlow won't work properly!", RAND_MAX);
  }

  if ((frame_size_ % 2) == 1) {
    frame_size_--;
  }
  scanDiameter = (2 * scan_radius_ + 1);
  scanCount    = (scanDiameter * scanDiameter);

  // | -------------------- choose the method ------------------- |
  switch (method_) {
    case 3: {
              ROS_INFO("Method 3 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
      /* processClass = new BlockMethod(frame_size_, sample_point_size_, scan_radius_, scanDiameter, scanCount, step_size_); */
      break;
    }
    case 4: {
      processClass = new FftMethod(frame_size_, sample_point_size_, max_pixel_speed_, store_video_, raw_enabled_, rotation_correction_, tilt_correction_,
                                   &video_path_, videoFPS);
      break;
    }
    case 5: {
              ROS_INFO("Method 5 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
      /* processClass = new FastSpacedBMMethod(sample_point_size_, scan_radius_, step_size_, cx, cy, fx, fy, k1, k2, k3, p1, p2, store_video_, &video_path_); */
      break;
    }
  }

  imPrev = cv::Mat(frame_size_, frame_size_, CV_8UC1);
  imPrev = cv::Scalar(0);
  processClass->setImPrev(imPrev);

  begin = ros::Time::now();

  // prepare scale rotation estimator
  if (scale_rotation && d3d_method_.compare("logpol") == STRING_EQUAL) {
    if (scale_rot_output_.compare("velocity") != 0) {
      if (scale_rot_output_.compare("altitude") != 0) {
        ROS_ERROR("[OpticFlow]: Wrong parameter scale_rot_output_ - possible choices: velocity, altitude. Entered: %s", scale_rot_output_.c_str());
        ros::shutdown();
      }
    }
    std::string sr_name      = video_path_.append("_scale_rot.avi");
    scale_rotation_estimator = new scaleRotationEstimator(frame_size_, scale_rotation_magnitude_, store_video_, &sr_name, videoFPS);
  }

  // --------------------------------------------------------------
  // |                         publishers                         |
  // --------------------------------------------------------------

  publisher_chosen_allsac        = nh_.advertise<geometry_msgs::Vector3>("allsac_chosen_out", 1);
  publisher_velocity             = nh_.advertise<geometry_msgs::TwistStamped>("velocity_out", 1);
  publisher_velocity_std         = nh_.advertise<geometry_msgs::Vector3>("velocity_stddev_out", 1);
  publisher_max_allowed_velocity = nh_.advertise<std_msgs::Float32>("max_velocity_out", 1);
  publisher_tilt_correction      = nh_.advertise<geometry_msgs::Vector3>("tilt_correction_out", 1);

  if (raw_enabled_) {
    publisher_points_raw = nh_.advertise< std_msgs::UInt32MultiArray >("points_raw_out", 1);
  }

  // --------------------------------------------------------------
  // |                         subscribers                        |
  // --------------------------------------------------------------

  subscriber_camera_info = nh_.subscribe("camera_info_in", 1, &OpticFlow::callbackCameraInfo, this, ros::TransportHints().tcpNoDelay());
  subscriber_image       = nh_.subscribe("camera_in", 1, &OpticFlow::callbackImage, this, ros::TransportHints().tcpNoDelay());
  subscriber_uav_height  = nh_.subscribe("uav_height_in", 1, &OpticFlow::callbackHeight, this, ros::TransportHints().tcpNoDelay());
  subscriber_odometry    = nh_.subscribe("odometry_in", 1, &OpticFlow::callbackOdometry, this, ros::TransportHints().tcpNoDelay());

  if (ang_rate_source_.compare("imu") == STRING_EQUAL) {
    subscriber_imu = nh_.subscribe("imu_in", 1, &OpticFlow::callbackImu, this);
  } else {
    if (ang_rate_source_.compare("odometry") != 0) {
      ROS_ERROR("[OpticFlow]: Wrong parameter ang_rate_source_ - possible choices: imu, odometry. Entered: %s", ang_rate_source_.c_str());
      ros::shutdown();
    }
  }

  // --------------------------------------------------------------
  // |                          profiler                          |
  // --------------------------------------------------------------

  profiler = new mrs_lib::Profiler(nh_, "OpticFlow", profiler_enabled_);

  // --------------------------------------------------------------
  // |                           timers                           |
  // --------------------------------------------------------------

  cam_init_timer = nh_.createTimer(ros::Rate(10), &OpticFlow::camInitTimer, this);

  // | ----------------------- finish init ---------------------- |

  if (!param_loader.loaded_successfully()) {
    ROS_ERROR("[OpticFlow]: Could not load all parameters!");
    ros::shutdown();
  }

  is_initialized = true;

  ROS_INFO("[OpticFlow]: initialized");
}

//}

// --------------------------------------------------------------
// |                           timers                           |
// --------------------------------------------------------------

/* //{ camInitTimer() */

void OpticFlow::camInitTimer([[maybe_unused]] const ros::TimerEvent& event) {

  if (!got_image) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera");
    camera_info_timeout = ros::Time::now();
    return;
  }

  if (!got_camera_info && ((ros::Time::now() - camera_info_timeout).toSec() < 15.0)) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera info");
    return;
  }

  if (!got_camera_info) {

    ROS_WARN("[OpticFlow]: missing camera calibration parameters! (nothing on camera_info topic/wrong calibration matricies). Loading default parameters");

    fx = fallback_camera_data[0];
    cx = fallback_camera_data[2];
    fy = fallback_camera_data[4];
    cy = fallback_camera_data[5];

    k1 = fallback_distortion_coeffs[0];
    k2 = fallback_distortion_coeffs[1];
    k3 = fallback_distortion_coeffs[4];
    p1 = fallback_distortion_coeffs[2];
    p2 = fallback_distortion_coeffs[3];

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

/* //{ callbackHeight() */

void OpticFlow::callbackHeight(const mrs_msgs::Float64StampedConstPtr& msg) {

  if (!is_initialized)
    return;

  mrs_lib::Routine routine_callback_height = profiler->createRoutine("callbackUavHeight");

  if (absf(msg->value) < 0.001) {
    return;
  }

  got_height = true;

  {
    std::scoped_lock lock(mutex_uav_height);

    uav_height = msg->value;
  }
}

//}

/* //{ callbackImu() */

void OpticFlow::callbackImu(const sensor_msgs::ImuConstPtr& msg) {

  if (!is_initialized)
    return;

  mrs_lib::Routine routine_callback_imu = profiler->createRoutine("callbackImu");

  // angular rate source is imu aka gyro
  if (ang_rate_source_.compare("imu") == STRING_EQUAL) {

    {
      std::scoped_lock lock(mutex_angular_rate);

      angular_rate = cv::Point3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    }

    got_imu = true;
  }
}

//}

/* //{ callbackOdometry() */

void OpticFlow::callbackOdometry(const nav_msgs::OdometryConstPtr& msg) {

  if (!is_initialized)
    return;

  mrs_lib::Routine routine_callback_odometry = profiler->createRoutine("callbackOdometry");

  got_odometry = true;

  tf::Quaternion bt;
  tf::quaternionMsgToTF(msg->pose.pose.orientation, bt);
  tf::Matrix3x3(bt).getRPY(odometry_roll, odometry_pitch, odometry_yaw);

  if (ang_rate_source_.compare("odometry") == STRING_EQUAL) {
    {
      std::scoped_lock lock(mutex_angular_rate);

      angular_rate = cv::Point3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
    }
  }

  {
    std::scoped_lock lock(mutex_odometry);

    odometry_speed = cv::Point2f(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    odometry_stamp = ros::Time::now();
  }
}

//}

/* //{ callbackImage() */

void OpticFlow::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

  if (!is_initialized)
    return;

  mrs_lib::Routine routine_callback_image = profiler->createRoutine("callbackImage");

  got_image = true;

  ros::Time nowTime = msg->header.stamp;

  if (!first_image && (nowTime - begin).toSec() < 1 / max_processing_rate_) {
    if (debug_) {
      ROS_INFO("[OpticFlow]: MAX frequency overrun (%f). Skipping...", (nowTime - begin).toSec());
    }
    return;
  }

  dur   = nowTime - begin;
  begin = nowTime;
  if (debug_) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: freq = %fHz", 1.0 / dur.toSec());
  }

  cv_bridge::CvImagePtr image;
  image = cv_bridge::toCvCopy(msg, enc::BGR8);
  processImage(image);
}

//}

/* //{ callbackCameraInfo() */

void OpticFlow::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg) {

  if (!is_initialized)
    return;

  if (got_camera_info) {
    return;
  }

  // TODO: deal with binning
  if (msg->binning_x != 0) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: TODO: deal with binning when loading camera parameters.");
  }

  // check if the matricies have any data
  if (msg->K.size() < 6 || msg->D.size() < 5) {

    ROS_ERROR_THROTTLE(1.0, "[OpticFlow]: Camera info has wrong calibration matricies.");

  } else {

    fx = msg->K.at(0);
    fy = msg->K.at(4);
    cx = msg->K.at(2);
    cy = msg->K.at(5);

    k1              = msg->D.at(0);
    k2              = msg->D.at(1);
    p1              = msg->D.at(2);
    p2              = msg->D.at(3);
    k3              = msg->D.at(4);
    got_camera_info = true;

    if (debug_) {
      ROS_INFO("[OpticFlow]: Camera params: %f %f %f %f %f %f %f %f %f", fx, fy, cx, cy, k1, k2, p1, p2, k3);
    }
  }
}

//}

// --------------------------------------------------------------
// |                          routines                          |
// --------------------------------------------------------------

/* //{ processImage() */

void OpticFlow::processImage(const cv_bridge::CvImagePtr image) {

  // let's wait for two images
  if (first_image) {
    first_image = false;
    return;
  }

  // we need camera info!
  if (!got_camera_info) {
    return;
  }

  // we need to know the UAV height
  if (!got_height) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for uav height!");
    return;
  }

  // scale the image
  if (fabs(scale_factor_ - 1.0) > 0.01) {
    cv::resize(image->image, image_scaled, cv::Size(image->image.size().width / scale_factor_, image->image.size().height / scale_factor_));
  } else {
    image_scaled = image->image.clone();
  }

  // cropping
  /* int image_center_x = image_scaled.size().width / 2; */
  int image_center_x = cx; //The distortion will be more symmetrical ->  better compensation
  int image_center_y = image_scaled.size().height / 2;
  int xi             = image_center_x - (frame_size_ / 2);
  int yi             = image_center_y - (frame_size_ / 2);

  // rectification
  cv::Rect    cropping_rectangle = cv::Rect(xi, yi, frame_size_, frame_size_);
  cv::Point2i mid_point          = cv::Point2i((frame_size_ / 2), (frame_size_ / 2));

  //  convert to grayscale
  cv::cvtColor(image_scaled(cropping_rectangle), imCurr, CV_RGB2GRAY);

  // | ----------------- angular rate correction ---------------- |

  cv::Point2d tiltCorr = cv::Point2d(0, 0);

  if (tilt_correction_) {

    // do tilt correction (in pixels)
    {
      std::scoped_lock lock(mutex_angular_rate);

      double angrateX, angrateY;
      angrateX = angular_rate.x;
      angrateY = angular_rate.y;
      rotate2d(angrateX, angrateY, -camera_yaw_offset_);

      tiltCorr.x = tan(-angrateX * dur.toSec())*fx;  // version V - the good one dammit
      tiltCorr.y = tan(angrateY * dur.toSec())*fy;

      /* double xTiltCorr = -fx * sqrt(2 - 2 * cos(angular_rate.x * dur.toSec())) * angular_rate.x / abs(angular_rate.x);  // version 5 */
      /* double yTiltCorr = fy * sqrt(2 - 2 * cos(angular_rate.y * dur.toSec())) * angular_rate.y / abs(angular_rate.y); */
    }

    geometry_msgs::Vector3 tiltCorrOut;
    tiltCorrOut.x = tiltCorr.x;  // (tan(angular_rate.y*dur.toSec())*uav_height)/dur.toSec(); // if enabling, dont forget to mutex range and angular_rate
    tiltCorrOut.y = tiltCorr.y;  // (tan(angular_rate.x*dur.toSec())*uav_height)/dur.toSec(); // if enabling, dont forget to mutex range and angular_rate
    tiltCorrOut.z = 0;
    publisher_tilt_correction.publish(tiltCorrOut);
  }

  // Estimate scale and rotation (if enabled)
  cv::Point2d scale_and_rotation = cv::Point2d(0, 0);

  if (scale_rotation && d3d_method_.compare("logpol") == STRING_EQUAL) {

    scale_and_rotation = scale_rotation_estimator->processImage(imCurr, gui_, debug_);
    scale_and_rotation.y /= dur.toSec();

    if (scale_rot_output_.compare("altitude") == STRING_EQUAL) {

      // Altitude from velocity

      if (abs(scale_and_rotation.x - 1) > 0.01) {
        scale_and_rotation.x = 0;  //(Zvelocity*dur.toSec())/(scale_and_rotation.x - 1);
      } else {
        // ROS_INFO("[OpticFlow]: Scale too small: %f",scale_and_rotation.x);
        scale_and_rotation.x = 0;
      }

    } else {
      // Velocity from altitude
      {
        std::scoped_lock lock(mutex_uav_height);

        scale_and_rotation.x = ((scale_and_rotation.x - 1) / uav_height) / dur.toSec();
      }
    }
  }

  // process image
  std::vector<cv::Point2d> optic_flow_vectors;
  std::vector<cv::Point2d> optic_flow_vectors_raw;
  double                   temp_angular_rate;

  {
    std::scoped_lock lock(mutex_angular_rate);

    temp_angular_rate = angular_rate.z;
  }

  optic_flow_vectors = processClass->processImage(imCurr, gui_, debug_, mid_point, temp_angular_rate * dur.toSec(), tiltCorr,optic_flow_vectors_raw);

  // check for nans
  optic_flow_vectors = removeNanPoints(optic_flow_vectors);
  if (optic_flow_vectors.size() <= 0) {
    ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
    return;
  }

  // raw velocity without tilt corrections
  if (raw_enabled_) {

    if (method_ == 4 && (optic_flow_vectors.size() != sample_point_count_)) {
      ROS_WARN("[OpticFlow]: Raw enabled and the processing function returned unexpected number of vectors. If this is not normal, disable raw veolcity.");
      return;
    }

    std_msgs::UInt32MultiArray msg_raw;
    {
      std::scoped_lock lock(mutex_uav_height);


      msg_raw.layout.dim.push_back(std_msgs::MultiArrayDimension());
      msg_raw.layout.dim.push_back(std_msgs::MultiArrayDimension());
      msg_raw.layout.dim[0].size   = optic_flow_vectors_raw.size();
      msg_raw.layout.dim[0].label  = "count";
      msg_raw.layout.dim[0].stride = optic_flow_vectors_raw.size() * 2;
      msg_raw.layout.dim[1].size   = 2;
      msg_raw.layout.dim[1].label  = "value";
      msg_raw.layout.dim[1].stride = 2;
      std::vector< unsigned int > convert;
      for (int i = 0; i < (int)(optic_flow_vectors_raw.size()); i++) {
        convert.push_back(optic_flow_vectors_raw[i].x);
        convert.push_back(optic_flow_vectors_raw[i].y);
      }
    msg_raw.data = convert;
    }

    try {
      publisher_points_raw.publish(msg_raw);
    }
    catch (...) {
      ROS_ERROR("Exception caught during publishing topic %s.", publisher_points_raw.getTopic().c_str());
    }
  }

  // | ----------------- advanced 3D positioning ---------------- |
  if (scale_rotation && d3d_method_.compare("advanced") == STRING_EQUAL) {
    ROS_ERROR("[opticflow]: This implementation of SO3 motion called \"advanced\" is stupid. Wait for a new one, named \"pnp\".");
        return;

    if (optic_flow_vectors.size() != 9) {
      ROS_ERROR("[opticflow]: Advanced 3D positioning requires 3X3 sample points! Skipping...");
      return;
    }

    if (scale_rot_output_.compare("altitude") == STRING_EQUAL) {
      ROS_ERROR("[OpticFlow]: Cannot output altitude with 3D positioning, just vertical velocity!");
      return;
    }

    if (filter_method_.compare("average") == STRING_EQUAL) {
      ROS_ERROR("[OpticFlow]: Cannot do averaging with advanced 3D positioning, just allsac!");
      return;
    }

    std::vector<cv::Point2d> trvv;
    {
      std::scoped_lock lock(mutex_uav_height);

      trvv = estimateTranRotVvel(optic_flow_vectors, (double)sample_point_size_, fx, fy, uav_height, RansacThresholdRadSq, dur.toSec(), max_vertical_speed_, max_yaw_rate_);
    }

    optic_flow_vectors.clear();
    optic_flow_vectors.push_back(trvv[0]);  // translation in px
    scale_and_rotation.x = trvv[1].y;     // rotation in rad/s
    scale_and_rotation.y = trvv[1].x;     // vertical velocity
  }

  optic_flow_vectors = removeNanPoints(optic_flow_vectors);

  if (optic_flow_vectors.size() <= 0) {
    ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
    return;
  }

  // --------------------------------------------------------------
  // |               scale the velocity using height              |
  // --------------------------------------------------------------

  {
    std::scoped_lock lock(mutex_uav_height);

    multiplyAllPts(optic_flow_vectors, -uav_height / (fx * dur.toSec()), uav_height / (fy * dur.toSec()));
  }

  // rotate by camera yaw
  rotateAllPts(optic_flow_vectors, camera_yaw_offset_);

  // rotate to global system
  rotateAllPts(optic_flow_vectors, odometry_yaw);

  // Print output
  if (debug_) {
    ROS_INFO_THROTTLE(0.1, "[OpticFlow]: After recalc.");
    for (uint i = 0; i < optic_flow_vectors.size(); i++) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: %d -> vxr = %f; vyr=%f", i, optic_flow_vectors[i].x, optic_flow_vectors[i].y);
    }
  }

  // FILTERING
  // Bound optic_flow_vectors
  ros::Duration timeDif               = ros::Time::now() - odometry_stamp;
  float         max_sp_dif_from_accel = max_horizontal_acceleration_ * timeDif.toSec() + speed_noise;  // speed_noise is always significantly hihger

  // Backup optic_flow_vectors for silent debug_
  std::vector<cv::Point2d> bck_speeds;
  if (silent_debug_) {
    bck_speeds = optic_flow_vectors;
  }

  int af_abs = 0;
  int af_acc = 0;

  /* absolute bouding //{ */

  if (apply_abs_bounding_) {

    // Bounding of optic_flow_vectors, if enabled
    if (debug_)
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds before bound #%lu", optic_flow_vectors.size());

    optic_flow_vectors = getOnlyInAbsBound(optic_flow_vectors, max_horizontal_speed_);  // bound according to max speed
    if (silent_debug_) {
      af_abs = optic_flow_vectors.size();
    }
    if (debug_) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after speed bound #%lu, max speed: %f", optic_flow_vectors.size(), max_horizontal_speed_);
    }
  }

  //}

  /* relative bounding //{ */

  if (apply_rel_bouding_) {

    {
      std::scoped_lock lock(mutex_odometry);

      optic_flow_vectors = getOnlyInRadiusFromTruth(odometry_speed, optic_flow_vectors, max_sp_dif_from_accel);
    }

    if (silent_debug_) {
      af_acc = optic_flow_vectors.size();
    }

    if (debug_) {
      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after acceleration bound #%lu, max speed from acc: %f", optic_flow_vectors.size(), max_sp_dif_from_accel);
    }

    if (optic_flow_vectors.size() < 1) {
      ROS_WARN("[OpticFlow]: No optic_flow_vectors after bounding, can't publish!");

      if (silent_debug_) {
        {
          std::scoped_lock lock(mutex_odometry);

          for (uint i = 0; i < bck_speeds.size(); i++) {
            ROS_INFO_THROTTLE(0.1, "[OpticFlow]: %d -> vx = %f; vy=%f; v=%f; dist from odom=%f", i, bck_speeds[i].x, bck_speeds[i].y,
                              sqrt(getNormSq(bck_speeds[i])), sqrt(getDistSq(bck_speeds[i], odometry_speed)));
          }
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Absolute max: %f, Odometry: vx = %f, vy = %f, v = %f, Max odom distance: %f", max_horizontal_speed_,
                            odometry_speed.x, odometry_speed.y, sqrt(getNormSq(odometry_speed)), max_sp_dif_from_accel);
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: After absoulute bound: #%d, after accel: #%d", af_abs, af_acc);
        }
      }

      return;
    }

  } else if (debug_) {
    ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Bounding of optic_flow_vectors not enabled.");
  }

  //}

  /* post-process by Allsac/Ransac/Averaging //{ */

  // apply Allsac/Ransac/Averaging
  cv::Point2f out;
  if (filter_method_.compare("average") == STRING_EQUAL) {

    out = pointMean(optic_flow_vectors);

  } else if (filter_method_.compare("allsac") == STRING_EQUAL) {

    int chosen;
    out = allsacMean(optic_flow_vectors, RansacThresholdRadSq, &chosen);
    geometry_msgs::Vector3 allsacChosen;
    allsacChosen.x = chosen;
    allsacChosen.y = 0;
    allsacChosen.z = 0;
    publisher_chosen_allsac.publish(allsacChosen);

  } else if (filter_method_.compare("ransac") == STRING_EQUAL) {

    out = ransacMean(optic_flow_vectors, ransac_num_of_chosen_, RansacThresholdRadSq, ransac_num_of_iter_);

  } else {
    ROS_ERROR("[OpticFlow]: Entered filtering method (filter_method_) does not match to any of these: average,ransac,allsac.");
  }

  //}
  
  /* apply odometry calibration coefficients //{ */

  {
    double unrot_x = out.x;
    double unrot_y = out.y;
    // rotate to UAV frame
    rotate2d(unrot_x, unrot_y, -odometry_yaw);
    // rotate to camera frame
    rotate2d(unrot_x, unrot_y, -camera_yaw_offset_);
    unrot_x = unrot_x * calibration_coeff_x_;
    unrot_y = unrot_y * calibration_coeff_y_;
    // rotate to UAV frame
    rotate2d(unrot_x, unrot_y, camera_yaw_offset_);
    // rotate to world frame
    rotate2d(unrot_x, unrot_y, odometry_yaw);
    out.x = unrot_x;
    out.y = unrot_y;
  }

  //}

  vam = sqrt(getNormSq(out));
  vxm = out.x;
  vym = out.y;

  // | -------------------- publish velocity -------------------- |
  geometry_msgs::TwistStamped velocity;

  velocity.header.frame_id = "local_origin";
  velocity.header.stamp    = ros::Time::now();

  velocity.twist.linear.x  = vxm;
  velocity.twist.linear.y  = vym;
  velocity.twist.linear.z  = scale_and_rotation.x;
  velocity.twist.angular.z = -scale_and_rotation.y;

  if (debug_) {
    {
      std::scoped_lock lock(mutex_uav_height);

      ROS_INFO_THROTTLE(0.1, "[OpticFlow]: vxm = %f; vym=%f; vam=%f; range=%f; odometry_yaw=%f", vxm, vym, vam, uav_height, odometry_yaw);
    }
  }

  // Add speedbox to lastspeeds array
  SpeedBox sb;
  sb.time  = ros::Time::now();
  sb.speed = out;
  {
    std::scoped_lock lock(mutex_odometry);

    sb.odometry_speed = odometry_speed;
  }

  if (int(lastSpeeds.size()) >= last_speeds_size_) {
    lastSpeeds.erase(lastSpeeds.begin());
  }

  lastSpeeds.push_back(sb);

  // Create statistical data
  //
  ros::Time fromTime = sb.time - ros::Duration(analyze_duration_);
  StatData  sd       = analyzeSpeeds(fromTime, lastSpeeds);

  velocity.twist.angular.x = odometry_yaw;  // PUBLISING odometry_yaw as something else...
  publisher_velocity.publish(velocity);

  geometry_msgs::Vector3 v3;
  v3.x = sd.stdDevX;
  v3.y = sd.stdDevY;
  v3.z = sd.stdDev;
  publisher_velocity_std.publish(v3);

  if (method_ == 5) {

    std_msgs::Float32 maxVel;
    {
      std::scoped_lock lock(mutex_uav_height);

      maxVel.data = scan_radius_ * uav_height / (dur.toSec() * std::max(fx, fy));
    }

    publisher_max_allowed_velocity.publish(maxVel);
  }
}

//}

}  // namespace optic_flow

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(optic_flow::OpticFlow, nodelet::Nodelet)
