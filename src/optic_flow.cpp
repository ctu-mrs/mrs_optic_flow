#define VERSION "0.0.5.0"

/* includes //{ */

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <string>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <image_transport/image_transport.h>
#include <mrs_msgs/Float64Stamped.h>
#include <mrs_msgs/ControlManagerDiagnostics.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <std_msgs/UInt32MultiArray.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float32.h>

#define LONGRANGE_INLIER_THRESHOLD 15

using namespace std;

//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>

#include <OpticFlowCalc.h>
#include <BlockMethod.h>
#include <FftMethod.h>
#include <utilityFunctions.h>
#include <scaleRotationEstimator.h>

/* #ifdef OPENCL_ENABLE */

/* #include "mrs_optic_flow/FastSpacedBMMethod_OCL.h" */
/* #include "mrs_optic_flow/FastSpacedBMOptFlow.h" */
/* #include <opencv2/gpu/gpu.hpp> */

/* #endif */

#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/Profiler.h>
#include <mutex>

//}

namespace enc = sensor_msgs::image_encodings;

#define LONG_RANGE_RATIO 4

#define filter_ratio 1.0

namespace mrs_optic_flow
{

// | -------------- helper functions and structs -------------- |

/* cvMat33ToTf2Mat33() //{ */

tf2::Matrix3x3 cvMat33ToTf2Mat33(cv::Mat& input) {

  tf2::Matrix3x3 output;
  for (int j = 0; j < 3; j++) {
    for (int k = 0; k < 3; k++) {
      output[k][j] = input.at<double>(j, k);
    }
  }
  return output;
}

//}

/* rotX() //{ */

cv::Matx33d rotX(double ang) {

  cv::Matx33d output = cv::Matx33d::zeros();
  output(0, 0)       = 1;
  output(1, 1)       = cos(ang);
  output(2, 2)       = cos(ang);
  output(2, 1)       = -sin(ang);
  output(1, 2)       = sin(ang);
  return output;
}

//}

/*  rotY() //{ */

cv::Matx33d rotY(double ang) {

  cv::Matx33d output = cv::Matx33d::zeros();
  output(1, 1)       = 1;
  output(0, 0)       = cos(ang);
  output(2, 2)       = cos(ang);
  output(2, 0)       = sin(ang);
  output(0, 2)       = -sin(ang);
  return output;
}

//}

/* struct PointValue //{ */

struct PointValue
{
  int         value;
  cv::Point2i location;
};

//}

// | ----------------------- main class ----------------------- |

/* class OpticFlow //{ */

class OpticFlow : public nodelet::Nodelet {

public:
  virtual void onInit();

  std::string     _version_;
  ros::NodeHandle nh_;

private:
  void callbackHeight(const mrs_msgs::Float64StampedConstPtr& msg);
  void callbackImu(const sensor_msgs::ImuConstPtr& msg);
  void callbackOdometry(const nav_msgs::OdometryConstPtr& msg);
  void callbackImage(const sensor_msgs::ImageConstPtr& msg);
  void callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);
  void callbackTrackerStatus(const mrs_msgs::ControlManagerDiagnosticsConstPtr& msg);

  int nrep_;

  void processImage(const cv_bridge::CvImagePtr image);
  bool getRT(std::vector<cv::Point2d> shifts, double height, cv::Point2d ulCorner, tf2::Quaternion& o_rot, tf2::Vector3& o_tran);
  bool get2DT(std::vector<cv::Point2d> shifts, double height, cv::Point2d ulCorner, tf2::Vector3& o_tran, tf2::Vector3& o_tran_diff);

  bool isUavLandoff();

  void       timerTf(const ros::TimerEvent& event);
  ros::Timer timer_tf_;
  bool       got_c2b_ = false;
  bool       got_b2c_ = false;
  bool       got_tfs_ = false;

  bool                    got_tracker_status_ = false;
  mrs_msgs::TrackerStatus tracker_status_;

  ros::Timer timer_cam_init_;
  void       timerCamInit(const ros::TimerEvent& event);

  tf2_ros::Buffer                 buffer_;
  tf2_ros::TransformListener*     listener_;
  geometry_msgs::TransformStamped transformCam2Base_;
  geometry_msgs::TransformStamped transformBase2Cam_;
  /* geometry_msgs::TransformStamped transformBase2CamLink; */
  double cam_yaw_;
  /* geometry_msgs::Transform  transformBase2World; */

  double _max_processing_rate_;
  double _max_pixel_speed_;
  double _max_horizontal_speed_;
  double _max_horizontal_acceleration_;
  double _max_vertical_speed_;
  double _max_yaw_rate_;

  ros::Subscriber subscriber_image_;
  ros::Subscriber subscriber_uav_height_;
  ros::Subscriber subscriber_camera_info_;
  ros::Subscriber subscriber_tracker_status_;
  ros::Subscriber subscriber_odometry_;
  ros::Subscriber subscriber_imu_;

  ros::Publisher publisher_velocity_;
  ros::Publisher publisher_velocity_longrange_;
  ros::Publisher publisher_velocity_longrange_diff_;
  ros::Publisher publisher_velocity_std_;
  ros::Publisher publisher_points_raw_;
  ros::Publisher publisher_max_allowed_velocity_;
  ros::Publisher publisher_chosen_allsac_;

  bool got_camera_info_ = false;
  bool got_image_       = false;
  bool got_height_      = false;
  bool got_imu_         = false;
  bool got_odometry_    = false;

  ros::Time camera_info_timeout_;

  double     uav_height_;
  std::mutex mutex_uav_height_;

  std::mutex mutex_tracker_status_;

  std::mutex mutex_process_;

  tf2::Quaternion tilt_prev_;
  tf2::Quaternion tilt_curr_;
  cv::Point3d     angular_rate_;
  tf2::Quaternion angular_rate_tf_;
  cv::Point3d     angle_diff_;
  tf2::Matrix3x3  rotMatDiff_;
  cv::Point3d     angle_diff_curr_;
  cv::Point3d     angular_rate_curr_;
  std::mutex      mutex_angular_rate_;
  std::mutex      mutex_static_tilt_;
  std::mutex      mutex_dynamic_tilt_;

  tf2::Quaternion odometry_orientation_;
  tf2::Quaternion imu_orientation_;
  double          odometry_roll_, odometry_pitch_, odometry_yaw_;
  double          imu_roll_, imu_pitch_, imu_yaw_;
  double          imu_roll_rate_, imu_pitch_rate_;
  double          odometry_roll_h_, odometry_pitch_h_, odometry_yaw_h_;
  cv::Point2f     odometry_speed_;
  ros::Time       odometry_stamp_;
  std::mutex      mutex_odometry_;
  std::mutex      mutex_tf_;

private:
  /* std::vector<double> fallback_camera_data; */
  /* std::vector<double> fallback_distortion_coeffs; */

private:
  bool first_image_ = true;

  cv::Mat image_scaled_;
  cv::Mat imCurr_;
  cv::Mat imPrev_;

  ros::Time     begin_;
  ros::Duration dur_;
  /* OpticFlowCalc* processClass; */
  FftMethod* fftProcessor_;

  /*double ypr_time;
    double roll_old, pitch_old, yaw_old;
    double ypr_old_time;

    double roll_im_old, pitch_im_old, yaw_im_old;
    double roll_dif, pitch_dif, yaw_dif;*/

  // Input arguments
  bool _debug_;
  bool _gui_;
  int  _method_;
  bool _silent_debug_;
  bool store_video_;

  // optic flow parameters
  int _shifted_pts_thr_;

  int _scan_radius_;
  int scanDiameter;
  int scanCount;
  int _step_size_;

  int _scale_factor_;

  double _takeoff_height_;

  int _frame_size_;
  int _sample_point_size_;
  int sample_point_size_lr;

  int sample_point_count_;
  int sample_point_count_sqrt_;

  double _calibration_coeff_x_;
  double _calibration_coeff_y_;

  double  cx_, cy_, fx_, fy_, s_;
  double  k1_, k2_, p1_, p2_, k3_;
  cv::Mat camMatrix_, distCoeffs_;

  int    _ransac_num_of_chosen_;
  int    _ransac_num_of_iter_;
  double _RansacThresholdRadSq_;

  std::string _camera_frame_, _camera_link_frame_, _uav_frame_, _uav_untilted_frame_;

  std::string _fft_cl_file_;
  bool        _useOCL_;

  std::string _long_range_mode_string_;

  std::string _filter_method_;

  bool                    _rotation_correction_, _raw_enabled_;
  std::string             _ang_rate_source_;
  bool                    _scale_rotation_;
  double                  _scale_rotation_magnitude_;
  std::string             _scale_rot_output_;
  scaleRotationEstimator* scale_rotation_estimator_;
  /* std::string             d3d_method_; */

  bool _apply_abs_bounding_;
  bool _apply_rel_bouding_;

  float _speed_noise_;

  std::vector<SpeedBox> lastSpeeds_;
  double                _analyze_duration_;

private:
  mrs_lib::Profiler* profiler_;
  bool               _profiler_enabled_ = false;

private:
  bool        is_initialized_ = false;
  std::string _uav_name_;
};

//}

/* getInlisers() //{ */

std::vector<unsigned int> getInliers(std::vector<cv::Point2d> shifts, double threshold) {

  std::vector<unsigned int> inliers;
  std::vector<unsigned int> inliers_tentative;

  for (int i = 0; i < int(shifts.size()); i++) {

    inliers_tentative.clear();
    inliers_tentative.push_back(i);

    for (int j = 0; j < int(shifts.size()); j++) {

      if (i == j)
        continue;

      if (cv::norm(shifts[i] - shifts[j]) < threshold)
        inliers_tentative.push_back(j);
    }

    if (inliers.size() < inliers_tentative.size())
      inliers = inliers_tentative;
  }
  return inliers;
}

//}

/* //{ isUavLandoff() */

bool OpticFlow::isUavLandoff() {

  std::scoped_lock lock(mutex_tracker_status_);

  if (got_tracker_status_) {

    if (tracker_status_.tracker == "LandoffTracker") {

      return true;
    } else {

      return false;
    }

  } else {

    ROS_WARN_THROTTLE(1.0, "[Odometry]: Tracker status not available");
    return false;
  }
}

//}

/* get2DT() //{ */
bool OpticFlow::get2DT(std::vector<cv::Point2d> shifts, double height, cv::Point2d ulCorner, tf2::Vector3& o_tran, tf2::Vector3& o_tran_diff) {
  if (shifts.size() < 1) {
    ROS_ERROR("[OpticFlow]: No points given, returning");
    return false;
  }
  if (!std::isfinite(1.0 / dur_.toSec())) {
    ROS_ERROR("[OpticFlow]: Duration is %f. Returning.", dur_.toSec());
    return false;
  }

  cv::Matx33d camMatrixLocal = camMatrix_;
  camMatrixLocal(0, 2) -= ulCorner.x;
  std::vector<cv::Point2d> initialPts, shiftedPts, shiftsPassed, undistPtsA, undistPtsB;

  int sqNum_lr = _frame_size_ / sample_point_size_lr;

  for (int j = 0; j < sqNum_lr; j++) {
    for (int i = 0; i < sqNum_lr; i++) {

      if (!std::isfinite(shifts[i + sqNum_lr * j].x) || !std::isfinite(shifts[i + sqNum_lr * j].y)) {
        ROS_ERROR("[OpticFlow]: NaN detected in variable \"shifts[i + sqNum_lr * j])\" - i = %d; j = %d!!!", i, j);
        continue;
      }

      int xi = i * sample_point_size_lr + (sample_point_size_lr / 2);
      int yi = j * sample_point_size_lr + (sample_point_size_lr / 2);
      initialPts.push_back(cv::Point2d(xi, yi));
      shiftedPts.push_back(cv::Point2d(xi, yi) + shifts[i + sqNum_lr * j]);
      shiftsPassed.push_back(shifts[i + sqNum_lr * j]);
    }
  }

  if (LONG_RANGE_RATIO == 2) {
    if (shiftedPts.size() < 3) {
      ROS_ERROR("[OpticFlow]: Not enough valid points found, returning");
      return false;
    }
  } else if (LONG_RANGE_RATIO == 4) {
    if (shiftedPts.size() < 1) {
      ROS_ERROR("[OpticFlow]: Valid point not found, returning");
      return false;
    }
  }

  /* if (shiftedPts.size() < uint(_shifted_pts_thr_)) { */
  /*   ROS_ERROR("[OpticFlow]: shiftPts contains many NaNs, returning"); */
  /*   return false; */
  /* } */

  /* ROS_INFO("Here A"); */
  /* ROS_INFO_STREAM("count: " << initialPts.size()); */
  for (auto pt : initialPts)
    /* ROS_INFO_STREAM(initialPts.isContinuous() << " " << intialPts.type() << " " << initialPts.depth() << " " <<  initialPts.channels() << " " <<
     * intialPts.cols << " " << initialPts.rows ); */
    cv::undistortPoints(initialPts, undistPtsA, camMatrixLocal, distCoeffs_);
  /* ROS_INFO("Here B"); */
  cv::undistortPoints(shiftedPts, undistPtsB, camMatrixLocal, distCoeffs_);
  /* ROS_INFO("Here C"); */

  std::vector<cv::Point2d> undistShifts;
  ;
  for (size_t i = 0; i < shiftedPts.size(); i++) {
    undistShifts.push_back(shiftedPts[i] - initialPts[i]);
  }

  cv::Point2d avgShift;

  if (LONG_RANGE_RATIO == 2) {
    std::vector<unsigned int> inliers = getInliers(undistShifts, LONGRANGE_INLIER_THRESHOLD);

    if (inliers.size() < 3) {
      ROS_ERROR("[OpticFlow]: less than 3 out of 4 samples are inliers, returning");
      return false;
    }

    avgShift = cv::Point2d(0.0, 0.0);
    for (size_t i = 0; i < inliers.size(); i++) {
      avgShift += undistShifts[inliers[i]];
    }
    avgShift = avgShift / (double)(inliers.size());
  } else if (LONG_RANGE_RATIO == 4) {
    avgShift = undistShifts[0];
  }

  double multiplier;
  if (LONG_RANGE_RATIO == 4)
    multiplier = 4;
  else if (LONG_RANGE_RATIO == 2)
    multiplier = 2;

  double x_corr_cam, y_corr_cam;
  {
    std::scoped_lock lock(mutex_tf_, mutex_dynamic_tilt_);
    double           x_corr   = -tan(imu_roll_rate_ * dur_.toSec()) * camMatrixLocal(0, 0) / multiplier;
    double           y_corr   = tan(imu_pitch_rate_ * dur_.toSec()) * camMatrixLocal(1, 1) / multiplier;
    double           t_corr   = sqrt(y_corr * y_corr + x_corr * x_corr);
    double           yaw_corr = atan2(y_corr, x_corr) + cam_yaw_;
    x_corr_cam                = cos(yaw_corr) * t_corr;
    y_corr_cam                = sin(yaw_corr) * t_corr;
  }
  ROS_INFO_STREAM("[OpticFlow]: cam_yaw_: " << cam_yaw_ << " x_corr_cam: " << x_corr_cam << " y_corr_cam: " << y_corr_cam);
  avgShift.x += x_corr_cam;
  avgShift.y += y_corr_cam;
  o_tran.setX(avgShift.x * (height / camMatrixLocal(0, 0) * multiplier));
  o_tran.setY(avgShift.y * (height / camMatrixLocal(1, 1) * multiplier));
  o_tran.setZ(0.0);

  o_tran = -o_tran / dur_.toSec();

  tf2::Vector3 o_tran_corr;

  avgShift.x += x_corr_cam;
  avgShift.y += y_corr_cam;
  o_tran_corr.setX(avgShift.x * (height / camMatrixLocal(0, 0) * multiplier));
  o_tran_corr.setY(avgShift.y * (height / camMatrixLocal(1, 1) * multiplier));
  o_tran_corr.setZ(0.0);

  o_tran_corr = -o_tran_corr / dur_.toSec();

  o_tran_diff = o_tran_corr - o_tran;

  return true;
}
//}

/* getRT() //{ */

bool OpticFlow::getRT(std::vector<cv::Point2d> shifts, double height, cv::Point2d ulCorner, tf2::Quaternion& o_rot, tf2::Vector3& o_tran) {
  if (!std::isfinite(1.0 / dur_.toSec())) {
    ROS_ERROR("[OpticFlow]:   Duration is %f. Returning.", dur_.toSec());
    return false;
  }

  cv::Matx33d camMatrixLocal = camMatrix_;
  camMatrixLocal(0, 2) -= ulCorner.x;
  std::vector<cv::Point2d> initialPts, shiftedPts, shiftsPassed, undistPtsA, undistPtsB;

  int sqNum = _frame_size_ / _sample_point_size_;

  for (int j = 0; j < sqNum; j++) {
    for (int i = 0; i < sqNum; i++) {

      if (!std::isfinite(shifts[i + sqNum * j].x) || !std::isfinite(shifts[i + sqNum * j].y)) {

        ROS_ERROR("[OpticFlow]: NaN detected in variable \"shifts[i + sqNum * j])\" - i = %d; j = %d!!!", i, j);
        continue;
      }

      int xi = i * _sample_point_size_ + (_sample_point_size_ / 2);
      int yi = j * _sample_point_size_ + (_sample_point_size_ / 2);
      initialPts.push_back(cv::Point2d(xi, yi));
      shiftedPts.push_back(cv::Point2d(xi, yi) + shifts[i + sqNum * j]);
      shiftsPassed.push_back(shifts[i + sqNum * j]);
    }
  }

  if (shiftedPts.size() < uint(_shifted_pts_thr_)) {
    ROS_ERROR("[OpticFlow]: shiftPts contains many NaNs, returning");
    return false;
  }

  cv::undistortPoints(initialPts, undistPtsA, camMatrixLocal, distCoeffs_);
  cv::undistortPoints(shiftedPts, undistPtsB, camMatrixLocal, distCoeffs_);

  /* std::cout << "Undist, vs orig: " << std::endl; */
  /* for (int i=0;i<(int)(undistPtsA.size()); i++){ */
  /*   std::cout << "A - Orig: " << initialPts[i] << " Undist: " << camMatrixLocal*undistPtsA[i] << std::endl; */
  /*   std::cout << "B - Orig: " << shiftedPts[i] << " Undist: " << camMatrixLocal*undistPtsB[i] << std::endl; */
  /* cv::Mat homography = cv::findHomography(undistPtsA, undistPtsB, 0, 3); */
  cv::Mat mask;
  cv::Mat homography = cv::findHomography(undistPtsA, undistPtsB, cv::RANSAC, 0.01, mask);

  bool allSmall  = false;
  uint remaining = 0;
  for (int z = 0; z < (int)(shiftedPts.size()); z++) {
    if (mask.at<unsigned char>(z) == 1) {
      remaining++;

      if (cv::norm(shiftsPassed[z]) > 0.01)
        allSmall = false;
    }
  }

  if (_debug_)
    ROS_INFO("[OpticFlow]: Motion estimated from %d points", remaining);


  if (remaining < uint(_shifted_pts_thr_)) {
    ROS_ERROR("[OpticFlow]: After RANSAC refinement, not enough points remain, returning");
    return false;
  }

  if (allSmall) {

    ROS_INFO("[OpticFlow]: No motion detected.");
    o_rot  = tf2::Quaternion(tf2::Vector3(0, 0, 1), 0);
    o_tran = tf2::Vector3(0, 0, 0);
    return true;
  }

  /* ROS_INFO_STREAM("[OpticFlow]: mask: "<< mask); */
  /* std::cout << "CamMat: " << camMatrixLocal << std::endl; */
  /* std::cout << "NO HOMO: " << homography << std::endl; */
  std::vector<cv::Mat> rot, tran, normals;
  int                  solutions = cv::decomposeHomographyMat(homography, cv::Matx33d::eye(), rot, tran, normals);
  std::vector<int>     filteredSolutions;

  tf2::Stamped<tf2::Transform> tempTfC2B, tempTfB2C;
  {
    std::scoped_lock lock(mutex_tf_);

    tf2::fromMsg(transformCam2Base_, tempTfC2B);
    tf2::fromMsg(transformBase2Cam_, tempTfB2C);
  }

  /* tf2::Quaternion cam_orientation = tempTf*odometry_orientation_; */
  /* tf2::Vector3 expNormal = */
  /* std::cout << "C2B: " << cam_orientation(tf2::Vector3(0,0,1)) << std::endl; */

  /* std::cout << "C2B: "<< tempTfC2B.getRotation().getAngle() << std::endl; */

  /* std::cout << "Next: " << std::endl; */
  double roll, pitch, yaw;
  {
    std::scoped_lock lock(mutex_angular_rate_);
    tf2::Matrix3x3(angular_rate_tf_).getRPY(roll, pitch, yaw);
  }
  /* std::cout << "Exp. rate: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
  /* tf2::Matrix3x3(angular_rate_tf_).getRPY(roll,pitch,yaw); */
  /* std::cout << "Exp. rate NEW: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
  tf2::Transform tempTransform;
  /* std::cout << std::endl; */

  int             bestIndex = -1;
  bool            bestInverseSolution;
  double          bestAngDiff = M_PI;
  tf2::Quaternion bestQuatRateOF;

  tf2::Quaternion bestQuatRateOF2;

  tf2::Quaternion quatRateOF, quatRateOFB;

  for (int i = 0; i < solutions; i++) {

    /* std::cout << normals[i] << std::endl; */
    /* std::cout << normals[i].at<double>(2) << std::endl; */

    /* if (normals[i].at<double>(2) <= DBL_EPSILON) { */
    bool inverseSolution = false;
    if (true) {

      tempTransform = tf2::Transform(cvMat33ToTf2Mat33(rot[i]));
      quatRateOF    = tempTransform.getRotation();


      quatRateOFB = tf2::Quaternion(tempTfC2B * (quatRateOF.getAxis()), quatRateOF.getAngle() / dur_.toSec());

      double angDiff;
      {
        std::scoped_lock lock(mutex_angular_rate_);
        double           angDiffPlus, angDiffMinus;
        angDiffPlus  = quatRateOFB.angle(angular_rate_tf_);
        angDiffMinus = quatRateOFB.angle(angular_rate_tf_.inverse());
        if (angDiffPlus < angDiffMinus) {
          angDiff = angDiffPlus;
        } else {
          angDiff = angDiffMinus;
        }

        if (normals[i].at<double>(2) < 0)
          inverseSolution = false;
        else
          inverseSolution = true;
      }
      /* std::cout << angDiff << std::endl; */

      if (bestAngDiff > angDiff) {
        bestAngDiff         = angDiff;
        bestIndex           = i;
        bestInverseSolution = inverseSolution;
        bestQuatRateOF      = quatRateOF;
      }
    }
  }


  if ((bestIndex != -1) && (solutions > 1)) {

    /* std::cout << normals[bestIndex] << std::endl; */
    /* std::cout << normals[bestIndex].at<double>(2) << std::endl; */

    if (cv::determinant(rot[bestIndex]) < 0) {
      /* std::cout << "Invalid rotation found" << std::endl; */
    }
    if (bestAngDiff > (M_PI / 4)) {
      ROS_WARN_THROTTLE(1.0, "[OpticFlow]: Angle difference greater than pi/4, skipping.");
      return false;
    }

    /* if (bestInverseSolution) { */
    /*   bestQuatRateOF = bestQuatRateOF.inverse(); */
    /* } */

    /* std::cout << "ANGLE: " << bestAngDiff << std::endl; */
    tf2::Matrix3x3(bestQuatRateOF).getRPY(roll, pitch, yaw);
    /* std::cout << "Angles  OF: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    tf2::Matrix3x3(angular_rate_tf_).getRPY(roll, pitch, yaw);
    /* std::cout << "Angles IMU: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    /* std::cout << "Normals: " << normals[bestIndex] << std::endl; */
    /* std::cout << std::endl; */
    /* std::cout << "Translations: " << tran[bestIndex] * uav_height_curr / dur_.toSec() << std::endl; */
    /* std::cout << std::endl; */

    o_rot = tf2::Quaternion(bestQuatRateOF.getAxis(), bestQuatRateOF.getAngle() / dur_.toSec());
    /* ROS_INFO("[OpticFlow]: o_rot: %f, %f, %f, %f ", o_rot.x(), o_rot.y(), o_rot.z(),o_rot.w() ); */

    /* { */
    /*   std::scoped_lock lock(mutex_uav_height_); */

    /*   o_tran =
     * tf2::Transform(bestQuatRateOF.inverse())*tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height_/dur_.toSec();
     */
    /* } */

    /* o_tran =
     * tf2::Transform(bestQuatRateOF.inverse())*tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height_curr/dur_.toSec();
     */
    /* o_tran = tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height_curr/dur_.toSec(); */

    double invUnit = (bestInverseSolution ? -1.0 : 1.0);
    o_tran         = tf2::Transform(bestQuatRateOF) *
             tf2::Vector3(invUnit * tran[bestIndex].at<double>(0), invUnit * tran[bestIndex].at<double>(1), invUnit * tran[bestIndex].at<double>(2)) * height /
             dur_.toSec();

    return true;

    /* if (bestIndex2 != -1) { */
    /*   std::cout << "ANGLE: " << bestAngDiff2 << std::endl; */
    /*   std::cout << "Det: " << cv::determinant(rot[bestIndex2]) << std::endl; */
    /*   tf2::Matrix3x3(bestQuatRateOF2).getRPY(roll,pitch,yaw); */
    /*   std::cout << "Angles  OF: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    /*   tf2::Matrix3x3(angular_rate_tf_).getRPY(roll,pitch,yaw); */
    /*   std::cout << "Angles IMU: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    /*   std::cout << "Normals: " << normals[bestIndex2] << std::endl; */
    /*   std::cout << std::endl; */
    /* } */
  }

  /* else if ((cv::norm(tran[0]) < 0.01) && (cv::norm(tran[2]) < 0.01)){ */
  else if (solutions == 1) {
    if (bestIndex == -1) {
      ROS_WARN("[OpticFlow]: Single solution found, but differs from IMU too much. Returning");
      return false;
    }

    /* if (cv::norm(tran[0]) < 0.001) { */
    /* std::cout << "No motion detected" << std::endl; */
    /* o_rot  = tf2::Quaternion(tf2::Vector3(0, 0, 1), 0); */
    /* o_tran = tf2::Vector3(0, 0, 0); */
    /* return true; */
    //

    o_rot = tf2::Quaternion(bestQuatRateOF.getAxis(), bestQuatRateOF.getAngle() / dur_.toSec());

    {
      std::scoped_lock lock(mutex_uav_height_);
      o_tran = tf2::Transform(bestQuatRateOF) * tf2::Vector3(tran[0].at<double>(0), tran[0].at<double>(1), tran[0].at<double>(2)) * height / dur_.toSec();
    }

    if (!std::isfinite(o_rot.x()) || !std::isfinite(o_rot.y()) || !std::isfinite(o_rot.z()) || !std::isfinite(o_rot.w()) || !std::isfinite(o_tran.x()) ||
        !std::isfinite(o_tran.y()) || !std::isfinite(o_tran.z())) {
      ROS_ERROR("[OpticFlow]: Single solution found, but contains NaNs/Infs. Returning");
      ROS_INFO("[OpticFlow]: o_rot: %f, %f, %f, %f ", o_rot.x(), o_rot.y(), o_rot.z(), o_rot.w());
      ROS_INFO("[OpticFlow]: o_tran: %f, %f, %f", o_tran.x(), o_tran.y(), o_tran.z());
      return false;
    }
    return true;
    /* } */

    ROS_ERROR("[OpticFlow]: Single solution found, but not small motion. ");
    ROS_INFO_STREAM("[OpticFlow]: rotMat = " << rot[0] << " tran = " << tran[0]);
    return false;
  }
  ROS_ERROR("[OpticFlow]: Unclassified motion estimation error, returning.");
  std::cout << "ERROR" << std::endl;
  return false;
}

//}

/* getExpectedShifts() //{ */

/* std::vector<cv::Point2d> getExpectedShifts(cv::Point2i ulCorner) { */
/*   camMatrixLocal = camMatrix_; */
/*   camMatrixLocal.at<double>(0, 2) -= ulCorner.x; */
/*   std::vector<cv::Point2d> initialPts, undistPts, differences; */
/*   std::vector<cv::Point3d> 3dPts, rotPts; */
/*   sqNum = frameSize / samplePointSize; */
/*   for (int i = 0; i < sqNum; i++) { */
/*     for (int j = 0; j < sqNum; j++) { */
/*       xi = i * samplePointSize; */
/*       yi = j * samplePointSize; */
/*       initialPts.push_back(cv::Point2f(xi, yi)); */
/*     } */
/*   } */
/*   cv::undistortPoints(initialPts, undistPts, camMatrixLocal, distCoeffs_); */
/*   for (const auto& ptCurr : undistPts) { */
/*     camMatrixLocal.inv() * cv::Point3d(ptCurr.x, ptCurr.y, 1) 3dPts.push_back(camMatrixLocal.inv() * cv::Point3d(ptCurr.x, ptCurr.y, 1)); */
/*   } */
/*   cv::Matx33d rotMatDiffCv = cv::Matx33d::zeros(); */
/*   for (int i = 0; i < 9; i++) { */
/*     output(i) = rotMatDiff_(3dPts, rotPts, rotMatDiffCv * rotZ()); */
/*   } */
/*   cv::transform(3dPts */
/* } */

//}

/* onInit() //{ */

void OpticFlow::onInit() {

  ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

  ros::Time::waitForValid();

  mrs_lib::ParamLoader param_loader(nh_, "OpticFlow");

  param_loader.load_param("version", _version_);

  if (_version_ != VERSION) {

    ROS_ERROR("[OpticFlow]: the version of the binary (%s) does not match the config file (%s), please build me!", VERSION, _version_.c_str());
    ros::shutdown();
  }

  // | -------------------- basic node params ------------------- |

  param_loader.load_param("uav_name", _uav_name_, std::string());
  param_loader.load_param("camera_frame", _camera_frame_);
  param_loader.load_param("camera_link_frame", _camera_link_frame_);
  param_loader.load_param("uav_frame", _uav_frame_);
  param_loader.load_param("uav_untilted_frame", _uav_untilted_frame_);
  param_loader.load_param("enable_profiler", _profiler_enabled_);
  param_loader.load_param("debug", _debug_);
  param_loader.load_param("gui", _gui_);
  param_loader.load_param("silent_debug", _silent_debug_);

  // | --------------------- general params --------------------- |

  param_loader.load_param("ang_rate_source", _ang_rate_source_);
  param_loader.load_param("raw_output", _raw_enabled_);

  param_loader.load_param("scale_rotation", _scale_rotation_);
  param_loader.load_param("scale_rot_magnitude", _scale_rotation_magnitude_);
  param_loader.load_param("scale_rot_output", _scale_rot_output_);
  /* param_loader.load_param("d3d_method", d3d_method_); */

  int videoFPS = param_loader.load_param2<int>("video_fps");

  // | -------------------- optic flow params ------------------- |

  param_loader.load_param("mrs_optic_flow/long_range_mode", _long_range_mode_string_);

  param_loader.load_param("FftCLFile", _fft_cl_file_);
  param_loader.load_param("useOCL", _useOCL_);

  param_loader.load_param("mrs_optic_flow/scale_factor", _scale_factor_);

  param_loader.load_param("mrs_optic_flow/shifted_pts_thr", _shifted_pts_thr_);

  param_loader.load_param("mrs_optic_flow/max_processing_rate", _max_processing_rate_);
  param_loader.load_param("mrs_optic_flow/method", _method_);
  param_loader.load_param("mrs_optic_flow/scan_radius", _scan_radius_);
  param_loader.load_param("mrs_optic_flow/step_size", _step_size_);
  param_loader.load_param("mrs_optic_flow/frame_size", _frame_size_);

  param_loader.load_param("mrs_optic_flow/takeoff_height", _takeoff_height_);

  if (fabs(_scale_factor_ - 1.0) > 0.01) {
    _frame_size_ = _frame_size_ / _scale_factor_;
  }

  param_loader.load_param("mrs_optic_flow/sample_point_size", _sample_point_size_);
  if (fabs(_scale_factor_ - 1.0) > 0.01) {
    _sample_point_size_ = _sample_point_size_ / _scale_factor_;
  }
  sample_point_size_lr     = _sample_point_size_ * 2;
  sample_point_count_sqrt_ = _frame_size_ / _sample_point_size_;
  sample_point_count_      = sample_point_count_sqrt_ * sample_point_count_sqrt_;
  param_loader.load_param("mrs_optic_flow/filter_method", _filter_method_);
  param_loader.load_param("mrs_optic_flow/apply_abs_bouding", _apply_abs_bounding_);
  param_loader.load_param("mrs_optic_flow/apply_rel_bouding", _apply_rel_bouding_);

  {
    double _calibration_coeff_both_;  // use this as a backup value in case calibrations for separate axes are not available
    param_loader.load_param("mrs_optic_flow/calibration/both_velocity_correction_ratio", _calibration_coeff_both_, 1.0);
    param_loader.load_param("mrs_optic_flow/calibration/x_velocity_correction_ratio", _calibration_coeff_x_, _calibration_coeff_both_);
    param_loader.load_param("mrs_optic_flow/calibration/y_velocity_correction_ratio", _calibration_coeff_y_, _calibration_coeff_both_);
  }

  param_loader.load_param("mrs_optic_flow/ransac/num_of_chosen", _ransac_num_of_chosen_);
  param_loader.load_param("mrs_optic_flow/ransac/num_of_iter", _ransac_num_of_iter_);
  _RansacThresholdRadSq_ = pow(param_loader.load_param2<double>("mrs_optic_flow/ransac/threshold_rad"), 2);

  param_loader.load_param("mrs_optic_flow/rotation_correction", _rotation_correction_);
  param_loader.load_param("mrs_optic_flow/filtering/analyze_duration", _analyze_duration_);
  // method check
  if (_method_ < 3 || _method_ > 5) {
    ROS_ERROR("[OpticFlow]: No such OpticFlow calculation method. Available: 3 = BM on CPU, 4 = FFT on CPU, 5 = BM on GPU via OpenCL");
  }

  // | ------------------------ filtering ----------------------- |

  param_loader.load_param("store_video", store_video_);
  std::string video_path_ = param_loader.load_param2<std::string>("video_path");

  // | ------------------ physical constraints ------------------ |

  param_loader.load_param("constraints/max_pixel_speed", _max_pixel_speed_);
  param_loader.load_param("constraints/max_horizontal_speed", _max_horizontal_speed_);
  param_loader.load_param("constraints/max_horizontal_acceleration", _max_horizontal_acceleration_);
  param_loader.load_param("constraints/max_vertical_speed", _max_vertical_speed_);
  param_loader.load_param("constraints/max_yaw_rate", _max_yaw_rate_);
  param_loader.load_param("constraints/speed_noise", _speed_noise_);

  // | --------------- fallback camera parameters --------------- |
  /* param_loader.load_param("camera_matrix/data", fallback_camera_data); */
  /* param_loader.load_param("distortion_coefficients/data", fallback_distortion_coeffs); */

  if (_gui_) {
    cv::namedWindow("ocv_optic_flow", cv::WINDOW_FREERATIO);
    /* cv::namedWindow("ocv_debugshit", cv::WINDOW_FREERATIO); */
    /* cv::namedWindow("OLD", cv::WINDOW_FREERATIO); */
    /* cv::namedWindow("ocv_NEW", cv::WINDOW_FREERATIO); */
    /* cv::namedWindow("ocv_iffc", cv::WINDOW_FREERATIO); */
  }

  /* if (_scale_rotation_ && (d3d_method_ == "advanced" || d3d_method_ == "logpol")) { */
  /*   ROS_ERROR("[OpticFlow]: Do not use R3xS1 estimation yet. Existing methods are logpol and advanced, but a better one - pnp - is comming soon. ~Viktor");
   */
  /*   ros::shutdown(); */
  /* } */

  // TODO this condition is possibly wrong, make the string comparison using ==
  if (_filter_method_.compare("ransac") && _ransac_num_of_chosen_ != 2) {
    ROS_WARN("[OpticFlow]: When Allsac is enabled, the _ransac_num_of_chosen_ can be only 2.");
  }

  if (store_video_) {
    ROS_INFO("[OpticFlow]: Video path: %s", video_path_.c_str());
  }

  if (RAND_MAX < 100) {
    ROS_WARN("[OpticFlow]: Why is RAND_MAX set to only %d? Ransac in OpticFlow won't work properly!", RAND_MAX);
  }

  if ((_frame_size_ % 2) == 1) {
    _frame_size_--;
  }
  scanDiameter = (2 * _scan_radius_ + 1);
  scanCount    = (scanDiameter * scanDiameter);

  // | -------------------- choose the method ------------------- |
  switch (_method_) {
    case 3: {
      ROS_ERROR("[OpticFlow]: Method 3 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
      /* processClass = new BlockMethod(_frame_size_, _sample_point_size_, _scan_radius_, scanDiameter, scanCount, _step_size_); */
      break;
    }
    case 4: {

      /* std::cout << cv::getBuildInformation() << std::endl; */

      if (_useOCL_ && !cv::ocl::haveOpenCL()) {
        ROS_ERROR(
            "[OpticFlow]: NO OCL SUPPORT - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to false.");
        return;
      }

      cv::ocl::Context context;
      if (_useOCL_ && !context.create(cv::ocl::Device::TYPE_GPU)) {
        ROS_ERROR(
            "Failed creating the context - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to false.");
        return;
      }

      if (_useOCL_ && (context.ndevices()) == 0) {
        ROS_ERROR(
            "[OpticFlow]: No OpenCL devices found - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to "
            "false.");
        return;
      }

      ROS_INFO("[OpticFlow]:  GPU devices are detected.");  // This bit provides an overview of the OpenCL devices you have in your computer
      for (int i = 0; i < int(context.ndevices()); i++) {
        cv::ocl::Device device = context.device(i);
        ROS_INFO("[OpticFlow]: name:              %s", device.name().c_str());
        if (device.available())
          ROS_INFO("[OpticFlow]: available!");
        else
          ROS_INFO("[OpticFlow]: unavailable");
        if (device.imageSupport())
          ROS_INFO("[OpticFlow]: image support!");
        else
          ROS_INFO("[OpticFlow]: no image support");
        ROS_INFO("[OpticFlow]: OpenCL_C_Version:  %s", device.OpenCL_C_Version().c_str());
      }

      cv::ocl::Device(context.device(0));  // Here is where you change which GPU to use (e.g. 0 or 1)

      cv::ocl::setUseOpenCL(true);

      fftProcessor_ = new FftMethod(_frame_size_, _sample_point_size_, _max_pixel_speed_, store_video_, _raw_enabled_, _rotation_correction_, false,
                                    &video_path_, videoFPS, _fft_cl_file_, _useOCL_);
      break;
    }

    case 5: {
      ROS_ERROR("[OpticFlow]: Method 5 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
      /* processClass = new FastSpacedBMMethod(_sample_point_size_, _scan_radius_, _step_size_, cx_, cy_, fx_, fy_, k1_, k2_, k3_, p1_, p2_, store_video_,
       * &video_path_);
       */
      break;
    }
    default: { throw std::invalid_argument("Invalid method ID!"); }
  }

  imPrev_ = cv::Mat(_frame_size_, _frame_size_, CV_8UC1);
  imPrev_ = cv::Scalar(0);
  fftProcessor_->setImPrev(imPrev_);

  begin_ = ros::Time::now();

  // prepare scale rotation estimator
  /* if (_scale_rotation_ && d3d_method_.compare("logpol") == STRING_EQUAL) { */
  /*   if (_scale_rot_output_.compare("velocity") != 0) { */
  /*     if (_scale_rot_output_.compare("altitude") != 0) { */
  /*       ROS_ERROR("[OpticFlow]: Wrong parameter _scale_rot_output_ - possible choices: velocity, altitude. Entered: %s", _scale_rot_output_.c_str()); */
  /*       ros::shutdown(); */
  /*     } */
  /*   } */
  /*   std::string sr_name      = video_path_.append("scale_rot.avi"); */
  /*   scale_rotation_estimator_ = new scaleRotationEstimator(_frame_size_, _scale_rotation_magnitude_, store_video_, &sr_name, videoFPS); */
  /* } */

  // | ----------------------- publishers ----------------------- |

  publisher_chosen_allsac_           = nh_.advertise<std_msgs::Int32>("allsac_chosen_out", 1);
  publisher_velocity_                = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("velocity_out", 1);
  publisher_velocity_longrange_      = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("velocity_out_longrange", 1);
  publisher_velocity_longrange_diff_ = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("velocity_out_longrange_diff", 1);
  publisher_velocity_std_            = nh_.advertise<geometry_msgs::Vector3>("velocity_stddev_out", 1);
  publisher_max_allowed_velocity_    = nh_.advertise<std_msgs::Float32>("max_velocity_out", 1);

  if (_raw_enabled_) {
    publisher_points_raw_ = nh_.advertise<std_msgs::UInt32MultiArray>("points_raw_out", 1);
  }

  // | ----------------------- subscribers ---------------------- |

  subscriber_tracker_status_ = nh_.subscribe("tracker_status_in", 1, &OpticFlow::callbackTrackerStatus, this, ros::TransportHints().tcpNoDelay());
  subscriber_camera_info_    = nh_.subscribe("camera_info_in", 1, &OpticFlow::callbackCameraInfo, this, ros::TransportHints().tcpNoDelay());
  subscriber_image_          = nh_.subscribe("camera_in", 1, &OpticFlow::callbackImage, this, ros::TransportHints().tcpNoDelay());
  ROS_INFO("[OpticFlow]: Image subscriber topic name is %s", subscriber_image_.getTopic().c_str());
  subscriber_uav_height_ = nh_.subscribe("uav_height_in", 1, &OpticFlow::callbackHeight, this, ros::TransportHints().tcpNoDelay());
  subscriber_odometry_   = nh_.subscribe("odometry_in", 1, &OpticFlow::callbackOdometry, this, ros::TransportHints().tcpNoDelay());
  nrep_                  = 0;

  if (_ang_rate_source_ == "imu") {
    subscriber_imu_ = nh_.subscribe("imu_in", 1, &OpticFlow::callbackImu, this);
  } else if (_ang_rate_source_ == "odometry_diff") {
  } else {
    if (_ang_rate_source_ != "odometry") {
      ROS_ERROR("[OpticFlow]: Wrong parameter _ang_rate_source_ - possible choices: imu, odometry, odometry_diff. Entered: %s", _ang_rate_source_.c_str());
      ros::shutdown();
    }
  }

  // | ----------------------- tf listener ---------------------- |

  /* tilt_curr_ = tf::Quaternion(); */
  /* tilt_prev_ = tilt_curr_; */
  listener_ = new tf2_ros::TransformListener(buffer_);

  // | ------------------------ profiler ------------------------ |

  profiler_ = new mrs_lib::Profiler(nh_, "OpticFlow", _profiler_enabled_);

  // | ------------------------- timers ------------------------- |

  timer_cam_init_ = nh_.createTimer(ros::Rate(10), &OpticFlow::timerCamInit, this);
  timer_tf_       = nh_.createTimer(ros::Rate(1), &OpticFlow::timerTf, this);

  // | ----------------------- finish init ---------------------- |

  if (!param_loader.loaded_successfully()) {
    ROS_ERROR("[OpticFlow]: Could not load all parameters!");
    ros::shutdown();
  }

  is_initialized_ = true;

  ROS_INFO("[OpticFlow]: initialized, version %s", VERSION);
}

//}

// --------------------------------------------------------------
// |                           timers                           |
// --------------------------------------------------------------

/* timerCamInit() //{ */

void OpticFlow::timerCamInit([[maybe_unused]] const ros::TimerEvent& event) {

  if (!is_initialized_)
    return;

  mrs_lib::Routine profiler_routine = profiler_->createRoutine("timerCamInit", 10.0, 0.004, event);

  if (!got_image_) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera");
    camera_info_timeout_ = ros::Time::now();
    return;
  }

  if (!got_camera_info_ && ((ros::Time::now() - camera_info_timeout_).toSec() < 15.0)) {

    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for camera info");
    return;
  }

  if (!got_camera_info_) {

    ROS_WARN("[OpticFlow]: missing camera calibration parameters! (nothing on camera_info topic/wrong calibration matricies). Loading default parameters");

    /* fx_ = fallback_camera_data[0]; */
    /* cx_ = fallback_camera_data[2]; */
    /* fy_ = fallback_camera_data[4]; */
    /* cy_ = fallback_camera_data[5]; */

    /* k1_ = fallback_distortion_coeffs[0]; */
    /* k2_ = fallback_distortion_coeffs[1]; */
    /* k3_ = fallback_distortion_coeffs[4]; */
    /* p1_ = fallback_distortion_coeffs[2]; */
    /* p2_ = fallback_distortion_coeffs[3]; */

    /* camMatrix_  = cv::Mat(3, 3, CV_64F, cv::Scalar(0)); */
    /* distCoeffs_ = cv::Mat(1, 5, CV_64F, cv::Scalar(0)); */

    /* camMatrix_.at<double>(0, 0) = fx_; */
    /* camMatrix_.at<double>(1, 1) = fy_; */
    /* camMatrix_.at<double>(0, 2) = cx_; */
    /* camMatrix_.at<double>(1, 2) = cy_; */
    /* camMatrix_.at<double>(2, 2) = 1; */

    /* distCoeffs_.at<double>(0) = k1_; */
    /* distCoeffs_.at<double>(1) = k2_; */
    /* distCoeffs_.at<double>(2) = p1_; */
    /* distCoeffs_.at<double>(3) = p2_; */
    /* distCoeffs_.at<double>(4) = k3_; */

    /* got_camera_info_ = true; */

  } else {
    ROS_INFO("[OpticFlow]: camera parameters loaded");
  }

  timer_cam_init_.stop();
}

//}

/* timerTf() //{ */

void OpticFlow::timerTf(const ros::TimerEvent& event) {

  if (!is_initialized_)
    return;

  mrs_lib::Routine profiler_routine = profiler_->createRoutine("timerTf", 1.0, 0.004, event);

  try {
    {
      std::scoped_lock lock(mutex_tf_);
      /* transformCam2Base_ = tf_buffer.lookupTransform(_uav_frame_, _camera_frame_, time - ros::Duration(uav2_delay), timeout); */
      transformCam2Base_ = buffer_.lookupTransform(_uav_frame_, _camera_frame_, ros::Time(0), ros::Duration(2));
    }
    ROS_INFO_STREAM("[OpticFlow]: received cam2base tf" << transformCam2Base_);

    double tf_roll, tf_pitch, tf_yaw;

    // calculate the euler angles
    tf::Quaternion quaternion_tf;
    quaternionMsgToTF(transformCam2Base_.transform.rotation, quaternion_tf);
    tf::Matrix3x3 m(quaternion_tf);
    m.getRPY(tf_roll, tf_pitch, tf_yaw);

    ROS_INFO("[OpticFlow]: R %f P %f Y %f", tf_roll, tf_pitch, tf_yaw);

    got_c2b_ = true;
  }
  catch (tf2::TransformException ex) {
    ROS_ERROR("[OpticFlow]: TF: %s", ex.what());
    ros::Duration(1.0).sleep();
    return;
  }

  tf::Quaternion quaternion_tf;
  double         dummy;
  try {
    {
      std::scoped_lock lock(mutex_tf_);
      transformBase2Cam_ = buffer_.lookupTransform(_camera_frame_, _uav_frame_, ros::Time(0), ros::Duration(2));
      /* transformBase2CamLink = buffer_.lookupTransform(_camera_link_frame_, _uav_frame_, ros::Time(0), ros::Duration(2)); */
      quaternionMsgToTF(transformBase2Cam_.transform.rotation, quaternion_tf);
      tf::Matrix3x3 m(quaternion_tf);
      m.getRPY(dummy, dummy, cam_yaw_);
      cam_yaw_ += M_PI_2;
    }

    ROS_INFO_STREAM("[OpticFlow]: received base2cam tf" << transformBase2Cam_);

    double tf_roll, tf_pitch, tf_yaw;


    // calculate the euler angles
    tf::Quaternion quaternion_tf;
    quaternionMsgToTF(transformBase2Cam_.transform.rotation, quaternion_tf);
    tf::Matrix3x3 m(quaternion_tf);
    m.getRPY(tf_roll, tf_pitch, tf_yaw);

    ROS_INFO("[OpticFlow]: R %f P %f Y %f", tf_roll, tf_pitch, tf_yaw);
    got_b2c_ = true;
  }
  catch (tf2::TransformException ex) {
    ROS_ERROR("[OpticFlow]: TF: %s", ex.what());
    ros::Duration(1.0).sleep();
    return;
  }

  // check whether we got everything we need
  // stop the timer if true
  if (got_c2b_ && got_b2c_) {

    ROS_INFO("[OpticFlow]: got TFs, stopping timerTf");

    got_tfs_ = true;

    delete listener_;

    timer_tf_.stop();
  }
}

//}

// --------------------------------------------------------------
// |                          callbacks                         |
// --------------------------------------------------------------

/* //{ callbackTrackerStatus() */

void OpticFlow::callbackTrackerStatus(const mrs_msgs::ControlManagerDiagnosticsConstPtr& msg) {

  if (!is_initialized_)
    return;

  mrs_lib::Routine profiler_routine = profiler_->createRoutine("callbackTrackerStatus");

  std::scoped_lock lock(mutex_tracker_status_);

  tracker_status_     = msg->tracker_status;
  got_tracker_status_ = true;
}
//}

/* callbackHeight() //{ */

void OpticFlow::callbackHeight(const mrs_msgs::Float64StampedConstPtr& msg) {

  if (!is_initialized_)
    return;

  mrs_lib::Routine routine_callback_height = profiler_->createRoutine("callbackUavHeight");

  if (absf(msg->value) < 0.001) {
    return;
  }

  std::scoped_lock lock(mutex_uav_height_);

  // IF YOU NEED TO PUT THIS BACK FOR ANY REASON, add mutex_odometry_ in the lock above!!!
  /* uav_height_ = msg->value; */
  /* if (!got_imu_) */
  /*   uav_height_ = msg->value; */
  /* else */
  /* uav_height_ = msg->value/(cos(odometry_pitch_)*(cos(odometry_roll_))); */

  uav_height_ = msg->value;

  got_height_ = true;
}

//}

/* callbackImu() //{ */

void OpticFlow::callbackImu(const sensor_msgs::ImuConstPtr& msg) {

  if (!is_initialized_)
    return;

  ROS_INFO_THROTTLE(1.0, "[OpticFlow]: getting IMU");

  mrs_lib::Routine routine_callback_imu = profiler_->createRoutine("callbackImu");

  // angular rate source is imu aka gyro
  if (_ang_rate_source_ == "imu") {

    {
      std::scoped_lock lock(mutex_angular_rate_);

      angular_rate_ = cv::Point3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
      angular_rate_tf_.setRPY(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    }

    got_imu_ = true;
  }

  {
    std::scoped_lock lock(mutex_static_tilt_);

    tf2::fromMsg(msg->orientation, imu_orientation_);
    tf2::Matrix3x3(imu_orientation_).getRPY(imu_roll_, imu_pitch_, imu_yaw_);
    /* std::cout << "OR IMUM CB: " << msg->orientation.x<< msg->orientation.y <<msg->orientation.z <<" - " << msg->orientation.w << std::endl; */
    /* std::cout << "OR IMU CB: " << imu_orientation_.getAxis().x() << imu_orientation_.getAxis().y() <<imu_orientation_.getAxis().z() <<" - " <<
     * imu_orientation_.getAngle() << std::endl; */
    /* std::cout << "RP IMU CB: " << imu_roll_ << " " << imu_pitch_ << std::endl; */
  }
  {
    std::scoped_lock lock(mutex_dynamic_tilt_);

    imu_roll_rate_  = imu_roll_rate_ * (1 - filter_ratio) + filter_ratio * msg->angular_velocity.x;
    imu_pitch_rate_ = imu_pitch_rate_ * (1 - filter_ratio) + filter_ratio * msg->angular_velocity.y;
    /* tf2::Matrix3x3(imu_orientation_).getRPY(imu_roll_, imu_pitch_, imu_yaw_); */
    /* std::cout << "OR IMUM CB: " << msg->orientation.x<< msg->orientation.y <<msg->orientation.z <<" - " << msg->orientation.w << std::endl; */
    /* std::cout << "OR IMU CB: " << imu_orientation_.getAxis().x() << imu_orientation_.getAxis().y() <<imu_orientation_.getAxis().z() <<" - " <<
     * imu_orientation_.getAngle() << std::endl; */
    /* std::cout << "RP IMU CB: " << imu_roll_ << " " << imu_pitch_ << std::endl; */
  }
}

//}

/* //{ callbackOdometry() */

void OpticFlow::callbackOdometry(const nav_msgs::OdometryConstPtr& msg) {

  if (!is_initialized_)
    return;

  mrs_lib::Routine routine_callback_odometry = profiler_->createRoutine("callbackOdometry");

  /* tf2::quaternionMsgToTF(msg->pose.pose.orientation, bt); */

  if (_ang_rate_source_ == "odometry") {
    {
      std::scoped_lock lock(mutex_angular_rate_);
      angular_rate_ = cv::Point3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
    }
  }

  {
    tf2::Quaternion bt;
    tf2::fromMsg(msg->pose.pose.orientation, bt);
    std::scoped_lock lock(mutex_odometry_);

    odometry_speed_       = cv::Point2d(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    odometry_stamp_       = ros::Time::now();
    odometry_orientation_ = bt;
    tf2::Matrix3x3(bt).getRPY(odometry_roll_, odometry_pitch_, odometry_yaw_);
  }

  got_odometry_ = true;
}

//}

/* //{ callbackImage() */

void OpticFlow::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

  /* imshow("NEW",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
  /* imshow("OLD",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
  /* imshow("cv_debugshit",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
  /* imshow("cv_optic_flow",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
  nrep_++;

  if (first_image_)
    begin_ = msg->header.stamp;

  ros::Time nowTime = msg->header.stamp;
  dur_              = nowTime - begin_;
  begin_            = nowTime;

  /* if ((nrep_ > 100) ) */
  /*   return; */

  if (!is_initialized_) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for initialization");
    return;
  }

  if (!got_odometry_) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for odometry");
    return;
  }

  if (!got_imu_) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for imu");
    return;
  }

  if ((!got_c2b_) || (!got_b2c_)) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for transform Base - Camera");
    return;
  }

  if (!std::isfinite(imu_roll_) || !std::isfinite(imu_pitch_)) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: Imu data contains NaNs...");
    return;
  }


  if (dur_.toSec() < 0.0 && !first_image_) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: time delta negative: %f", dur_.toSec());
    return;
  }

  if (fabs(dur_.toSec()) < 0.001 && !first_image_) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: time delta too small: %f", dur_.toSec());
    return;
  }

  mrs_lib::Routine routine_callback_image = profiler_->createRoutine("callbackImage");

  got_image_ = true;

  if (!first_image_)
    if (dur_.toSec() < 1 / _max_processing_rate_) {
      if (_debug_) {
        ROS_INFO("[OpticFlow]: MAX frequency overrun (%f). Skipping...", dur_.toSec());
      }
      return;
    }

  if (_debug_) {
    ROS_INFO_THROTTLE(1.0, "[OpticFlow]: freq = %fHz", 1.0 / dur_.toSec());
  }

  cv_bridge::CvImagePtr image;

  if (_ang_rate_source_ == "odometry_diff") {
    {
      std::scoped_lock lock(mutex_odometry_);
      tilt_curr_               = odometry_orientation_;
      tf2::Quaternion diffquat = tilt_prev_.inverse() * tilt_curr_;
      double          yaw_a, pitch_a, roll_a;
      tf2::Matrix3x3(diffquat).getRPY(roll_a, pitch_a, yaw_a);
      rotMatDiff_ = tf2::Matrix3x3(diffquat);
      angle_diff_ = cv::Point3d(roll_a, pitch_a, yaw_a);
    }
    tilt_prev_ = tilt_curr_;
  }
  image = cv_bridge::toCvCopy(msg, enc::BGR8);

  processImage(image);


  /* ROS_INFO("[OpticFlow]: callbackImage() end"); */
}

//}

/* callbackCameraInfo() //{ */

void OpticFlow::callbackCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg) {

  if (!is_initialized_)
    return;

  if (got_camera_info_) {
    return;
  }

  mrs_lib::Routine routine_camera_info = profiler_->createRoutine("callbackCameraInfo");

  // TODO: deal with binning
  if (msg->binning_x != 0) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: TODO: deal with binning when loading camera parameters.");
  }

  // check if the matricies have any data
  if (msg->K.size() < 6 || msg->D.size() < 5) {

    ROS_ERROR_THROTTLE(1.0, "[OpticFlow]: Camera info has wrong calibration matricies.");

  } else {

    fx_ = msg->K.at(0);
    fy_ = msg->K.at(4);
    cx_ = msg->K.at(2);
    cy_ = msg->K.at(5);

    k1_ = msg->D.at(0);
    k2_ = msg->D.at(1);
    p1_ = msg->D.at(2);
    p2_ = msg->D.at(3);
    k3_ = msg->D.at(4);

    camMatrix_                  = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
    distCoeffs_                 = cv::Mat(1, 5, CV_64F, cv::Scalar(0));
    camMatrix_.at<double>(0, 0) = fx_;
    camMatrix_.at<double>(1, 1) = fy_;
    camMatrix_.at<double>(0, 2) = cx_;
    camMatrix_.at<double>(1, 2) = cy_;
    camMatrix_.at<double>(2, 2) = 1;
    distCoeffs_.at<double>(0)   = k1_;
    distCoeffs_.at<double>(1)   = k2_;
    distCoeffs_.at<double>(2)   = p1_;
    distCoeffs_.at<double>(3)   = p2_;
    distCoeffs_.at<double>(4)   = k3_;
    got_camera_info_            = true;

    // maybe mutex this later

    if (_debug_) {
      ROS_INFO("[OpticFlow]: Camera params: %f %f %f %f %f %f %f %f %f", fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_);
    }
  }
}

//}

// --------------------------------------------------------------
// |                          routines                          |
// --------------------------------------------------------------

/* processImage() //{ */

void OpticFlow::processImage(const cv_bridge::CvImagePtr image) {


  // let's_ wait for two images
  if (first_image_) {
    first_image_ = false;
    return;
  }

  /* ROS_INFO("[OpticFlow]: processImage() start"); */

  // we need camera info!
  if (!got_camera_info_) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for camera info!");
    return;
  }

  // we need to know the UAV height
  if (!got_height_) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for uav height!");
    return;
  }

  if (!got_odometry_) {
    ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for odometry!");
    return;
  }

  double uav_height_curr;
  {
    std::scoped_lock lock(mutex_uav_height_);
    uav_height_curr = uav_height_;
  }

  bool long_range_mode;
  if (_long_range_mode_string_ == "always_on")
    long_range_mode = true;
  else if (_long_range_mode_string_ == "always_off")
    long_range_mode = false;
  else if (_long_range_mode_string_ == "takeoff_based")
    long_range_mode = isUavLandoff();
  else if (_long_range_mode_string_ == "height_based")
    long_range_mode = uav_height_curr < _takeoff_height_;
  else
    long_range_mode = false;


  if (_ang_rate_source_ == "odometry_diff") {
    {
      std::scoped_lock lock(mutex_odometry_);

      angle_diff_curr_ = angle_diff_;
    }
  } else {
    {
      std::scoped_lock lock(mutex_angular_rate_);

      angular_rate_curr_ = angular_rate_;
    }
  }

  // scale the image
  if (fabs(_scale_factor_ - 1.0) > 0.01) {
    cv::resize(image->image, image_scaled_, cv::Size(image->image.size().width / _scale_factor_, image->image.size().height / _scale_factor_));
  } else {
    image_scaled_ = image->image.clone();
  }

  // cropping
  /* int image_center_x = image_scaled_.size().width / 2; */
  int image_center_x = cx_;  // The distortion will be more symmetrical ->  better compensation
  int image_center_y = image_scaled_.size().height / 2;
  int xi             = image_center_x - (_frame_size_ / 2);
  int yi             = image_center_y - (_frame_size_ / 2);

  // rectification
  cv::Rect    cropping_rectangle = cv::Rect(xi, yi, _frame_size_, _frame_size_);
  cv::Point2i mid_point          = cv::Point2i((_frame_size_ / 2), (_frame_size_ / 2));

  //  convert to grayscale
  /* if (first_image_) */
  cv::cvtColor(image_scaled_(cropping_rectangle), imCurr_, CV_RGB2GRAY);

  // | ----------------- angular rate correction ---------------- |

  // Estimate scale and rotation (if enabled)
  cv::Point2d scale_and_rotation = cv::Point2d(0, 0);

  /* if (_scale_rotation_ && d3d_method_ == "logpol") { */

  /*   scale_and_rotation = scale_rotation_estimator_->processImage(imCurr_, _gui_, _debug_); */
  /*   scale_and_rotation.y /= dur_.toSec(); */

  /*   if (_scale_rot_output_ == "altitude") { */

  /*     // Altitude from velocity */

  /*     if (abs(scale_and_rotation.x - 1) > 0.01) { */
  /*       scale_and_rotation.x = 0;  //(Zvelocity*dur_.toSec())/(scale_and_rotation.x - 1); */
  /*     } else { */
  /*       // ROS_INFO("[OpticFlow]: Scale too small: %f",scale_and_rotation.x); */
  /*       scale_and_rotation.x = 0; */
  /*     } */

  /*   } else { */
  /*     // Velocity from altitude */

  /*     scale_and_rotation.x = ((scale_and_rotation.x - 1) / uav_height_curr) / dur_.toSec(); */
  /*   } */
  /* } */

  // process image
  std::vector<cv::Point2d> mrs_optic_flow_vectors;
  std::vector<cv::Point2d> mrs_optic_flow_vectors_raw;
  double                   temp_angle_diff;

  if (_ang_rate_source_ == "odometry_diff") {
    temp_angle_diff = angle_diff_curr_.z;
  } else {
    temp_angle_diff = angular_rate_curr_.z * dur_.toSec();
  }

  /* { */
  /*   std::scoped_lock lock(mutex_odometry_); */

  /*   cv::Point3d tilt_static = cv::Point3d(odometry_roll_, odometry_pitch_, odometry_yaw_); */
  /* } */

  /* std::vector<cv::Point3d> tempPts; */
  /* tempPts.push_back(cv::Point3d((rotX(-odometry_pitch_)*rotY(-odometry_roll_))*cv::Vec3d(0,0,1))); */
  /* std::vector<cv::Point2d> outputPts; */
  /* cv::projectPoints(tempPts,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),camMatrix_,distCoeffs_,outputPts); */
  /* cv::Point2d rot_center = outputPts[0]-cv::Point2d(xi, yi); */

  /* std::cout << "camMatrix_: " << camMatrix_ <<std::endl; */
  /* std::cout << "distCoeffs" << distCoeffs_ <<std::endl; */
  /* std::cout << "CENTER: " << rot_center <<std::endl; */

  tf2::Stamped<tf2::Transform> tempTfC2B;
  tf2::fromMsg(transformCam2Base_, tempTfC2B);

  {
    std::scoped_lock lock(mutex_process_);

    if (!long_range_mode)
      mrs_optic_flow_vectors =
          fftProcessor_->processImage(imCurr_, _gui_, _debug_, mid_point, temp_angle_diff, cv::Point(0, 0), mrs_optic_flow_vectors_raw, fx_, fy_);
    else
      mrs_optic_flow_vectors =
          fftProcessor_->processImageLongRange(imCurr_, _gui_, _debug_, mid_point, temp_angle_diff, cv::Point(0, 0), mrs_optic_flow_vectors_raw, fx_, fy_);
  }
  tf2::Quaternion                           rot;
  tf2::Vector3                              tran;
  geometry_msgs::TwistWithCovarianceStamped velocity;
  tf2::Quaternion                           detilt;

  {
    /* std::scoped_lock lock(mutex_odometry_); */

    std::scoped_lock lock(mutex_static_tilt_);

    // Velocities in the detilted body frame
    detilt.setRPY(imu_roll_, imu_pitch_, 0);
    // Velocities in the detilted global frame
    /* detilt.setRPY(imu_roll_, imu_pitch_, odometry_yaw_); */
    /* detilt.setRPY(imu_roll_,imu_pitch_,imu_yaw_); */
    /* detilt.setRPY(imu_roll_,imu_pitch_,0); */
    /* detilt.setRPY(odometry_roll_,odometry_pitch_,odometry_yaw_); */
    /* std::cout << "RP IMU: " << imu_roll_ << " " << imu_pitch_ << " " << imu_yaw_ << std::endl; */
  }

  /* detilt = detilt.inverse(); */
  /* detilt = tf2::Quaternion(tf2::Vector3(0,0,1),0); */
  /* std::cout << "Detilt: [" << odometry_roll_ << " " << odometry_pitch_ << " " << 0 << "]" << std::endl; */

  // if we got any data from the image
  // tran := translational speed
  if (!long_range_mode) {
    if (getRT(mrs_optic_flow_vectors, uav_height_curr, cv::Point2d(xi, yi), rot, tran)) {

      if (!std::isfinite(rot.x()) || !std::isfinite(rot.y()) || !std::isfinite(rot.z()) || !std::isfinite(rot.w()) || !std::isfinite(tran.x()) ||
          !std::isfinite(tran.y()) || !std::isfinite(tran.z())) {
        ROS_INFO("[OpticFlow]: tran: %f %f", tran.x(), tran.y());
        ROS_INFO("[OpticFlow]: rot: %f %f %f %f", rot.x(), rot.y(), rot.z(), rot.w());
        ROS_INFO("[OpticFlow]: Nans in output, returning.");
        return;
      }

      if ((tran.length()) > 7.0f) {
        ROS_INFO_STREAM("[OpticFlow]: LARGE SPEED: " << mrs_optic_flow_vectors);
      }

      if (_debug_) {
        ROS_INFO("[OpticFlow]: tran: %f %f", tran.x(), tran.y());
        ROS_INFO("[OpticFlow]: rot: %f %f %f %f", rot.x(), rot.y(), rot.z(), rot.w());
      }

      tran = tf2::Transform(detilt) * (tf2::Transform(tempTfC2B.getRotation()) * tran);
      /* std::cout << "Detilted: " << tran.x() << " " << tran.y() << " " << tran.z() << " " << std::endl; */

      /* double troll, tpitch, tyaw; */
      /* tf2::Matrix3x3(tempTfC2B.getRotation()).getRPY(troll,tpitch,tyaw); */
      /* std::cout << "C2B: " << troll << " " << tpitch << " "<< tyaw << " "<< std::endl; */

      /* rot = tf2::Quaternion(tf2::Transform(detilt)*tempTfC2B*(rot.getAxis()), rot.getAngle()); */
      rot = tf2::Quaternion(tempTfC2B * (rot.getAxis()), rot.getAngle());

      velocity.header.frame_id = _uav_untilted_frame_;
      velocity.header.stamp    = image->header.stamp;

      velocity.twist.twist.linear.x = tran.x();
      velocity.twist.twist.linear.y = tran.y();
      velocity.twist.twist.linear.z = tran.z();

      tf2::Matrix3x3(rot).getRPY(velocity.twist.twist.angular.x, velocity.twist.twist.angular.y, velocity.twist.twist.angular.z);

      velocity.twist.covariance[0]  = pow(50 * (uav_height_curr / fx_), 2);  // I expect error of 5 pixels. I presume fx_ and fy_ to be reasonably simillar.
      velocity.twist.covariance[7]  = velocity.twist.covariance[0];
      velocity.twist.covariance[14] = velocity.twist.covariance[0] * 2;

      velocity.twist.covariance[21] = atan(0.25);  // I expect error of 0.5 rad/s_.
      velocity.twist.covariance[28] = velocity.twist.covariance[21];
      velocity.twist.covariance[35] = velocity.twist.covariance[21];

      if (_debug_)
        if (fabs(tran.x()) <= 1e-5 || fabs(tran.y()) <= 1e-5 || fabs(tran.z()) <= 1e-5) {
          ROS_WARN("[OpticFlow]: OUTPUTTING ZEROS");
        }

      try {
        ROS_INFO_THROTTLE(1.0, "[OpticFlow]: publishing velocity");
        publisher_velocity_.publish(velocity);
      }
      catch (...) {
        ROS_ERROR("[OpticFlow]: exception caught during publishing topic %s", publisher_velocity_.getTopic().c_str());
      }
    }
  } else {
    tf2::Vector3 tran_diff;
    if (get2DT(mrs_optic_flow_vectors, uav_height_curr / (cos(imu_pitch_) * cos(imu_roll_)), cv::Point2d(xi, yi), tran, tran_diff)) {
      ROS_INFO_STREAM("vecs: " << mrs_optic_flow_vectors);

      if (!std::isfinite(tran.x()) || !std::isfinite(tran.y()) || !std::isfinite(tran.z())) {
        ROS_INFO("[OpticFlow]: tran: %f %f", tran.x(), tran.y());
        ROS_INFO("[OpticFlow]: Nans in output, returning.");
        return;
      }

      if ((tran.length()) > 7.0f) {
        ROS_INFO_STREAM("[OpticFlow]: LARGE SPEED: " << mrs_optic_flow_vectors);
      }

      if (_debug_) {
        ROS_INFO("[OpticFlow]: tran: %f %f", tran.x(), tran.y());
      }

      tran = (tf2::Transform(tempTfC2B.getRotation()) * tran);

      velocity.header.frame_id = _uav_frame_;
      velocity.header.stamp    = image->header.stamp;

      velocity.twist.twist.linear.x  = tran.x();
      velocity.twist.twist.linear.y  = tran.y();
      velocity.twist.twist.linear.z  = std::nan("");
      velocity.twist.twist.angular.x = std::nan("");
      ;
      velocity.twist.twist.angular.y = std::nan("");
      ;
      velocity.twist.twist.angular.z = std::nan("");
      ;

      velocity.twist.covariance[0]  = pow(50 * (uav_height_curr / fx_), 2);  // I expect error of 5 pixels. I presume fx_ and fy_ to be reasonably simillar.
      velocity.twist.covariance[7]  = velocity.twist.covariance[0];
      velocity.twist.covariance[14] = 666;

      velocity.twist.covariance[21] = 666;  // I expect error of 0.5 rad/s_.
      velocity.twist.covariance[28] = 666;
      velocity.twist.covariance[35] = 666;


      if (_debug_)
        if (fabs(tran.x()) <= 1e-5 || fabs(tran.y()) <= 1e-5 || fabs(tran.z()) <= 1e-5) {
          ROS_WARN("[OpticFlow]: OUTPUTTING ZEROS");
        }

      try {
        ROS_INFO("[OpticFlow]: long range mode: publishing velocity");
        /* ROS_INFO_THROTTLE(1.0, "[OpticFlow]: long range mode: publishing velocity"); */
        ROS_INFO_STREAM("[OpticFlow]: " << velocity.twist.twist.linear.x << " : " << velocity.twist.twist.linear.y);
        ROS_INFO_STREAM("[OpticFlow]: " << uav_height_curr << " : " << camMatrix_.at<double>(0, 0) << " : " << camMatrix_.at<double>(1, 1));
        publisher_velocity_longrange_.publish(velocity);
      }
      catch (...) {
        ROS_ERROR("[OpticFlow]: exception caught during publishing topic %s", publisher_velocity_longrange_.getTopic().c_str());
      }

      tran_diff = (tf2::Transform(tempTfC2B.getRotation()) * tran_diff);

      velocity.header.frame_id = _uav_frame_;
      velocity.header.stamp    = image->header.stamp;

      velocity.twist.twist.linear.x  = tran_diff.x();
      velocity.twist.twist.linear.y  = tran_diff.y();
      velocity.twist.twist.linear.z  = std::nan("");
      velocity.twist.twist.angular.x = std::nan("");
      ;
      velocity.twist.twist.angular.y = std::nan("");
      ;
      velocity.twist.twist.angular.z = std::nan("");
      ;

      velocity.twist.covariance[0]  = pow(50 * (uav_height_curr / fx_), 2);  // I expect error of 5 pixels. I presume fx_ and fy_ to be reasonably simillar.
      velocity.twist.covariance[7]  = velocity.twist.covariance[0];
      velocity.twist.covariance[14] = 666;

      velocity.twist.covariance[21] = 666;  // I expect error of 0.5 rad/s_.
      velocity.twist.covariance[28] = 666;
      velocity.twist.covariance[35] = 666;
      try {
        ROS_INFO("[OpticFlow]: long range mode: publishing velocity");
        /* ROS_INFO_THROTTLE(1.0, "[OpticFlow]: long range mode: publishing velocity"); */
        ROS_INFO_STREAM("[OpticFlow]: " << velocity.twist.twist.linear.x << " : " << velocity.twist.twist.linear.y);
        ROS_INFO_STREAM("[OpticFlow]: " << uav_height_curr << " : " << camMatrix_.at<double>(0, 0) << " : " << camMatrix_.at<double>(1, 1));
        publisher_velocity_longrange_diff_.publish(velocity);
      }
      catch (...) {
        ROS_ERROR("[OpticFlow]: exception caught during publishing topic %s", publisher_velocity_longrange_.getTopic().c_str());
      }
    }
  }
}

//}
}  // namespace mrs_optic_flow

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_optic_flow::OpticFlow, nodelet::Nodelet)
