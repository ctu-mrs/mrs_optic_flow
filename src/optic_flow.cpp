/* includes //{ */

#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <string>
#include <thread>
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

using namespace std;

//#include <opencv2/gpuoptflow.hpp>
//#include <opencv2/gpulegacy.hpp>
//#include <opencv2/gpuimgproc.hpp>
//#include <time.h>

#include "mrs_optic_flow/OpticFlowCalc.h"
#include "mrs_optic_flow/BlockMethod.h"
#include "mrs_optic_flow/FftMethod.h"
#include "mrs_optic_flow/utilityFunctions.h"
#include "mrs_optic_flow/scaleRotationEstimator.h"

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

#define STRING_EQUAL 0

namespace mrs_optic_flow
{

  cv::Matx33d rotX(double ang) {
    cv::Matx33d output = cv::Matx33d::zeros();
    output(0, 0)       = 1;
    output(1, 1)       = cos(ang);
    output(2, 2)       = cos(ang);
    output(2, 1)       = -sin(ang);
    output(1, 2)       = sin(ang);
    return output;
  }
  cv::Matx33d rotY(double ang) {
    cv::Matx33d output = cv::Matx33d::zeros();
    output(1, 1)       = 1;
    output(0, 0)       = cos(ang);
    output(2, 2)       = cos(ang);
    output(2, 0)       = sin(ang);
    output(0, 2)       = -sin(ang);
    return output;
  }

  struct PointValue
  {
    int         value;
    cv::Point2i location;
  };

  /* class OpticFlow //{ */

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

    int nrep;

  private:
    void        processImage(const cv_bridge::CvImagePtr image);
    bool        getRT(std::vector<cv::Point2d> shifts, cv::Point2d ulCorner, tf2::Quaternion& o_rot, tf2::Vector3& o_tran);
    void        TfThread();
    std::thread tf_thread;
    bool        got_tfs = false;

  private:
    ros::Timer cam_init_timer;
    void       camInitTimer(const ros::TimerEvent& event);

    tf2_ros::Buffer                 buffer;
    tf2_ros::TransformListener*     listener;
    geometry_msgs::TransformStamped transformCam2Base;
    geometry_msgs::TransformStamped transformBase2Cam;
    /* geometry_msgs::Transform  transformBase2World; */

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
    double     uav_height_curr;
    std::mutex mutex_uav_height;

    std::mutex mutex_process;

  private:
    tf2::Quaternion tilt_prev;
    tf2::Quaternion tilt_curr;
    cv::Point3d     angular_rate;
    tf2::Quaternion angular_rate_tf;
    cv::Point3d     angle_diff;
    tf2::Matrix3x3  rotMatDiff;
    cv::Point3d     angle_diff_curr;
    cv::Point3d     angular_rate_curr;
    std::mutex      mutex_angular_rate;
    std::mutex      mutex_static_tilt;

  private:
    tf2::Quaternion odometry_orientation;
    tf2::Quaternion imu_orientation;
    double          odometry_roll, odometry_pitch, odometry_yaw;
    double          imu_roll, imu_pitch, imu_yaw;
    double          odometry_roll_h, odometry_pitch_h, odometry_yaw_h;
    cv::Point2f     odometry_speed;
    ros::Time       odometry_stamp;
    std::mutex      mutex_odometry;
    std::mutex      mutex_tf;

  private:
    std::vector<double> fallback_camera_data;
    std::vector<double> fallback_distortion_coeffs;

  private:
    bool first_image = true;

    cv::Mat image_scaled;
    cv::Mat imCurr;
    cv::Mat imPrev;

    double vam;

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

    double  cx, cy, fx, fy, s;
    double  k1, k2, p1, p2, k3;
    cv::Mat camMatrix, distCoeffs;

    int    ransac_num_of_chosen_;
    int    ransac_num_of_iter_;
    double RansacThresholdRadSq;

    std::string camera_frame_, uav_frame_, uav_untilted_frame_;

    std::string fft_cl_file_;
    bool        useOCL_;

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
    double                analyze_duration_;

    double min_tilt_correction_;

  private:
    mrs_lib::Profiler* profiler;
    bool               profiler_enabled_ = false;

  private:
    bool        is_initialized = false;
    std::string uav_name;
  };

  //}

  /* TfThread() //{ */

  void OpticFlow::TfThread() {

    if (!is_initialized) {
      return;
    }

    ros::Rate transformRate(1.0);
    bool      got_c2b, got_b2c, got_b2w;
    got_c2b = false;
    got_b2c = false;
    got_b2w = false;

    while ((!got_c2b) || (!got_b2c)) {

      try {
        {
          std::scoped_lock lock(mutex_tf);
          /* transformCam2Base = tf_buffer.lookupTransform(uav_frame_, camera_frame_, time - ros::Duration(uav2_delay), timeout); */
          transformCam2Base = buffer.lookupTransform(uav_frame_, camera_frame_, ros::Time(0), ros::Duration(2));
        }
        ROS_INFO_STREAM("[OpticFlow]: received cam2base tf" << transformCam2Base);

        double tf_roll, tf_pitch, tf_yaw;

        // calculate the euler angles
        tf::Quaternion quaternion_tf;
        quaternionMsgToTF(transformCam2Base.transform.rotation, quaternion_tf);
        tf::Matrix3x3 m(quaternion_tf);
        m.getRPY(tf_roll, tf_pitch, tf_yaw);

        ROS_INFO("[OpticFlow]: R %f P %f Y %f", tf_roll, tf_pitch, tf_yaw);

        got_c2b = true;
      }
      catch (tf2::TransformException ex) {
        ROS_ERROR("TF: %s", ex.what());
        ros::Duration(1.0).sleep();
        continue;
      }

      try {
        {
          std::scoped_lock lock(mutex_tf);
          transformBase2Cam = buffer.lookupTransform(camera_frame_, uav_frame_, ros::Time(0), ros::Duration(2));
        }

        ROS_INFO_STREAM("[OpticFlow]: received base2cam tf" << transformBase2Cam);

        double tf_roll, tf_pitch, tf_yaw;

        // calculate the euler angles
        tf::Quaternion quaternion_tf;
        quaternionMsgToTF(transformBase2Cam.transform.rotation, quaternion_tf);
        tf::Matrix3x3 m(quaternion_tf);
        m.getRPY(tf_roll, tf_pitch, tf_yaw);

        ROS_INFO("[OpticFlow]: R %f P %f Y %f", tf_roll, tf_pitch, tf_yaw);
        got_b2c = true;
      }
      catch (tf2::TransformException ex) {
        ROS_ERROR("TF: %s", ex.what());
        ros::Duration(1.0).sleep();
        continue;
      }

      transformRate.sleep();
    }

    ROS_INFO("[OpticFlow]: got TFs, quiting tf thread");

    got_tfs = true;
  }

  //}

  /* getRT() //{ */

  bool OpticFlow::getRT(std::vector<cv::Point2d> shifts, cv::Point2d ulCorner, tf2::Quaternion& o_rot, tf2::Vector3& o_tran) {

    cv::Matx33d camMatrixLocal = camMatrix;
    camMatrixLocal(0, 2) -= ulCorner.x;
    std::vector<cv::Point2d> initialPts, shiftedPts, undistPtsA, undistPtsB;

    int sqNum = frame_size_ / sample_point_size_;

    for (int j = 0; j < sqNum; j++) {
      for (int i = 0; i < sqNum; i++) {

        if (!std::isfinite(shifts[i + sqNum * j].x) || !std::isfinite(shifts[i + sqNum * j].y)) {

          ROS_ERROR("NaN detected in variable \"shifts[i + sqNum * j])\" - i = %d; j = %d!!!", i,j);
          continue;
        }

        int xi = i * sample_point_size_ + (sample_point_size_ / 2);
        int yi = j * sample_point_size_ + (sample_point_size_ / 2);
        initialPts.push_back(cv::Point2d(xi, yi));
        shiftedPts.push_back(cv::Point2d(xi, yi) + shifts[i + sqNum * j]);
      }
    }

    // TODO: this number should be parametrized and put to config
    if (shiftedPts.size() < 8) {
      return false; 
    }

    cv::undistortPoints(initialPts, undistPtsA, camMatrixLocal, distCoeffs);
    cv::undistortPoints(shiftedPts, undistPtsB, camMatrixLocal, distCoeffs);

    /* std::cout << "Undist, vs orig: " << std::endl; */
    /* for (int i=0;i<(int)(undistPtsA.size()); i++){ */
    /*   std::cout << "A - Orig: " << initialPts[i] << " Undist: " << camMatrixLocal*undistPtsA[i] << std::endl; */
    /*   std::cout << "B - Orig: " << shiftedPts[i] << " Undist: " << camMatrixLocal*undistPtsB[i] << std::endl; */
    cv::Mat homography = cv::findHomography(undistPtsA, undistPtsB, 0, 3);
    /* std::cout << "CamMat: " << camMatrixLocal << std::endl; */
    /* std::cout << "NO HOMO: " << homography << std::endl; */
    std::vector<cv::Mat> rot, tran, normals;
    int                  solutions = cv::decomposeHomographyMat(homography, cv::Matx33d::eye(), rot, tran, normals);
    std::vector<int>     filteredSolutions;

    tf2::Stamped<tf2::Transform> tempTfC2B, tempTfB2C;
    {
      std::scoped_lock lock(mutex_tf);

      tf2::fromMsg(transformCam2Base, tempTfC2B);
      tf2::fromMsg(transformBase2Cam, tempTfB2C);
    }

    /* tf2::Quaternion cam_orientation = tempTf*odometry_orientation; */
    /* tf2::Vector3 expNormal = */
    /* std::cout << "C2B: " << cam_orientation(tf2::Vector3(0,0,1)) << std::endl; */

    /* std::cout << "C2B: "<< tempTfC2B.getRotation().getAngle() << std::endl; */

    /* std::cout << "Next: " << std::endl; */
    double roll, pitch, yaw;
    tf2::Matrix3x3(angular_rate_tf).getRPY(roll, pitch, yaw);
    /* std::cout << "Exp. rate: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    /* tf2::Matrix3x3(angular_rate_tf).getRPY(roll,pitch,yaw); */
    /* std::cout << "Exp. rate NEW: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
    tf2::Matrix3x3 tempRotMat;
    tf2::Transform tempTransform;
    /* std::cout << std::endl; */

    int             bestIndex   = -1;
    double          bestAngDiff = M_PI;
    tf2::Quaternion bestQuatRateOF;

    int             bestIndex2   = -1;
    double          bestAngDiff2 = M_PI;
    tf2::Quaternion bestQuatRateOF2;

    tf2::Quaternion quatRateOF, quatRateOFB;

    for (int i = 0; i < solutions; i++) {

      /* std::cout << normals[i] << std::endl; */
      /* std::cout << normals[i].at<double>(2) << std::endl; */

      if (normals[i].at<double>(2) < 0) {

        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            tempRotMat[k][j] = rot[i].at<double>(j, k);
          }
        }

        tempTransform = tf2::Transform(tempRotMat);
        quatRateOF    = tempTransform.getRotation();

        /* quatRateOFB = tf2::Quaternion((quatRateOF.getAxis()), quatRateOF.getAngle()/dur.toSec()); */

        // TODO: dur can be =0, 
        quatRateOFB = tf2::Quaternion(tempTfC2B * (quatRateOF.getAxis()), quatRateOF.getAngle() / dur.toSec());

        /* tf2::Matrix3x3(quatRateOFB).getRPY(roll,pitch,yaw); */
        /* std::cout << "Angles  OFT: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
        double angDiff = quatRateOFB.angle(angular_rate_tf);
        /* std::cout << angDiff << std::endl; */

        if (fabs(bestAngDiff - angDiff) < 0.0001) {
          /* std::cout << "SMALL DIFFERENCE" << std::endl; */
          bestAngDiff2    = angDiff;
          bestIndex2      = i;
          bestQuatRateOF2 = quatRateOF;
        } else if (bestAngDiff > angDiff) {
          bestAngDiff    = angDiff;
          bestIndex      = i;
          bestQuatRateOF = quatRateOF;
        }
      }
    }

    if ((bestIndex != -1) && (solutions > 1)) {

      if (cv::determinant(rot[bestIndex]) < 0) {
        /* std::cout << "Invalid rotation found" << std::endl; */
      }
      if (bestAngDiff > (M_PI / 4)) {
        ROS_WARN_THROTTLE(1.0, "[OpticFlow]: Angle difference greater than pi/4, skipping.");
        return false;
      }

      /* std::cout << "ANGLE: " << bestAngDiff << std::endl; */
      tf2::Matrix3x3(bestQuatRateOF).getRPY(roll, pitch, yaw);
      /* std::cout << "Angles  OF: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
      tf2::Matrix3x3(angular_rate_tf).getRPY(roll, pitch, yaw);
      /* std::cout << "Angles IMU: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
      /* std::cout << "Normals: " << normals[bestIndex] << std::endl; */
      /* std::cout << std::endl; */
      /* std::cout << "Translations: " << tran[bestIndex] * uav_height_curr / dur.toSec() << std::endl; */
      /* std::cout << std::endl; */

      // TODO: dur can be =0, 

      o_rot = tf2::Quaternion(bestQuatRateOF.getAxis(), bestQuatRateOF.getAngle() / dur.toSec());

      /* o_tran =
       * tf2::Transform(bestQuatRateOF.inverse())*tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height/dur.toSec();
       */

      // TODO: dur can be =0, 

      {
        std::scoped_lock lock(mutex_uav_height);
        /* o_tran =
         * tf2::Transform(bestQuatRateOF.inverse())*tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height_curr/dur.toSec();
         */
        o_tran = tf2::Transform(bestQuatRateOF) * tf2::Vector3(tran[bestIndex].at<double>(0), tran[bestIndex].at<double>(1), tran[bestIndex].at<double>(2)) *
                 uav_height_curr / dur.toSec();
        /* o_tran = tf2::Vector3(tran[bestIndex].at<double>(0),tran[bestIndex].at<double>(1),tran[bestIndex].at<double>(2))*uav_height_curr/dur.toSec(); */
      }

      return true;

      /* if (bestIndex2 != -1) { */
      /*   std::cout << "ANGLE: " << bestAngDiff2 << std::endl; */
      /*   std::cout << "Det: " << cv::determinant(rot[bestIndex2]) << std::endl; */
      /*   tf2::Matrix3x3(bestQuatRateOF2).getRPY(roll,pitch,yaw); */
      /*   std::cout << "Angles  OF: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
      /*   tf2::Matrix3x3(angular_rate_tf).getRPY(roll,pitch,yaw); */
      /*   std::cout << "Angles IMU: [" << roll << " " << pitch << " " << yaw << "]" << std::endl; */
      /*   std::cout << "Translations: " << tran[bestIndex2]*uav_height/dur.toSec() << std::endl; */
      /*   std::cout << "Normals: " << normals[bestIndex2] << std::endl; */
      /*   std::cout << std::endl; */
      /* } */
    }

    /* else if ((cv::norm(tran[0]) < 0.01) && (cv::norm(tran[2]) < 0.01)){ */
    else if (solutions == 1) {

      // TODO: do something which all shifts are small, then return zeros

      /* if (cv::norm(tran[0]) < 0.001) { */
      /*   std::cout << "No motion detected" << std::endl; */
      /*   o_rot  = tf2::Quaternion(tf2::Vector3(0, 0, 1), 0); */
      /*   o_tran = tf2::Vector3(0, 0, 0); */
      /*   return true; */
      /* } */

      ROS_INFO_STREAM("[OpticFlow]: shiftedPts: " << shiftedPts);

      ROS_INFO_STREAM("[OpticFlow]: homography: " << homography);

      return false;

    } else {
      std::cout << "ERROR" << std::endl;
    }

    return false;
  }

  //}

  /* getExpectedShifts() //{ */

  /* std::vector<cv::Point2d> getExpectedShifts(cv::Point2i ulCorner) { */
  /*   camMatrixLocal = camMatrix; */
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
  /*   cv::undistortPoints(initialPts, undistPts, camMatrixLocal, distCoeffs); */
  /*   for (const auto& ptCurr : undistPts) { */
  /*     camMatrixLocal.inv() * cv::Point3d(ptCurr.x, ptCurr.y, 1) 3dPts.push_back(camMatrixLocal.inv() * cv::Point3d(ptCurr.x, ptCurr.y, 1)); */
  /*   } */
  /*   cv::Matx33d rotMatDiffCv = cv::Matx33d::zeros(); */
  /*   for (int i = 0; i < 9; i++) { */
  /*     output(i) = rotMatDiff(3dPts, rotPts, rotMatDiffCv * rotZ()); */
  /*   } */
  /*   cv::transform(3dPts */
  /* } */

  //}

  /* onInit() //{ */

  void OpticFlow::onInit() {

    ros::NodeHandle nh_ = nodelet::Nodelet::getMTPrivateNodeHandle();

    ros::Time::waitForValid();

    mrs_lib::ParamLoader param_loader(nh_, "OpticFlow");

    // | -------------------- basic node params ------------------- |
    param_loader.load_param("uav_name", uav_name, std::string());
    param_loader.load_param("camera_frame", camera_frame_);
    param_loader.load_param("uav_frame", uav_frame_);
    param_loader.load_param("uav_untilted_frame", uav_untilted_frame_);
    param_loader.load_param("enable_profiler", profiler_enabled_);
    param_loader.load_param("debug", debug_);
    param_loader.load_param("gui", gui_);
    param_loader.load_param("silent_debug", silent_debug_);

    // | --------------------- general params --------------------- |

    param_loader.load_param("ang_rate_source", ang_rate_source_);
    param_loader.load_param("raw_output", raw_enabled_);
    param_loader.load_param("camera_yaw_offset", camera_yaw_offset_);

    param_loader.load_param("scale_rotation", scale_rotation);
    param_loader.load_param("scale_rot_magnitude", scale_rotation_magnitude_);
    param_loader.load_param("scale_rot_output", scale_rot_output_);
    param_loader.load_param("d3d_method", d3d_method_);

    int videoFPS = param_loader.load_param2<int>("video_fps");

    // | -------------------- optic flow params ------------------- |
    param_loader.load_param("FftCLFile", fft_cl_file_);
    param_loader.load_param("useOCL", useOCL_);

    param_loader.load_param("mrs_optic_flow/scale_factor", scale_factor_);

    param_loader.load_param("mrs_optic_flow/max_processing_rate", max_processing_rate_);
    param_loader.load_param("mrs_optic_flow/method", method_);
    param_loader.load_param("mrs_optic_flow/scan_radius", scan_radius_);
    param_loader.load_param("mrs_optic_flow/step_size", step_size_);
    param_loader.load_param("mrs_optic_flow/frame_size", frame_size_);

    if (fabs(scale_factor_ - 1.0) > 0.01) {
      frame_size_ = frame_size_ / scale_factor_;
    }

    param_loader.load_param("mrs_optic_flow/sample_point_size", sample_point_size_);
    if (fabs(scale_factor_ - 1.0) > 0.01) {
      sample_point_size_ = sample_point_size_ / scale_factor_;
    }
    sample_point_count_sqrt_ = frame_size_ / sample_point_size_;
    sample_point_count_      = sample_point_count_sqrt_ * sample_point_count_sqrt_;
    param_loader.load_param("mrs_optic_flow/filter_method", filter_method_);
    param_loader.load_param("mrs_optic_flow/apply_abs_bouding", apply_abs_bounding_);
    param_loader.load_param("mrs_optic_flow/apply_rel_bouding", apply_rel_bouding_);

    {
      double calibration_coeff_both;  // use this as a backup value in case calibrations for separate axes are not available
      param_loader.load_param("mrs_optic_flow/calibration/both_velocity_correction_ratio", calibration_coeff_both, 1.0);
      param_loader.load_param("mrs_optic_flow/calibration/x_velocity_correction_ratio", calibration_coeff_x_, calibration_coeff_both);
      param_loader.load_param("mrs_optic_flow/calibration/y_velocity_correction_ratio", calibration_coeff_y_, calibration_coeff_both);
    }


    param_loader.load_param("mrs_optic_flow/ransac/num_of_chosen", ransac_num_of_chosen_);
    param_loader.load_param("mrs_optic_flow/ransac/num_of_iter", ransac_num_of_iter_);
    RansacThresholdRadSq = pow(param_loader.load_param2<double>("mrs_optic_flow/ransac/threshold_rad"), 2);

    param_loader.load_param("mrs_optic_flow/rotation_correction", rotation_correction_);
    param_loader.load_param("mrs_optic_flow/tilt_correction", tilt_correction_);
    param_loader.load_param("mrs_optic_flow/minimum_tilt_correction", min_tilt_correction_, 0.01);
    param_loader.load_param("mrs_optic_flow/filtering/analyze_duration", analyze_duration_);
    // method check
    if (method_ < 3 || method_ > 5) {
      ROS_ERROR("[OpticFlow]: No such OpticFlow calculation method. Available: 3 = BM on CPU, 4 = FFT on CPU, 5 = BM on GPU via OpenCL");
    }

    // | ------------------------ filtering ----------------------- |


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
      cv::namedWindow("ocv_optic_flow", cv::WINDOW_FREERATIO);
      /* cv::namedWindow("ocv_debugshit", cv::WINDOW_FREERATIO); */
      /* cv::namedWindow("OLD", cv::WINDOW_FREERATIO); */
      /* cv::namedWindow("ocv_NEW", cv::WINDOW_FREERATIO); */
      /* cv::namedWindow("ocv_iffc", cv::WINDOW_FREERATIO); */
    }

    if (scale_rotation && (d3d_method_.compare("advanced") == 0 || d3d_method_.compare("logpol") == 0)) {
      ROS_ERROR("[OpticFlow]: Do not use R3xS1 estimation yet. Existing methods are logpol and advanced, but a better one - pnp - is comming soon. ~Viktor");
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
        ROS_ERROR("Method 3 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
        /* processClass = new BlockMethod(frame_size_, sample_point_size_, scan_radius_, scanDiameter, scanCount, step_size_); */
        break;
      }
      case 4: {

        /* std::cout << cv::getBuildInformation() << std::endl; */

        if (useOCL_ && !cv::ocl::haveOpenCL()) {
          ROS_ERROR("NO OCL SUPPORT - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to false.");
          return;
        }

        cv::ocl::Context context;
        if (useOCL_ && !context.create(cv::ocl::Device::TYPE_GPU)) {
          ROS_ERROR(
              "Failed creating the context - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to false.");
          return;
        }

        if (useOCL_ && (context.ndevices()) == 0) {
          ROS_ERROR(
              "No OpenCL devices found - cannot run with GPU acceleration. Consider running the CPU implementation by setting useOCL parameter to false.");
          return;
        }

        ROS_INFO(" GPU devices are detected.");  // This bit provides an overview of the OpenCL devices you have in your computer
        for (int i = 0; i < int(context.ndevices()); i++) {
          cv::ocl::Device device = context.device(i);
          ROS_INFO("name:              %s", device.name().c_str());
          if (device.available())
            ROS_INFO("available!");
          else
            ROS_INFO("unavailable");
          if (device.imageSupport())
            ROS_INFO("image support!");
          else
            ROS_INFO("no image support");
          ROS_INFO("OpenCL_C_Version:  %s", device.OpenCL_C_Version().c_str());
        }

        cv::ocl::Device(context.device(0));  // Here is where you change which GPU to use (e.g. 0 or 1)

        cv::ocl::setUseOpenCL(true);

        processClass = new FftMethod(frame_size_, sample_point_size_, max_pixel_speed_, store_video_, raw_enabled_, rotation_correction_, tilt_correction_,
                                     &video_path_, videoFPS, fft_cl_file_, useOCL_);
        break;
      }

      case 5: {
        ROS_ERROR("Method 5 is currently ON ICE. Use method 4, or get someone to fix the BlockMatching method");
        /* processClass = new FastSpacedBMMethod(sample_point_size_, scan_radius_, step_size_, cx, cy, fx, fy, k1, k2, k3, p1, p2, store_video_, &video_path_);
         */
        break;
      }
      default: { throw std::invalid_argument("Invalid method ID!"); }
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

    publisher_chosen_allsac        = nh_.advertise<std_msgs::Int32>("allsac_chosen_out", 1);
    publisher_velocity             = nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>("velocity_out", 1);
    publisher_velocity_std         = nh_.advertise<geometry_msgs::Vector3>("velocity_stddev_out", 1);
    publisher_max_allowed_velocity = nh_.advertise<std_msgs::Float32>("max_velocity_out", 1);
    publisher_tilt_correction      = nh_.advertise<geometry_msgs::Vector3>("tilt_correction_out", 1);

    if (raw_enabled_) {
      publisher_points_raw = nh_.advertise<std_msgs::UInt32MultiArray>("points_raw_out", 1);
    }

    // --------------------------------------------------------------
    // |                         subscribers                        |
    // --------------------------------------------------------------

    subscriber_camera_info = nh_.subscribe("camera_info_in", 1, &OpticFlow::callbackCameraInfo, this, ros::TransportHints().tcpNoDelay());
    subscriber_image       = nh_.subscribe("camera_in", 1, &OpticFlow::callbackImage, this, ros::TransportHints().tcpNoDelay());
    ROS_INFO("[OpticFlow]: Image subscriber topic name is %s", subscriber_image.getTopic().c_str());
    subscriber_uav_height = nh_.subscribe("uav_height_in", 1, &OpticFlow::callbackHeight, this, ros::TransportHints().tcpNoDelay());
    subscriber_odometry   = nh_.subscribe("odometry_in", 1, &OpticFlow::callbackOdometry, this, ros::TransportHints().tcpNoDelay());
    nrep                  = 0;

    if (ang_rate_source_.compare("imu") == STRING_EQUAL) {
      subscriber_imu = nh_.subscribe("imu_in", 1, &OpticFlow::callbackImu, this);
    } else if (ang_rate_source_.compare("odometry_diff") == STRING_EQUAL) {
    } else {
      if (ang_rate_source_.compare("odometry") != 0) {
        ROS_ERROR("[OpticFlow]: Wrong parameter ang_rate_source_ - possible choices: imu, odometry, odometry_diff. Entered: %s", ang_rate_source_.c_str());
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
    //
    if (!param_loader.loaded_successfully()) {
      ROS_ERROR("[OpticFlow]: Could not load all parameters!");
      ros::shutdown();
    }

    /* tilt_curr = tf::Quaternion(); */
    /* tilt_prev = tilt_curr; */
    listener       = new tf2_ros::TransformListener(buffer);
    tf_thread      = std::thread(&OpticFlow::TfThread, this);
    is_initialized = true;

    ROS_INFO("[OpticFlow]: initialized");
  }

  //}

  // --------------------------------------------------------------
  // |                           timers                           |
  // --------------------------------------------------------------

  /* camInitTimer() //{ */

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

      camMatrix  = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
      distCoeffs = cv::Mat(1, 5, CV_64F, cv::Scalar(0));

      camMatrix.at<double>(0, 0) = fx;
      camMatrix.at<double>(1, 1) = fy;
      camMatrix.at<double>(0, 2) = cx;
      camMatrix.at<double>(1, 2) = cy;
      camMatrix.at<double>(2, 2) = 1;

      distCoeffs.at<double>(0) = k1;
      distCoeffs.at<double>(1) = k2;
      distCoeffs.at<double>(2) = p1;
      distCoeffs.at<double>(3) = p2;
      distCoeffs.at<double>(4) = k3;

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

  /* callbackHeight() //{ */

  void OpticFlow::callbackHeight(const mrs_msgs::Float64StampedConstPtr& msg) {

    if (!is_initialized)
      return;

    mrs_lib::Routine routine_callback_height = profiler->createRoutine("callbackUavHeight");

    if (absf(msg->value) < 0.001) {
      return;
    }

    {
      std::scoped_lock lock(mutex_uav_height);
      /* uav_height = msg->value; */
      /* if (!got_imu) */
      /*   uav_height = msg->value; */
      /* else */
      {
        std::scoped_lock lock(mutex_odometry);
        /* uav_height = msg->value/(cos(odometry_pitch)*(cos(odometry_roll))); */
        uav_height = msg->value;
      }
    }
    got_height = true;
  }

  //}

  /* callbackImu() //{ */

  void OpticFlow::callbackImu(const sensor_msgs::ImuConstPtr& msg) {

    if (!is_initialized)
      return;

    mrs_lib::Routine routine_callback_imu = profiler->createRoutine("callbackImu");

    // angular rate source is imu aka gyro
    if (ang_rate_source_.compare("imu") == STRING_EQUAL) {

      {
        std::scoped_lock lock(mutex_angular_rate);

        angular_rate = cv::Point3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        angular_rate_tf.setRPY(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
      }

      got_imu = true;
    }

    {
      std::scoped_lock lock(mutex_static_tilt);

      tf2::fromMsg(msg->orientation, imu_orientation);
      tf2::Matrix3x3(imu_orientation).getRPY(imu_roll, imu_pitch, imu_yaw);
      /* std::cout << "OR IMUM CB: " << msg->orientation.x<< msg->orientation.y <<msg->orientation.z <<" - " << msg->orientation.w << std::endl; */
      /* std::cout << "OR IMU CB: " << imu_orientation.getAxis().x() << imu_orientation.getAxis().y() <<imu_orientation.getAxis().z() <<" - " <<
       * imu_orientation.getAngle() << std::endl; */
      /* std::cout << "RP IMU CB: " << imu_roll << " " << imu_pitch << std::endl; */
    }
  }

  //}

  /* //{ callbackOdometry() */

  void OpticFlow::callbackOdometry(const nav_msgs::OdometryConstPtr& msg) {

    if (!is_initialized)
      return;

    mrs_lib::Routine routine_callback_odometry = profiler->createRoutine("callbackOdometry");

    /* tf2::quaternionMsgToTF(msg->pose.pose.orientation, bt); */


    if (ang_rate_source_.compare("odometry") == STRING_EQUAL) {
      {
        std::scoped_lock lock(mutex_angular_rate);
        angular_rate = cv::Point3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);
      }
    }

    {
      tf2::Quaternion bt;
      tf2::fromMsg(msg->pose.pose.orientation, bt);
      std::scoped_lock lock(mutex_odometry);

      odometry_speed       = cv::Point2d(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
      odometry_stamp       = ros::Time::now();
      odometry_orientation = bt;
      tf2::Matrix3x3(bt).getRPY(odometry_roll, odometry_pitch, odometry_yaw);
    }

    got_odometry = true;
  }

  //}

  /* //{ callbackImage() */

  void OpticFlow::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

    /* imshow("NEW",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
    /* imshow("OLD",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
    /* imshow("cv_debugshit",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
    /* imshow("cv_optic_flow",cv::Mat(100,100,CV_8UC1,cv::Scalar(0))); */
    nrep++;

    /* if ((nrep > 100) ) */
    /*   return; */

    if (!is_initialized) {
      ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for initialization");
      return;
    }

    if (!got_odometry) {
      ROS_INFO_THROTTLE(1.0, "[OpticFlow]: waiting for odometry");
      return;
    }

    /* ROS_INFO("[OpticFlow]: callbackImage() start"); */

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

    if (ang_rate_source_.compare("odometry_diff") == STRING_EQUAL) {
      {
        std::scoped_lock lock(mutex_odometry);
        tilt_curr                = odometry_orientation;
        tf2::Quaternion diffquat = tilt_prev.inverse() * tilt_curr;
        double          yaw_a, pitch_a, roll_a;
        tf2::Matrix3x3(diffquat).getRPY(roll_a, pitch_a, yaw_a);
        rotMatDiff = tf2::Matrix3x3(diffquat);
        angle_diff = cv::Point3d(roll_a, pitch_a, yaw_a);
      }
      tilt_prev = tilt_curr;
    }
    image = cv_bridge::toCvCopy(msg, enc::BGR8);

    processImage(image);

    /* ROS_INFO("[OpticFlow]: callbackImage() end"); */
  }

  //}

  /* callbackCameraInfo() //{ */

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

      k1 = msg->D.at(0);
      k2 = msg->D.at(1);
      p1 = msg->D.at(2);
      p2 = msg->D.at(3);
      k3 = msg->D.at(4);

      camMatrix                  = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
      distCoeffs                 = cv::Mat(1, 5, CV_64F, cv::Scalar(0));
      camMatrix.at<double>(0, 0) = fx;
      camMatrix.at<double>(1, 1) = fy;
      camMatrix.at<double>(0, 2) = cx;
      camMatrix.at<double>(1, 2) = cy;
      camMatrix.at<double>(2, 2) = 1;
      distCoeffs.at<double>(0)   = k1;
      distCoeffs.at<double>(1)   = k2;
      distCoeffs.at<double>(2)   = p1;
      distCoeffs.at<double>(3)   = p2;
      distCoeffs.at<double>(4)   = k3;
      got_camera_info            = true;

      // maybe mutex this later

      if (debug_) {
        ROS_INFO("[OpticFlow]: Camera params: %f %f %f %f %f %f %f %f %f", fx, fy, cx, cy, k1, k2, p1, p2, k3);
      }
    }
  }

  //}

  // --------------------------------------------------------------
  // |                          routines                          |
  // --------------------------------------------------------------

  /* processImage() //{ */

  void OpticFlow::processImage(const cv_bridge::CvImagePtr image) {

    // let's wait for two images
    if (first_image) {
      first_image = false;
      return;
    }

    /* ROS_INFO("[OpticFlow]: processImage() start"); */

    // we need camera info!
    if (!got_camera_info) {
      return;
    }

    // we need to know the UAV height
    if (!got_height) {
      ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for uav height!");
      return;
    }

    if (!got_odometry) {
      ROS_WARN_THROTTLE(1.0, "[OpticFlow]: waiting for odometry!");
      return;
    }

    {
      std::scoped_lock lock(mutex_uav_height);
      uav_height_curr = uav_height;
    }

    if (ang_rate_source_.compare("odometry_diff") == STRING_EQUAL) {
      {
        std::scoped_lock lock(mutex_odometry);

        angle_diff_curr = angle_diff;
      }
    } else {
      {
        std::scoped_lock lock(mutex_angular_rate);

        angular_rate_curr = angular_rate;
      }
    }

    // scale the image
    if (fabs(scale_factor_ - 1.0) > 0.01) {
      cv::resize(image->image, image_scaled, cv::Size(image->image.size().width / scale_factor_, image->image.size().height / scale_factor_));
    } else {
      image_scaled = image->image.clone();
    }

    // cropping
    /* int image_center_x = image_scaled.size().width / 2; */
    int image_center_x = cx;  // The distortion will be more symmetrical ->  better compensation
    int image_center_y = image_scaled.size().height / 2;
    int xi             = image_center_x - (frame_size_ / 2);
    int yi             = image_center_y - (frame_size_ / 2);

    // rectification
    cv::Rect    cropping_rectangle = cv::Rect(xi, yi, frame_size_, frame_size_);
    cv::Point2i mid_point          = cv::Point2i((frame_size_ / 2), (frame_size_ / 2));

    //  convert to grayscale
    /* if (first_image) */
    cv::cvtColor(image_scaled(cropping_rectangle), imCurr, CV_RGB2GRAY);

    // | ----------------- angular rate correction ---------------- |

    cv::Point2d tiltCorr    = cv::Point2d(0, 0);
    cv::Point2d tiltCorrPix = cv::Point2d(0, 0);
    cv::Point2d tiltCorrVel = cv::Point2d(0, 0);

    if (tilt_correction_) {

      if (ang_rate_source_.compare("odometry_diff") == STRING_EQUAL) {
        tiltCorr.x = angle_diff_curr.x;
        tiltCorr.y = angle_diff_curr.y;
      } else {
        tiltCorr.x = angular_rate_curr.x;
        tiltCorr.y = angular_rate_curr.y;
      }
      tiltCorr.x = tan(tiltCorr.x);  // version V - the good one dammit
      tiltCorr.y = tan(-tiltCorr.y);

      rotate2d(tiltCorr.x, tiltCorr.y, -camera_yaw_offset_);

      tiltCorrPix.x = tiltCorr.x * fx;
      tiltCorrPix.y = tiltCorr.y * fy;

      /* if (fabs(tiltCorr.x)<min_tilt_correction_) tiltCorr.x=0; */
      /* if (fabs(tiltCorr.y)<min_tilt_correction_) tiltCorr.y=0; */

      /* double xTiltCorr = -fx * sqrt(2 - 2 * cos(angular_rate.x * dur.toSec())) * angular_rate.x / abs(angular_rate.x);  // version 5 */
      /* double yTiltCorr = fy * sqrt(2 - 2 * cos(angular_rate.y * dur.toSec())) * angular_rate.y / abs(angular_rate.y); */
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

        scale_and_rotation.x = ((scale_and_rotation.x - 1) / uav_height_curr) / dur.toSec();
      }
    }

    // process image
    std::vector<cv::Point2d> mrs_optic_flow_vectors;
    std::vector<cv::Point2d> mrs_optic_flow_vectors_raw;
    double                   temp_angle_diff;

    if (ang_rate_source_.compare("odometry_diff") == STRING_EQUAL) {
      temp_angle_diff = angle_diff_curr.z;
    } else {
      temp_angle_diff = angular_rate_curr.z * dur.toSec();
    }

    {
      std::scoped_lock lock(mutex_odometry);

      cv::Point3d tilt_static = cv::Point3d(odometry_roll, odometry_pitch, odometry_yaw);
    }

    /* std::vector<cv::Point3d> tempPts; */
    /* tempPts.push_back(cv::Point3d((rotX(-odometry_pitch)*rotY(-odometry_roll))*cv::Vec3d(0,0,1))); */
    /* std::vector<cv::Point2d> outputPts; */
    /* cv::projectPoints(tempPts,cv::Vec3d(0,0,0),cv::Vec3d(0,0,0),camMatrix,distCoeffs,outputPts); */
    /* cv::Point2d rot_center = outputPts[0]-cv::Point2d(xi, yi); */

    /* std::cout << "camMatrix: " << camMatrix <<std::endl; */
    /* std::cout << "distCoeffs" << distCoeffs <<std::endl; */
    /* std::cout << "CENTER: " << rot_center <<std::endl; */

    tf2::Stamped<tf2::Transform> tempTfC2B;
    tf2::fromMsg(transformCam2Base, tempTfC2B);

    {
      std::scoped_lock lock(mutex_process);

      mrs_optic_flow_vectors =
          processClass->processImage(imCurr, gui_, debug_, mid_point, temp_angle_diff, cv::Point(0, 0), tiltCorr, mrs_optic_flow_vectors_raw, fx, fy);
    }
    tf2::Quaternion                           rot;
    tf2::Vector3                              tran;
    geometry_msgs::TwistWithCovarianceStamped velocity;
    tf2::Quaternion                           detilt;

    {
      /* std::scoped_lock lock(mutex_odometry); */

      std::scoped_lock lock(mutex_static_tilt);

      /* detilt.setRPY(imu_roll,imu_pitch,0); */
      /* detilt.setRPY(imu_roll,imu_pitch,imu_yaw); */
      detilt.setRPY(imu_roll, imu_pitch, odometry_yaw);
      /* detilt.setRPY(imu_roll,imu_pitch,0); */
      /* detilt.setRPY(odometry_roll,odometry_pitch,odometry_yaw); */
      /* std::cout << "RP IMU: " << imu_roll << " " << imu_pitch << " " << imu_yaw << std::endl; */
    }

    /* detilt = detilt.inverse(); */
    /* detilt = tf2::Quaternion(tf2::Vector3(0,0,1),0); */
    /* std::cout << "Detilt: [" << odometry_roll << " " << odometry_pitch << " " << 0 << "]" << std::endl; */

    if (getRT(mrs_optic_flow_vectors, cv::Point2d(xi, yi), rot, tran)) {

      ROS_INFO("[OpticFlow]: tran: %f %f", tran.x(), tran.y());
      ROS_INFO("[OpticFlow]: rot: %f %f %f %f", rot.x(), rot.y(), rot.z(), rot.w());

      tran = tf2::Transform(detilt) * (tf2::Transform(tempTfC2B.getRotation()) * tran);
      /* std::cout << "Detilted: " << tran.x() << " " << tran.y() << " " << tran.z() << " " << std::endl; */

      /* double troll, tpitch, tyaw; */
      /* tf2::Matrix3x3(tempTfC2B.getRotation()).getRPY(troll,tpitch,tyaw); */
      /* std::cout << "C2B: " << troll << " " << tpitch << " "<< tyaw << " "<< std::endl; */

      /* rot = tf2::Quaternion(tf2::Transform(detilt)*tempTfC2B*(rot.getAxis()), rot.getAngle()); */
      rot = tf2::Quaternion(tempTfC2B * (rot.getAxis()), rot.getAngle());

      velocity.header.frame_id = uav_untilted_frame_;
      velocity.header.stamp    = ros::Time::now();

      velocity.twist.twist.linear.x = tran.x();
      velocity.twist.twist.linear.y = tran.y();
      velocity.twist.twist.linear.z = tran.z();

      tf2::Matrix3x3(rot).getRPY(velocity.twist.twist.angular.x, velocity.twist.twist.angular.y, velocity.twist.twist.angular.z);

      velocity.twist.covariance[0]  = pow(50 * (uav_height_curr / fx), 2);  // I expect error of 5 pixels. I presume fx and fy to be reasonably simillar.
      velocity.twist.covariance[7]  = velocity.twist.covariance[0];
      velocity.twist.covariance[14] = velocity.twist.covariance[0] * 2;

      velocity.twist.covariance[21] = atan(0.25);  // I expect error of 0.5 rad/s.
      velocity.twist.covariance[28] = velocity.twist.covariance[21];
      velocity.twist.covariance[35] = velocity.twist.covariance[21];

      if (fabs(tran.x()) <= 1e-5 || fabs(tran.y()) <= 1e-5 || fabs(tran.z()) <= 1e-5) {
        ROS_ERROR("[OpticFlow]: OUTPUTTING ZEROS");
      }

      try {
        publisher_velocity.publish(velocity);
      }
      catch (...) {
        ROS_ERROR("Exception caught during publishing topic %s.", publisher_velocity.getTopic().c_str());
      }
    }

    /* ROS_INFO("[OpticFlow]: processImage() end"); */

    return;

    /* THIS IS NOT USED //{ */

    // check for nans
    mrs_optic_flow_vectors = removeNanPoints(mrs_optic_flow_vectors);
    if (mrs_optic_flow_vectors.size() <= 0) {
      ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
      return;
    }

    // raw velocity without tilt corrections
    if (raw_enabled_) {

      if (method_ == 4 && ((int)(mrs_optic_flow_vectors.size()) != sample_point_count_)) {
        ROS_WARN("[OpticFlow]: Raw enabled and the processing function returned unexpected number of vectors. If this is not normal, disable raw veolcity.");
        return;
      }

      std_msgs::UInt32MultiArray msg_raw;

      msg_raw.layout.dim.push_back(std_msgs::MultiArrayDimension());
      msg_raw.layout.dim.push_back(std_msgs::MultiArrayDimension());
      msg_raw.layout.dim[0].size   = mrs_optic_flow_vectors_raw.size();
      msg_raw.layout.dim[0].label  = "count";
      msg_raw.layout.dim[0].stride = mrs_optic_flow_vectors_raw.size() * 2;
      msg_raw.layout.dim[1].size   = 2;
      msg_raw.layout.dim[1].label  = "value";
      msg_raw.layout.dim[1].stride = 2;
      std::vector<unsigned int> convert;
      for (int i = 0; i < (int)(mrs_optic_flow_vectors_raw.size()); i++) {
        convert.push_back(mrs_optic_flow_vectors_raw[i].x);
        convert.push_back(mrs_optic_flow_vectors_raw[i].y);
      }
      msg_raw.data = convert;

      try {
        publisher_points_raw.publish(msg_raw);
      }
      catch (...) {
        ROS_ERROR("Exception caught during publishing topic %s.", publisher_points_raw.getTopic().c_str());
      }
    }

    // | ----------------- advanced 3D positioning ---------------- |
    if (scale_rotation && d3d_method_.compare("advanced") == STRING_EQUAL) {

      ROS_ERROR("[opticflow]: This implementation of R3xS1 motion called \"advanced\" is stupid. Wait for a new one, named \"pnp\". ~Viktor");
      return;

      if (mrs_optic_flow_vectors.size() != 9) {
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

      trvv = estimateTranRotVvel(mrs_optic_flow_vectors, (double)sample_point_size_, fx, fy, uav_height_curr, RansacThresholdRadSq, dur.toSec(),
                                 max_vertical_speed_, max_yaw_rate_);

      mrs_optic_flow_vectors.clear();
      mrs_optic_flow_vectors.push_back(trvv[0]);  // translation in px
      scale_and_rotation.x = trvv[1].y;           // rotation in rad/s
      scale_and_rotation.y = trvv[1].x;           // vertical velocity

    } else {

      mrs_optic_flow_vectors = removeNanPoints(mrs_optic_flow_vectors);

      if (mrs_optic_flow_vectors.size() <= 0) {
        ROS_WARN("[OpticFlow]: Processing function returned no valid points!");
        return;
      }

      std::vector<cv::Point2d> physical_speed_vectors;
      // scale the velocity using height
      physical_speed_vectors = multiplyAllPts(mrs_optic_flow_vectors, uav_height_curr / (fx * dur.toSec()), -uav_height_curr / (fy * dur.toSec()),
                                              false);  // -x fixes the difference in chirality between the image axes and the XY plane of the UAV.

      // rotate by camera yaw
      rotateAllPts(physical_speed_vectors, -camera_yaw_offset_ - (M_PI_2));  // -pi/2 to turn X,Y into the body axes.

      // rotate to global system
      rotateAllPts(physical_speed_vectors, odometry_yaw);  // is it wise to do here? It should be done with tf, and the output should be in the body frame, or
                                                           // even better, the camera frame! TODO

      // apply odometry calibration coefficients
      multiplyAllPts(mrs_optic_flow_vectors, calibration_coeff_x_, calibration_coeff_y_);

      // Print output
      if (debug_) {
        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: After recalc.");
        for (uint i = 0; i < physical_speed_vectors.size(); i++) {
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: %d -> vxr = %f; vyr=%f", i, physical_speed_vectors[i].x, physical_speed_vectors[i].y);
        }
      }

      // FILTERING
      // Bound physical_speed_vectors
      ros::Duration timeDif               = ros::Time::now() - odometry_stamp;
      float         max_sp_dif_from_accel = max_horizontal_acceleration_ * timeDif.toSec() + speed_noise;  // speed_noise is always significantly hihger

      // Backup physical_speed_vectors for silent debug_
      std::vector<cv::Point2d> bck_speeds;
      if (silent_debug_) {
        bck_speeds = physical_speed_vectors;
      }

      int af_abs = 0;
      int af_acc = 0;

      // absolute bouding
      //
      if (apply_abs_bounding_) {

        // Bounding of physical_speed_vectors, if enabled
        if (debug_)
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds before bound #%lu", physical_speed_vectors.size());

        physical_speed_vectors = getOnlyInAbsBound(physical_speed_vectors, max_horizontal_speed_);  // bound according to max speed
        if (silent_debug_) {
          af_abs = physical_speed_vectors.size();
        }
        if (debug_) {
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after speed bound #%lu, max speed: %f", physical_speed_vectors.size(), max_horizontal_speed_);
        }
      } else if (debug_) {
        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Maximum speed-based bounding of physical_speed_vectors not enabled.");
      }


      // relative bounding
      if (apply_rel_bouding_) {

        {
          std::scoped_lock lock(mutex_odometry);
          physical_speed_vectors = getOnlyInRadiusFromExpected(odometry_speed, physical_speed_vectors, max_sp_dif_from_accel);
        }

        if (silent_debug_) {
          af_acc = physical_speed_vectors.size();
        }

        if (debug_) {
          ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Speeds after acceleration bound #%lu, max speed from acc: %f", physical_speed_vectors.size(),
                            max_sp_dif_from_accel);
        }

        if (physical_speed_vectors.size() < 1) {
          ROS_WARN("[OpticFlow]: No physical_speed_vectors after bounding, can't publish!");

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
        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: Acceleration-based bounding of physical_speed_vectors not enabled.");
      }

      // post-process by Allsac/Ransac/Averaging

      // apply Allsac/Ransac/Averaging
      cv::Point2d filtered_speed_vector;
      if (filter_method_.compare("average") == STRING_EQUAL) {

        filtered_speed_vector = pointMean(physical_speed_vectors);

      } else if (filter_method_.compare("allsac") == STRING_EQUAL) {

        int chosen;
        filtered_speed_vector = allsacMean(physical_speed_vectors, RansacThresholdRadSq, &chosen);
        std_msgs::Int32 allsacChosen;
        allsacChosen.data = chosen;
        publisher_chosen_allsac.publish(allsacChosen);

      } else if (filter_method_.compare("ransac") == STRING_EQUAL) {

        filtered_speed_vector = ransacMean(physical_speed_vectors, ransac_num_of_chosen_, RansacThresholdRadSq, ransac_num_of_iter_);

      } else {
        ROS_ERROR("[OpticFlow]: Entered filtering method (filter_method_) does not match to any of these: average,ransac,allsac.");
      }


      tiltCorrVel.x = tiltCorr.x * uav_height_curr;
      tiltCorrVel.y = tiltCorr.y * uav_height_curr;

      geometry_msgs::Vector3 tiltCorrOut;
      tiltCorrOut.x =
          tiltCorrVel.x;  // (tan(angular_rate.y*dur.toSec())*uav_height_curr)/dur.toSec(); // if enabling, dont forget to mutex range and angular_rate
      tiltCorrOut.y =
          tiltCorrVel.y;  // (tan(angular_rate.x*dur.toSec())*uav_height_curr)/dur.toSec(); // if enabling, dont forget to mutex range and angular_rate
      tiltCorrOut.z = 0;
      publisher_tilt_correction.publish(tiltCorrOut);

      /* if (tilt_correction_) */
      /*   filtered_speed_vector = filtered_speed_vector  + tiltCorrVel; */

      vam = sqrt(getNormSq(filtered_speed_vector));

      // | -------------------- publish velocity -------------------- |
      geometry_msgs::TwistWithCovarianceStamped velocity;

      velocity.header.frame_id = "local_origin";
      velocity.header.stamp    = ros::Time::now();

      velocity.twist.twist.linear.x = filtered_speed_vector.x;
      velocity.twist.twist.linear.y = filtered_speed_vector.y;
      velocity.twist.twist.linear.z = 0;
      /* velocity.twist.linear.z  = scale_and_rotation.x; */
      /* velocity.twist.angular.z = -scale_and_rotation.y; */
      velocity.twist.twist.linear.z  = 0;
      velocity.twist.twist.angular.z = 0;
      velocity.twist.covariance[0]   = pow(10 * (uav_height_curr / fx), 2);  // I expect error of 5 pixels. I presume fx and fy to be reasonably simillar.
      velocity.twist.covariance[7]   = velocity.twist.covariance[0];
      publisher_velocity.publish(velocity);
      if (debug_) {

        ROS_INFO_THROTTLE(0.1, "[OpticFlow]: vxm = %f; vym=%f; vam=%f; range=%f; odometry_yaw=%f", filtered_speed_vector.x, filtered_speed_vector.y, vam,
                          uav_height_curr, odometry_yaw);
      }

      // Add speedbox to lastspeeds array - speedbox carries time, optflow velocity and odom. velocity
      SpeedBox sb;
      sb.time  = ros::Time::now();
      sb.speed = filtered_speed_vector;
      {
        std::scoped_lock lock(mutex_odometry);

        sb.odometry_speed = odometry_speed;
      }

      ros::Time fromTime = sb.time - ros::Duration(analyze_duration_);
      if ((int)(lastSpeeds.size()) > 0)
        while (lastSpeeds.begin()->time < fromTime) {
          lastSpeeds.erase(lastSpeeds.begin());
        }

      lastSpeeds.push_back(sb);

      // Create statistical data

      StatData sd = analyzeSpeeds(fromTime, lastSpeeds);  // check what bullshit this one contains later TODO

      geometry_msgs::Vector3 v3;
      v3.x = sd.stdDevX;
      v3.y = sd.stdDevY;
      v3.z = sd.stdDev;

      publisher_velocity_std.publish(v3);

      if (method_ == 5) {
        // ON ICE
        std_msgs::Float32 maxVel;
        {
          std::scoped_lock lock(mutex_uav_height);

          // TODO: dur can be =0, 

          maxVel.data = scan_radius_ * uav_height_curr / (dur.toSec() * std::max(fx, fy));
        }
        publisher_max_allowed_velocity.publish(maxVel);
      }
    }

    //}
  }

  //}
}  // namespace mrs_optic_flow

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mrs_optic_flow::OpticFlow, nodelet::Nodelet)
