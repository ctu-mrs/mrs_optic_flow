#include "../include/optic_flow/FftMethod.h"


FftMethod::FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
                     bool i_tilt_corr_enable, std::string *videoPath, int videoFPS) {


  frameSize       = i_frameSize;
  samplePointSize = i_samplePointSize;
  max_px_speed_sq = pow(max_px_speed_t, 2);

  storeVideo = i_storeVideo;
  if (storeVideo) {
    outputVideo.open(*videoPath, CV_FOURCC('M', 'P', 'E', 'G'), videoFPS, cv::Size(frameSize, frameSize), false);
    if (!outputVideo.isOpened())
      ROS_ERROR("Could not open output video file: %s", videoPath->c_str());
  }

  if ((frameSize % 2) == 1) {
    frameSize--;
  }
  if ((frameSize % samplePointSize) != 0) {
    samplePointSize = frameSize;
    ROS_WARN(
        "Oh, what kind of setting for OpticFlow is this? Frame size must be a multiple of SamplePointSize! Forcing FrameSize = SamplePointSize (i.e. one "
        "window)..");
  }


  sqNum            = frameSize / samplePointSize;
  raw_enable       = i_raw_enable;
  rot_corr_enable  = i_rot_corr_enable;
  tilt_corr_enable = i_tilt_corr_enable;
  if (rot_corr_enable) {
    ROS_INFO("FFT method - rotation correction enabled");
  }
  if (tilt_corr_enable) {
    ROS_INFO("FFT method - tilt correction enabled");
  }
  first = true;
}

std::vector<cv::Point2f> FftMethod::processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr) {

  // save image for GUI
  if (gui || storeVideo) {
    imView = imCurr.clone();
  }

  // copy first to second
  if (first) {
    imCurr.copyTo(imPrev);
    first = false;
  }

  if (debug) {
    ROS_INFO("Curr type: %d prev type: %d", imCurr.type(), imPrev.type());
  }

  // convert images to float images
  cv::Mat imCurrF, imPrevF;
  imCurr.convertTo(imCurrF, CV_32FC1);
  imPrev.convertTo(imPrevF, CV_32FC1);

  // clear the vector with speeds
  speeds.clear();

  double midX = imCurr.cols / 2;
  double midY = imCurr.rows / 2;

  double distX, distY;  // distance from middle
  double corrX, corrY;  // yaw corrections

  // calculate correlation for each window and store it if it doesn't exceed the limit
  for (int i = 0; i < sqNum; i++) {
    for (int j = 0; j < sqNum; j++) {
      xi    = i * samplePointSize;
      yi    = j * samplePointSize;
      shift = cv::phaseCorrelate(imCurrF(cv::Rect(xi, yi, samplePointSize, samplePointSize)), imPrevF(cv::Rect(xi, yi, samplePointSize, samplePointSize)));

      if (raw_enable) {
        // push without correction first
        if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
            absd(shift.y) > ((double)samplePointSize / 2)) {
          ROS_WARN("FFT - invalid correlation in window x %d y %d", i, j);
          speeds.push_back(cv::Point2f(nan(""), nan("")));
        } else {
          // ROS_WARN("Hacks going on in raw...");  // hack for Gazebo Mobius
          // speeds.push_back(cv::Point2f(-shift.x,-shift.y));
          speeds.push_back(cv::Point2f(shift.x, shift.y));  // normal operation
        }
      }

      if (rot_corr_enable) {
        // rotation correction
        distX = xi + samplePointSize / 2 - midX;
        distY = midY - (yi + samplePointSize / 2);

        corrX = distY * yaw_angle;
        corrY = distX * yaw_angle;

        shift.x = shift.x + corrX;
        shift.y = shift.y + corrY;
      }


      if (tilt_corr_enable) {
        shift = shift + tiltCorr;  // should be + for bluefox, - for Gazebo Mobius
      }

      // ROS_INFO("i %d j %d -> xi:%d yi:%d, velo: %f %f px",i,j,xi,yi,shift.x,shift.y);

      // ROS_WARN("Hacks going on..."); // hack for Gazebo Mobius
      // shift.x = - shift.x;
      // shift.y = - shift.y;

      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
          absd(shift.y) > ((double)samplePointSize / 2)) {
        ROS_WARN("FFT - invalid correlation in window x %d y %d", i, j);
        speeds.push_back(cv::Point2f(nan(""), nan("")));
      } else {
        speeds.push_back(cv::Point2f(shift.x, shift.y));
      }

      // draw nice lines if gui is enabled
      if (gui || storeVideo) {
        cv::line(imView, cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2),
                 cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(shift.x * 5.0), (int)(shift.y * 5.0)), cv::Scalar(255));
      }
    }
  }

  imPrev = imCurr.clone();

  if (gui) {
    cv::imshow("main", imView);
    cv::waitKey(1);
  }

  if (storeVideo) {
    outputVideo << imView;
  }

  return speeds;
}
