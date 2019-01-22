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
      ROS_ERROR("[OpticFlow]: Could not open output video file: %s", videoPath->c_str());
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
    ROS_INFO("[OpticFlow]: FFT method - rotation correction enabled");
  }
  if (tilt_corr_enable) {
    ROS_INFO("[OpticFlow]: FFT method - tilt correction enabled");
  }
  first = true;
}

std::vector<cv::Point2d> FftMethod::processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr, std::vector<cv::Point2d> &raw_output, double i_fx, double i_fy) {

  /* ROS_INFO("FX:%f, FY%f",i_fx,i_fy); */

  fx = i_fx;
  fy = i_fy;


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
    ROS_INFO("[OpticFlow]: Curr type: %d prev type: %d", imCurr.type(), imPrev.type());
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
      shift = cv::phaseCorrelate(imPrevF(cv::Rect(xi, yi, samplePointSize, samplePointSize)), imCurrF(cv::Rect(xi, yi, samplePointSize, samplePointSize)));

      bool valid=true;
      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
          absd(shift.y) > ((double)samplePointSize / 2)) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        valid=false;
      }

      if (raw_enable) {
        raw_output.push_back(cv::Point2d(shift.x, shift.y));  // normal operation
      }

      if (rot_corr_enable) {
        // rotation correction
        distX = (xi + samplePointSize / 2) - midX;
        distY = (yi + samplePointSize / 2) - midY;

        corrX = -distY * yaw_angle;
        corrY = distX * yaw_angle;

        shift.x = shift.x + corrX;
        shift.y = shift.y + corrY;
      }


      if (tilt_corr_enable) {
        distX = fabs( (xi + samplePointSize / 2) - midX);
        distY = fabs( (yi + samplePointSize / 2) - midY);

        /* double spDist = sqrt(pow(fx,2)+pow(xi,2)+pow((fx/fy)*yi,2)); */
        cv::Point2d tiltCorrCurrSample;
        tiltCorrCurrSample.x = tan(atan(distX/fx)+tiltCorr.x)*fx-distX;
        tiltCorrCurrSample.y = tan(atan(distY/fy)+tiltCorr.y)*fy-distY;
        shift = shift + tiltCorrCurrSample;
      }

      // ROS_INFO("[OpticFlow]: i %d j %d -> xi:%d yi:%d, velo: %f %f px",i,j,xi,yi,shift.x,shift.y);

      // ROS_WARN("[OpticFlow]: Hacks going on..."); // hack for Gazebo Mobius
      // shift.x = - shift.x;
      // shift.y = - shift.y;

      if (!valid) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        speeds.push_back(cv::Point2d(nan(""), nan("")));
      } else {
        speeds.push_back(cv::Point2d(shift.x, shift.y));
      }

      // draw nice lines if gui is enabled
      if (gui || storeVideo) {
        cv::line(imView, cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2),
                 cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(shift.x * 5.0), (int)(shift.y * 5.0)), cv::Scalar(255),valid?2:1);
      }
    }
  }

  cv::line(imView, cv::Point2i(imView.size()/2), cv::Point2i(imView.size()/2)+cv::Point2i(tan(tiltCorr.x)*fx*5,tan(tiltCorr.y)*fy*5), cv::Scalar(255),5);

  imPrev = imCurr.clone();

  if (gui) {
    cv::imshow("optic_flow", imView);
    cv::waitKey(1);
  }

  if (storeVideo) {
    outputVideo << imView;
  }

  return speeds;
}
