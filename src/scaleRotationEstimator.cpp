#include <scaleRotationEstimator.h>

scaleRotationEstimator::scaleRotationEstimator(int res, double m, bool i_storeVideo, std::string *videoPath, int videoFPS) {

  resolution = res;

  // set video output
  storeVideo = i_storeVideo;
  if (storeVideo) {
#ifdef ROS_MELODIC
    outputVideo.open(*videoPath, CV_FOURCC('M', 'P', 'E', 'G'), videoFPS, cv::Size(resolution, resolution), false);
#endif
#ifdef ROS_NOETIC
    outputVideo.open(*videoPath, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), videoFPS, cv::Size(resolution, resolution), false);
#endif
    if (!outputVideo.isOpened())
      ROS_ERROR("[OpticFlow]: Could not open output video file: %s", videoPath->c_str());
  }

  // assuming image resoultion is square
  // optimM = ((double)resolution) / (log((double)resolution)/log(M_E));
  optimM = m;
  ROS_INFO("[OpticFlow]: ScaleRotation estimator -> Optimal LogPol scale: %f\n", optimM);

  center = cv::Point2f(resolution / 2, resolution / 2);
  Ky     = ((double)resolution) / 360.0;
  tempIm = cv::Mat::zeros(cv::Size(resolution, resolution), CV_8UC1);

  // magLL = cv::Mat::zeros(cv::Size(resolution,resolution), CV_32FC1);

  first = true;
}

cv::Point2d scaleRotationEstimator::processImage(cv::Mat imCurr, bool gui, [[maybe_unused]] bool debug) {

  if (first) {

// old code
#ifdef ROS_MELODIC
    ipl_ta = imCurr;
    ipl_tb = tempIm;
    cvLogPolar(&ipl_ta, &ipl_tb, center, optimM, CV_INTER_CUBIC);
#endif
#ifdef ROS_NOETIC
    cv::logPolar(imCurr, tempIm, center, optimM, cv::INTER_CUBIC);
#endif

    tempIm.convertTo(prevIm_F32, CV_32FC1);

    /*
    imCurr.convertTo(tempIm,CV_32FC1);

    cv::Mat planes[] = {cv::Mat_<float>(tempIm), cv::Mat::zeros(tempIm.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    cv::Mat mag = planes[0];

    cv::Laplacian(mag,mag,CV_32FC1);

    ipl_ta = mag;
    ipl_tb = magLL;

    cvLogPolar(&ipl_ta,&ipl_tb,center,optimM);

    magLL_prev = magLL.clone();
    */

    first = false;
    return cv::Point2d(1, 0);
  }

  // FFT - LogPol - PhaseCorr
  /*    imCurr.convertTo(tempIm,CV_32FC1);

      cv::Mat planes[] = {cv::Mat_<float>(tempIm), cv::Mat::zeros(tempIm.size(), CV_32F)};
      cv::Mat complexI;
      cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
      cv::dft(complexI, complexI);            // this way the result may fit in the source matrix
      cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
      cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
      cv::Mat mag = planes[0];

      cv::Laplacian(mag,mag,CV_32FC1);

      //
      ipl_ta = mag;
      ipl_tb = magLL;
      cvLogPolar(&ipl_ta,&ipl_tb,center,optimM);

      cv::Point2d pt = cv::phaseCorrelate(magLL, magLL_prev);
      double scale = exp(pt.x/optimM);
      double rotat = (pt.y/Ky)*(M_PI/180);

      cv::Point2d out = cv::Point2d(scale,rotat);

      //ROS_INFO("[OpticFlow]: sc: %f, rot: %f",scale,rotat);

      magLL_prev = magLL.clone();*/


  // OLD thing
#ifdef ROS_MELODIC
    ipl_ta = imCurr;
    ipl_tb = tempIm;
    cvLogPolar(&ipl_ta, &ipl_tb, center, optimM, CV_INTER_LANCZOS4);
#endif
#ifdef ROS_NOETIC
    cv::logPolar(imCurr, tempIm, center, optimM, cv::INTER_LANCZOS4);
#endif

  tempIm.convertTo(tempIm_F32, CV_32FC1);

  cv::Point2d pt = cv::phaseCorrelate(tempIm_F32, prevIm_F32);

  if (abs(pt.x) > resolution / 2 || abs(pt.x) > resolution / 2) {
    return cv::Point2d(1, 0);
  }

  double scale = exp(pt.x / optimM);
  double rotat = (pt.y / Ky) * (M_PI / 180);

  cv::Point2d out = cv::Point2d(scale, rotat);

  prevIm_F32 = tempIm_F32.clone();

  if (gui) {
    cv::line(tempIm, cv::Point2i(resolution / 2, resolution / 2),
             cv::Point2i(resolution / 2, resolution / 2) + cv::Point2i((int)(pt.x * 5.0), (int)(pt.y * 5.0)), cv::Scalar(255));

    cv::imshow("LogPol tf", tempIm);

    /*magLL += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log(magLL, magLL);
    cv::normalize(magLL, magLL, 0, 1, CV_MINMAX);
    imshow("spectrum magnitude", magLL);*/

    cv::waitKey(1);
  }
  if (storeVideo) {
    outputVideo << tempIm;
  }

  return out;
}
