#include <BlockMethod.h>

BlockMethod::BlockMethod(int i_frameSize, int i_samplePointSize, int i_scanRadius, int i_scanDiameter, int i_scanCount, int i_stepSize) {
  frameSize       = i_frameSize;
  samplePointSize = i_samplePointSize;
  scanRadius      = i_scanRadius;
  scanDiameter    = i_scanDiameter;
  scanCount       = i_scanCount;
  stepSize        = i_stepSize;

  maxSamplesSide = (frameSize - scanRadius * 2) / (samplePointSize);

  midPoint = cv::Point2i((frameSize / 2), (frameSize / 2));

  xHist = new int[scanDiameter];
  yHist = new int[scanDiameter];

  imPrev = cv::Mat(frameSize, frameSize, CV_8UC1);
  imPrev = cv::Scalar(0);

  // ROS_WARN("[OpticFlow]: Block Matching wasn't rewritten for Allsac yet!!! Please don't use it...");
}


std::vector<cv::Point2d> BlockMethod::processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d tiltCorr) {

  // ROS_WARN("[OpticFlow]: Really, you are using BM while it's not rewritten yet? C'mon...rewrite it for ALLsac");

  // save image for gui
  if (gui)
    imView = imCurr.clone();

  // fill histogram with zeros
  for (int i = 0; i < scanDiameter; i++) {
    xHist[i] = 0;
    yHist[i] = 0;
  }


  cv::Point2i startPos;
  int         index;

  for (int m = 0; m < maxSamplesSide; m++) {
    for (int n = 0; n < maxSamplesSide; n++) {
      startPos    = cv::Point2i(n * (samplePointSize) + scanRadius, m * (samplePointSize) + scanRadius);
      absDiffsMat = cv::Mat(scanDiameter, scanDiameter, CV_32S);

      index = 0;

      for (int i = -scanRadius; i <= scanRadius; i++) {

        for (int j = -scanRadius; j <= scanRadius; j++) {
          cv::absdiff(imCurr(cv::Rect(startPos, cv::Size(samplePointSize, samplePointSize))),
                      imPrev(cv::Rect(startPos + cv::Point2i(j, i), cv::Size(samplePointSize, samplePointSize))), imDiff);
          absDiffsMat.at<int32_t>(cv::Point2i(scanRadius, scanRadius) + cv::Point2i(j, i)) = cv::sum(imDiff)[0];
          index++;
        }
      }

      double    min, max;
      cv::Point min_loc, max_loc;

      cv::minMaxLoc(absDiffsMat, &min, &max, &min_loc, &max_loc);

      xHist[min_loc.x]++;
      yHist[min_loc.y]++;


      if (gui)
        cv::line(imView, startPos + cv::Point2i(samplePointSize / 2, samplePointSize / 2),
                 startPos + cv::Point2i(samplePointSize / 2, samplePointSize / 2) + min_loc - cv::Point2i(scanRadius, scanRadius), cv::Scalar(255));
    }
  }

  int outputX = std::distance(xHist, std::max_element(xHist, xHist + scanDiameter)) - scanRadius;
  int outputY = std::distance(yHist, std::max_element(yHist, yHist + scanDiameter)) - scanRadius;
  // ROS_INFO("[OpticFlow]: x = %d; y = %d\n",outputX,outputY);

  cv::Point2d refined = Refine(imCurr, imPrev, cv::Point2i(outputX, outputY), 2);


  if (gui) {
    cv::line(imView, midPoint, midPoint + cv::Point2i((int)(refined.x * 4), (int)(refined.y * 4)), cv::Scalar(255), 2);
    cv::imshow("main", imView);
    cv::waitKey(10);
  }


  imPrev = imCurr.clone();
  std::vector<cv::Point2d> ret;
  ret.push_back(refined);
  return ret;
  // return refined;
}

cv::Point2d BlockMethod::Refine(cv::Mat imCurr, cv::Mat imPrev, cv::Point2i fullpixFlow, int passes) {
  cv::Mat imCurr2x = imCurr.clone();
  ;
  cv::Mat imPrev2x = imPrev.clone();
  cv::Mat imDiff2x;

  cv::Point2i totalOffset = fullpixFlow;

  int pixScale = 1;

  for (int i = 1; i <= passes; i++) {
    pixScale    = pixScale * 2;
    totalOffset = totalOffset * 2;

    cv::resize(imCurr2x, imPrev2x, cv::Size(imPrev.size().width * 2, imPrev.size().height * 2));  // optimalizuj -uloz aj neskreslene
    cv::resize(imCurr2x, imCurr2x, cv::Size(imCurr.size().width * 2, imCurr.size().height * 2));

    cv::Point2i startpoint;
    if ((totalOffset.x < 0) && (totalOffset.y < 0))
      startpoint = cv::Point2i(-totalOffset.x + 1, -totalOffset.y + 1);
    else if ((totalOffset.x < 0) && (totalOffset.y >= 0))
      startpoint = cv::Point2i(-totalOffset.x + 1, 1);
    else if ((totalOffset.x >= 0) && (totalOffset.y < 0))
      startpoint = cv::Point2i(1, -totalOffset.y + 1);
    else
      startpoint = cv::Point2i(1, 1);

    cv::Size2i cutoutSize = cv::Size2i(imCurr2x.size().width - (abs(totalOffset.x) + 2), imCurr2x.size().height - (abs(totalOffset.y) + 2));


    absDiffsMatSubpix = cv::Mat(3, 3, CV_32S);
    for (int m = -1; m <= 1; m++) {
      for (int n = -1; n <= 1; n++) {
        // ROS_INFO("[OpticFlow]: m=%d, n=%d, scale=%d, tx=%d, ty=%d",m,n,pixScale,totalOffset.x,totalOffset.y);
        // ROS_INFO("[OpticFlow]: spx=%d, spy=%d, szx=%d, szy=%d",startpoint.x,startpoint.y,cutoutSize.width,cutoutSize.height);

        cv::absdiff(imCurr2x(cv::Rect(cv::Point2i(1, 1), cutoutSize)), imPrev2x(cv::Rect(startpoint + cv::Point2i(n, m), cutoutSize)), imDiff2x);

        absDiffsMatSubpix.at<int32_t>(cv::Point2i(1, 1) + cv::Point2i(n, m)) = cv::sum(imDiff2x)[0];
      }
    }

    double    min, max;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(absDiffsMatSubpix, &min, &max, &min_loc, &max_loc);

    totalOffset = totalOffset + min_loc - cv::Point2i(1, 1);
  }
  cv::Point2d output = cv::Point2d(totalOffset.x / (float)pixScale, totalOffset.y / (float)pixScale);

  return output;
}
