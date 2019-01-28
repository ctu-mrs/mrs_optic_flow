#include "../include/optic_flow/FftMethod.h"

static void magSpectrums( cv::InputArray _src, cv::OutputArray _dst)
{
  cv::Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    if(src.depth() == CV_32F)
        _dst.create( src.rows, src.cols, CV_32FC1 );
    else
        _dst.create( src.rows, src.cols, CV_64FC1 );

    cv::Mat dst = _dst.getMat();
    dst.setTo(0);//Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if( is_1d )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataSrc = src.ptr<float>();
        float* dataDst = dst.ptr<float>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = (float)std::sqrt((double)dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                          (double)dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
    else
    {
        const double* dataSrc = src.ptr<double>();
        double* dataDst = dst.ptr<double>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = std::sqrt(dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                   dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
}
static void divSpectrums( cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB)
{
  cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    cv::Mat dst = _dst.getMat();

    CV_Assert(dst.data != srcA.data); // non-inplace check
    CV_Assert(dst.data != srcB.data); // non-inplace check

    bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
             srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & cv::DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {

                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] -
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
        }
    }

}

static void fftShift(cv::InputOutputArray _out)
{
  cv::Mat out = _out.getMat();

    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    std::vector<cv::Mat> planes;
    split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if(is_1d)
    {
        int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
        xMid = xMid + yMid;

        for(size_t i = 0; i < planes.size(); i++)
        {
          cv::Mat tmp;
          cv::Mat half0(planes[i], cv::Rect(0, 0, xMid + is_odd, 1));
          cv::Mat half1(planes[i], cv::Rect(xMid + is_odd, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(planes[i](cv::Rect(0, 0, xMid, 1)));
            tmp.copyTo(planes[i](cv::Rect(xMid, 0, xMid + is_odd, 1)));
        }
    }
    else
    {
        int isXodd = out.cols % 2 == 1;
        int isYodd = out.rows % 2 == 1;
        for(size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
          cv::Mat q0(planes[i], cv::Rect(0,    0,    xMid + isXodd, yMid + isYodd));
          cv::Mat q1(planes[i], cv::Rect(xMid + isXodd, 0,    xMid, yMid + isYodd));
          cv::Mat q2(planes[i], cv::Rect(0,    yMid + isYodd, xMid + isXodd, yMid));
          cv::Mat q3(planes[i], cv::Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

            if(!(isXodd || isYodd))
            {
              cv::Mat tmp;
                q0.copyTo(tmp);
                q3.copyTo(q0);
                tmp.copyTo(q3);

                q1.copyTo(tmp);
                q2.copyTo(q1);
                tmp.copyTo(q2);
            }
            else
            {
              cv::Mat tmp0, tmp1, tmp2 ,tmp3;
                q0.copyTo(tmp0);
                q1.copyTo(tmp1);
                q2.copyTo(tmp2);
                q3.copyTo(tmp3);

                tmp0.copyTo(planes[i](cv::Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
                tmp3.copyTo(planes[i](cv::Rect(0, 0, xMid, yMid)));

                tmp1.copyTo(planes[i](cv::Rect(0, yMid, xMid, yMid + isYodd)));
                tmp2.copyTo(planes[i](cv::Rect(xMid, 0, xMid + isXodd, yMid)));
            }
        }
    }

    merge(planes, out);
}

static cv::Point2d weightedCentroid(cv::InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
  cv::Mat src = _src.getMat();

    int type = src.type();
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    int minr = peakLocation.y - (weightBoxSize.height >> 1);
    int maxr = peakLocation.y + (weightBoxSize.height >> 1);
    int minc = peakLocation.x - (weightBoxSize.width  >> 1);
    int maxc = peakLocation.x + (weightBoxSize.width  >> 1);

    cv::Point2d centroid;
    double sumIntensity = 0.0;

    // clamp the values to min and max if needed.
    if(minr < 0)
    {
        minr = 0;
    }

    if(minc < 0)
    {
        minc = 0;
    }

    if(maxr > src.rows - 1)
    {
        maxr = src.rows - 1;
    }

    if(maxc > src.cols - 1)
    {
        maxc = src.cols - 1;
    }

    if(type == CV_32FC1)
    {
        const float* dataIn = src.ptr<float>();
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += (double)dataIn[x];
            }

            dataIn += src.cols;
        }
    }
    else
    {
        const double* dataIn = src.ptr<double>();
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += dataIn[x];
            }

            dataIn += src.cols;
        }
    }

    if(response)
        *response = sumIntensity;

    sumIntensity += DBL_EPSILON; // prevent div0 problems...

    centroid.x /= sumIntensity;
    centroid.y /= sumIntensity;

    return centroid;
}

std::vector<cv::Point2d> FftMethod::phaseCorrelateField(cv::Mat &_src1, cv::Mat &_src2, unsigned int X, unsigned int Y, double* response)
{
  CV_Assert( _src1.type() == _src2.type());
  CV_Assert( _src1.type() == CV_32FC1 || _src1.type() == CV_64FC1 );
  CV_Assert( _src1.size() == _src2.size());

    clock_t                    begin, end,begin_overall;
    double                     elapsedTimeI,elapsedTime1,elapsedTime2,elapsedTime3,elapsedTime4,elapsedTime5,elapsedTime6,elapsedTimeO;
    elapsedTimeI=0;
    elapsedTime1=0;
    elapsedTime2=0;
    elapsedTime3=0;
    elapsedTime4=0;
    elapsedTime5=0;
    elapsedTime6=0;
    elapsedTimeO=0;

  begin = std::clock();
  begin_overall= std::clock();
  std::vector<cv::Point2d> output;


  /* _src1.copyTo(usrc1,cv::ACCESS_READ); */
  /* _src2.copyTo(usrc2,cv::ACCESS_READ); */

  usrc1 = _src1.getUMat(cv::ACCESS_READ);
  usrc2 = _src2.getUMat(cv::ACCESS_READ);



  /* int M = cv::getOptimalDFTSize(samplePointSize); */
  /* int N = cv::getOptimalDFTSize(samplePointSize); */
  /* ROS_INFO("M: %d",M); */

    cv::Rect roi;

    end         = std::clock();
    elapsedTimeI = double(end - begin) / CLOCKS_PER_SEC;
    ROS_INFO("INITIALIZATION: %f s, %f Hz", elapsedTimeI , 1.0 / elapsedTimeI);



    for (int i = 0; i < X; i++) {
      for (int j = 0; j < Y; j++) {
        begin = std::clock();


        xi    = i * samplePointSize;
        yi    = j * samplePointSize;
        roi = cv::Rect(xi,yi,samplePointSize,samplePointSize);



        window1 = usrc1(roi);
        window2 = usrc2(roi);
          /* ROS_INFO_ONCE("padded size: %dx%d",padded2.rows,padded2.cols); */

        end         = std::clock();
        elapsedTime1 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();

        // execute phase correlation equation
        // Reference: http://en.wikipedia.org/wiki/Phase_correlation
        /* dft(usrc1(roi), FFT1, cv::DFT_REAL_OUTPUT); */
        /* dft(usrc2(roi), FFT2, cv::DFT_REAL_OUTPUT); */
        dft(window1, FFT1, cv::DFT_REAL_OUTPUT);
        dft(window2, FFT2, cv::DFT_REAL_OUTPUT);

        /* ROS_INFO("[%d]: FFT TYPE", FFT1.type()); */

        end         = std::clock();
        elapsedTime2 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();

        mulSpectrums(FFT1, FFT2, P, 0, true);

        end         = std::clock();
        elapsedTime3 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();

        magSpectrums(P, Pm);

        end         = std::clock();
        elapsedTime4 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();
        
        divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

        end         = std::clock();
        elapsedTime5 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();
        

        idft(C, C); // gives us the nice peak shift location...

        end         = std::clock();
        elapsedTime6 += double(end - begin) / CLOCKS_PER_SEC;
        begin = std::clock();

        fftShift(C); // shift the energy to the center of the frame.


        // locate the highest peak
        cv::Point peakLoc;
        minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

        // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
        cv::Point2d t;
        t = weightedCentroid(C, peakLoc, cv::Size(5, 5), response);

        // max response is M*N (not exactly, might be slightly larger due to rounding errors)
        if(response)
          *response /= samplePointSize*samplePointSize;

        /* end         = std::clock(); */
        /* elapsedTime = double(end - begin) / CLOCKS_PER_SEC; */
        /* ROS_INFO("Step 4: %f s, %f Hz", elapsedTime , 1.0 / elapsedTime); */
        /* begin = std::clock(); */

        // adjust shift relative to image center...
        cv::Point2d center((double)window1.cols / 2.0, (double)window1.rows / 2.0);

        output.push_back(center - t);
        /* output.push_back(cv::Point(0,0)); */
      }
    }
    ROS_INFO("Step 1: %f s, %f Hz", elapsedTime1 , 1.0 / elapsedTime1);
    ROS_INFO("Step 2: %f s, %f Hz", elapsedTime2 , 1.0 / elapsedTime2);
    ROS_INFO("Step 3: %f s, %f Hz", elapsedTime3 , 1.0 / elapsedTime3);
    ROS_INFO("Step 4: %f s, %f Hz", elapsedTime4 , 1.0 / elapsedTime4);
    ROS_INFO("Step 5: %f s, %f Hz", elapsedTime5 , 1.0 / elapsedTime5);
    ROS_INFO("Step 6: %f s, %f Hz", elapsedTime6 , 1.0 / elapsedTime6);
    end         = std::clock();
    elapsedTimeO = double(end - begin_overall) / CLOCKS_PER_SEC;
    ROS_INFO("OVERALL: %f s, %f Hz", elapsedTimeO , 1.0 / elapsedTimeO);
    return output;
}
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

    usrc1.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    usrc2.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    window1.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    window2.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    FFT1.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    FFT2.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);

  first = true;
}

std::vector<cv::Point2d> FftMethod::processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center, cv::Point2d tiltCorr_dynamic, std::vector<cv::Point2d> &raw_output, double i_fx, double i_fy) {

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
  speeds = phaseCorrelateField(imCurrF, imPrevF,sqNum,sqNum);

  for (int i = 0; i < sqNum; i++) {
    for (int j = 0; j < sqNum; j++) {
      xi    = i * samplePointSize;
      yi    = j * samplePointSize;
      shift = cv::phaseCorrelate(imPrevF(cv::Rect(xi, yi, samplePointSize, samplePointSize)), imCurrF(cv::Rect(xi, yi, samplePointSize, samplePointSize)));
      shift_raw = shift;

      bool valid=true;
      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
          absd(shift.y) > ((double)samplePointSize / 2)) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        valid=false;
      }

      if (raw_enable) {
        // push without correction first
        if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
            absd(shift.y) > ((double)samplePointSize / 2)) {
          ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
          speeds[j*sqNum+i] = cv::Point2d(nan(""), nan(""));
        } else {
          // ROS_WARN("[OpticFlow]: Hacks going on in raw...");  // hack for Gazebo Mobius
          // speeds.push_back(cv::Point2f(-shift.x,-shift.y));
          speeds[j*sqNum+i] = cv::Point2d(shift.x, shift.y);
        }
      }


      /* if (tilt_corr_enable) { */
      /*   distX = fabs( (xi + samplePointSize / 2) - midX); */
      /*   distY = fabs( (yi + samplePointSize / 2) - midY); */

      /*   /1* double spDist = sqrt(pow(fx,2)+pow(xi,2)+pow((fx/fy)*yi,2)); *1/ */
      /*   cv::Point2d tiltCorrDynamicCurrSample; */
      /*   tiltCorrDynamicCurrSample.x = tan(atan(distX/fx)+tiltCorr_dynamic.x)*fx-distX; */
      /*   tiltCorrDynamicCurrSample.y = tan(atan(distY/fy)+tiltCorr_dynamic.y)*fy-distY; */
      /*   shift = shift + tiltCorrDynamicCurrSample; */
      /* } */

      /* if (rot_corr_enable) { */
      /*   // rotation correction */
      /*   distX = (xi + samplePointSize / 2) - rot_center.x; */
      /*   distY = (yi + samplePointSize / 2) - rot_center.y; */

      /*   corrX = (distX*cos(yaw_angle) -distY*sin(yaw_angle))-distX; */
      /*   corrY = (distX*sin(yaw_angle) +distY*cos(yaw_angle))-distY; */

      /*   shift.x = shift.x + corrX; */
      /*   shift.y = shift.y + corrY; */
      /* } */

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
                 cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(shift.x * 5.0), (int)(shift.y * 5.0)), cv::Scalar(255),valid?5:1);
        /* cv::line(imView, cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2), */
        /*          cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(shift_raw.x * 5.0), (int)(shift_raw.y * 5.0)), cv::Scalar(155),valid?3:1); */
        /* cv::line(imView, cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2), */
        /*          cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(corrX * 5.0), (int)(corrY * 5.0)), cv::Scalar(0),valid?1:1); */
      }
    }
  }

  /* cv::circle(imView, cv::Point2i(rot_center), 5, cv::Scalar(255),5); */
  cv::line(imView, cv::Point2i(imView.size()/2), cv::Point2i(imView.size()/2)+cv::Point2i(tan(tiltCorr_dynamic.x)*fx*5,tan(tiltCorr_dynamic.y)*fy*5), cv::Scalar(155),5);

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
