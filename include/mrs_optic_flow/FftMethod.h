#ifndef FFTMETHOD_H
#define FFTMETHOD_H

//#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <image_transport/image_transport.h>
#include "mrs_optic_flow/OpticFlowCalc.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <mutex>
#include "mrs_optic_flow/utilityFunctions.h"

cv::String buildOptions;

struct OCL_FftPlan
{
private:
  cv::UMat twiddles;
    int dft_size;
    int dft_depth;
    bool status;
    cv::ocl::Kernel k_fft,k_fft_forw_row, k_fft_inv_row, k_fft_forw_col, k_fft_inv_col;
    cv::ocl::Kernel k_phase_corr;
    std::string cl_file_name;

public:
    OCL_FftPlan(int _size, int _depth, std::string i_cl_file_name);
    /* bool enqueueTransform(cv::InputArray _src, cv::OutputArray _dst, int num_dfts, int flags, int fftType, bool rows = true); */
    bool enqueueTransform(cv::InputArray _src1, cv::InputArray _src2, cv::InputOutputArray _fft1, cv::InputArray _fft2, cv::InputOutputArray _fftr1, cv::InputArray _fftr2, cv::InputArray _mul, cv::InputArray _pcr, cv::OutputArray _dst, int rowsPerWI,int Xfields,int Yfields, std::vector<cv::Point> &output,int thread_count,int block_count);
private:
    static void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix);
    template <typename T>
    static void fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes);
};

class OCL_FftPlanCache
{
  /* private: */ 
  /*   static OCL_FftPlanCache* instance; */

  public:
  /*   static OCL_FftPlanCache & getInstance() */
  /*   { */
  /*     if (instance == NULL) */
  /*       instance = new OCL_FftPlanCache(); */ 
  /*     return *instance; */
  /*   } */

    cv::Ptr<OCL_FftPlan> getFftPlan(int dft_size, int depth, std::string i_cl_file_name)
    {
        int key = (dft_size << 16) | (depth & 0xFFFF);
        std::map<int, cv::Ptr<OCL_FftPlan> >::iterator f = planStorage.find(key);
        if (f != planStorage.end())
        {
            return f->second;
        }
        else
        {
          cv::Ptr<OCL_FftPlan> newPlan = cv::Ptr<OCL_FftPlan>(new OCL_FftPlan(dft_size, depth, i_cl_file_name));
            planStorage[key] = newPlan;
            return newPlan;
        }
    }

    ~OCL_FftPlanCache()
    {
        planStorage.clear();
    }

    OCL_FftPlanCache() :
        planStorage()
    { }
protected:
    std::map<int, cv::Ptr<OCL_FftPlan> > planStorage;
};
class FftMethod : public OpticFlowCalc {
private:

  std::mutex process_mutex;

  bool useOCL;
  bool useNewKernel;
  OCL_FftPlanCache cache;
  std::string cl_file_name;

  cv::UMat usrc1, usrc2;
  cv::UMat window1, window2;
  cv::UMat FFT1, FFT2, FFTR1, FFTR2, MUL, PCR, P, Pm, C, ML;
  cv::UMat twiddles;

  int frameSize;
  int samplePointSize;

  int imCenterX, imCenterY;  // center of original image
  int xi, yi;                // frame corner coordinates
  double fx,fy;

  std::vector<cv::Point2d> speeds;

  int sqNum;

  cv::Point2d shift;
  cv::Point2d shift_raw;

  bool first;
  bool raw_enable;
  bool rot_corr_enable;
  bool tilt_corr_enable;

  bool storeVideo;

  cv::VideoWriter outputVideo;

  void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix);

    template <typename T>
  static void fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes);

  /* bool ocl_dft_rows(cv::InputArray _src, cv::OutputArray _dst, int nonzero_rows, int flags, int fftType); */
  /* bool ocl_dft_cols(cv::InputArray _src, cv::OutputArray _dst, int nonzero_cols, int flags, int fftType); */

  /* bool ocl_dft(cv::InputArray _src, cv::OutputArray _dst, int flags, int nonzero_rows); */

  bool phaseCorrelate_ocl(cv::InputArray _src1,cv::InputArray _src2, std::vector<cv::Point2i> &out, int vec_rows, int vec_cols);

  std::vector<cv::Point2d> phaseCorrelateField(cv::Mat &_src1, cv::Mat &_src2, unsigned int X,unsigned int Y,
                                     CV_OUT double* response = 0);

  /* void dft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags); */
  /* void idft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags=0); */

  void mulSpectrums_special( cv::InputArray _srcA, cv::InputArray _srcB,
                       cv::OutputArray _dst, int flags, bool conjB );

  bool ocl_mulSpectrums( cv::InputArray _srcA, cv::InputArray _srcB,
                              cv::OutputArray _dst, int flags, bool conjB );
public:

  FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
            bool i_tilt_corr_enable, std::string *videoPath, int videoFPS, std::string cl_file_name);

  std::vector<cv::Point2d> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center, cv::Point2d tiltCorr_dynamic, std::vector<cv::Point2d> &raw_output, double i_fx, double i_fy);
};

#endif  // FFTMETHOD_H
