#ifndef FFTMETHOD_H
#define FFTMETHOD_H

//#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <image_transport/image_transport.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <mutex>
#include <thread>

#include <OpticFlowCalc.h>
#include <utilityFunctions.h>

enum FftType
{
  R2R = 0,  // real to CCS in case forward transform, CCS to real otherwise
  C2R = 1,  // complex to real in case inverse transform
  R2C = 2,  // real to complex in case forward transform
  C2C = 3   // complex to complex
};

struct OCL_FftPlan
{
private:
  cv::UMat        twiddles;
  int             dft_size;
  int             dft_depth;
  bool            status;
  cv::ocl::Kernel k_fft, k_fft_forw_row, k_fft_inv_row, k_fft_forw_col, k_fft_inv_col;
  cv::ocl::Kernel k_phase_corr;
  std::string     cl_file_name;

public:
  OCL_FftPlan(int _size, int _depth, std::string i_cl_file_name);
  bool enqueueTransform(cv::InputArray _src1, cv::InputArray _src2, cv::InputOutputArray _fft1, cv::InputOutputArray _fft2, cv::InputOutputArray _fftr1,
                        cv::InputOutputArray _fftr2, cv::InputOutputArray _mul, cv::InputOutputArray _ifftc, cv::InputOutputArray _pcr,
                        cv::InputOutputArray _dst, cv::InputArray _l_smem, cv::InputArray _l_maxval, cv::InputArray _l_maxloc, int rowsPerWI, int Xfields,
                        int Yfields, std::vector<cv::Point2f>& output, int thread_count, int block_count, cv::ocl::Queue& mainQueue);

private:
  static void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix);
  template <typename T>
  static void fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes);
};

struct OCL_FftPlanClassic
{
public:
  cv::String buildOptions_deprecated;

private:
  cv::UMat twiddles;
  int      thread_count;
  int      dft_size;
  int      dft_depth;
  bool     status;

  static int DFTFactorize(int n, int* factors) {
    int nf = 0, f, i, j;

    if (n <= 5) {
      factors[0] = n;
      return 1;
    }

    f = (((n - 1) ^ n) + 1) >> 1;
    if (f > 1) {
      factors[nf++] = f;
      n             = f == n ? 1 : n / f;
    }

    for (f = 3; n > 1;) {
      int d = n / f;
      if (d * f == n) {
        factors[nf++] = f;
        n             = d;
      } else {
        f += 2;
        if (f * f > n)
          break;
      }
    }

    if (n > 1)
      factors[nf++] = n;

    f = (factors[0] & 1) == 0;
    for (i = f; i < (nf + f) / 2; i++)
      CV_SWAP(factors[i], factors[nf - i - 1 + f], j);

    return nf;
  }

public:
  OCL_FftPlanClassic(int _size, int _depth) : dft_size(_size), dft_depth(_depth), status(true) {
    CV_Assert(dft_depth == CV_32F || dft_depth == CV_64F);

    int              min_radix;
    std::vector<int> radixes, blocks;
    ocl_getRadixes(dft_size, radixes, blocks, min_radix);
    thread_count = dft_size / min_radix;

    if (thread_count > (int)cv::ocl::Device::getDefault().maxWorkGroupSize()) {
      status = false;
      return;
    }

    // generate string with radix calls
    cv::String radix_processing;
    int        n = 1, twiddle_size = 0;
    for (size_t i = 0; i < radixes.size(); i++) {
      int radix = radixes[i], block = blocks[i];
      if (block > 1)
        radix_processing += cv::format("fft_radix%d_B%d(smem,twiddles+%d,ind,%d,%d);", radix, block, twiddle_size, n, dft_size / radix);
      else
        radix_processing += cv::format("fft_radix%d(smem,twiddles+%d,ind,%d,%d);", radix, twiddle_size, n, dft_size / radix);
      twiddle_size += (radix - 1) * n;
      n *= radix;
    }

    twiddles.create(1, twiddle_size, CV_MAKE_TYPE(dft_depth, 2));
    if (dft_depth == CV_32F)
      fillRadixTable<float>(twiddles, radixes);
    else
      fillRadixTable<double>(twiddles, radixes);

    buildOptions_deprecated =
        cv::format("-D LOCAL_SIZE=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s", dft_size, min_radix, cv::ocl::typeToStr(dft_depth),
                   cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)), dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str());
  }

  static cv::ocl::ProgramSource prep_ocl_kernel(const char* filename);


  bool enqueueTransformClassic(cv::InputArray _src, cv::OutputArray _dst, int num_dfts, int flags, int fftType, bool rows = true) const {
    if (!status)
      return false;

    cv::UMat src = _src.getUMat();
    cv::UMat dst = _dst.getUMat();

    size_t     globalsize[2];
    size_t     localsize[2];
    cv::String kernel_name;

    bool       is1d    = (flags & cv::DFT_ROWS) != 0 || num_dfts == 1;
    bool       inv     = (flags & cv::DFT_INVERSE) != 0;
    cv::String options = buildOptions_deprecated;

    if (rows) {
      globalsize[0] = thread_count;
      globalsize[1] = src.rows;
      localsize[0]  = thread_count;
      localsize[1]  = 1;
      kernel_name   = !inv ? "fft_multi_radix_rows" : "ifft_multi_radix_rows";
      if ((is1d || inv) && (flags & cv::DFT_SCALE))
        options += " -D DFT_SCALE";
    } else {
      globalsize[0] = num_dfts;
      globalsize[1] = thread_count;
      localsize[0]  = 1;
      localsize[1]  = thread_count;
      kernel_name   = !inv ? "fft_multi_radix_cols" : "ifft_multi_radix_cols";
      if (flags & cv::DFT_SCALE)
        options += " -D DFT_SCALE";
    }

    options += src.channels() == 1 ? " -D REAL_INPUT" : " -D COMPLEX_INPUT";
    options += dst.channels() == 1 ? " -D REAL_OUTPUT" : " -D COMPLEX_OUTPUT";
    options += is1d ? " -D IS_1D" : "";

    if (!inv) {
      if ((is1d && src.channels() == 1) || (rows && (fftType == R2R)))
        options += " -D NO_CONJUGATE";
    } else {
      if (rows && (fftType == C2R || fftType == R2R))
        options += " -D NO_CONJUGATE";
      if (dst.cols % 2 == 0)
        options += " -D EVEN";
    }
    cv::ocl::ProgramSource ps = prep_ocl_kernel("/home/viktor/OpenCV/opencv_3/modules/core/src/opencl/fft.cl");

    std::cout << "G0 " << globalsize[0] << " G1 " << globalsize[1] << " L0 " << localsize[0] << " L1 " << localsize[1] << std::endl;
    std::cout << options << std::endl;

    cv::ocl::Kernel k(kernel_name.c_str(), ps, options);
    if (k.empty())
      return false;

    k.args(cv::ocl::KernelArg::ReadOnly(src), cv::ocl::KernelArg::WriteOnly(dst), cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts);
    return k.run(2, globalsize, localsize, false);
  }

private:
  static void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix) {
    int factors[34];
    int nf = DFTFactorize(cols, factors);

    int n            = 1;
    int factor_index = 0;
    min_radix        = INT_MAX;

    // 2^n transforms
    if ((factors[factor_index] & 1) == 0) {
      for (; n < factors[factor_index];) {
        int radix = 2, block = 1;
        if (8 * n <= factors[0])
          radix = 8;
        else if (4 * n <= factors[0]) {
          radix = 4;
          if (cols % 12 == 0)
            block = 3;
          else if (cols % 8 == 0)
            block = 2;
        } else {
          if (cols % 10 == 0)
            block = 5;
          else if (cols % 8 == 0)
            block = 4;
          else if (cols % 6 == 0)
            block = 3;
          else if (cols % 4 == 0)
            block = 2;
        }

        radixes.push_back(radix);
        blocks.push_back(block);
        min_radix = std::min(min_radix, block * radix);
        n *= radix;
      }
      factor_index++;
    }

    // all the other transforms
    for (; factor_index < nf; factor_index++) {
      int radix = factors[factor_index], block = 1;
      if (radix == 3) {
        if (cols % 12 == 0)
          block = 4;
        else if (cols % 9 == 0)
          block = 3;
        else if (cols % 6 == 0)
          block = 2;
      } else if (radix == 5) {
        if (cols % 10 == 0)
          block = 2;
      }
      radixes.push_back(radix);
      blocks.push_back(block);
      min_radix = std::min(min_radix, block * radix);
    }
  }

  template <typename T>
  static void fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes) {
    cv::Mat tw        = twiddles.getMat(cv::ACCESS_WRITE);
    T*      ptr       = tw.ptr<T>();
    int     ptr_index = 0;

    int n = 1;
    for (size_t i = 0; i < radixes.size(); i++) {
      int radix = radixes[i];
      n *= radix;

      for (int j = 1; j < radix; j++) {
        double theta = -CV_2PI * j / n;

        for (int k = 0; k < (n / radix); k++) {
          ptr[ptr_index++] = (T)cos(k * theta);
          ptr[ptr_index++] = (T)sin(k * theta);
        }
      }
    }
  }
};

class OCL_FftPlanCache {
  /* private: */
  /*   static OCL_FftPlanCache* instance; */

public:
  /*   static OCL_FftPlanCache & getInstance() */
  /*   { */
  /*     if (instance == NULL) */
  /*       instance = new OCL_FftPlanCache(); */
  /*     return *instance; */
  /*   } */

  cv::Ptr<OCL_FftPlan> getFftPlan(int dft_size, int depth, std::string i_cl_file_name) {
    int                                           key = (dft_size << 16) | (depth & 0xFFFF);
    std::map<int, cv::Ptr<OCL_FftPlan>>::iterator f   = planStorage.find(key);
    if (f != planStorage.end()) {
      return f->second;
    } else {
      cv::Ptr<OCL_FftPlan> newPlan = cv::Ptr<OCL_FftPlan>(new OCL_FftPlan(dft_size, depth, i_cl_file_name));
      planStorage[key]             = newPlan;
      return newPlan;
    }
  }

  ~OCL_FftPlanCache() {
    planStorage.clear();
  }

  OCL_FftPlanCache() : planStorage() {
  }

protected:
  std::map<int, cv::Ptr<OCL_FftPlan>> planStorage;
};


class OCL_FftPlanCacheClassic {
  static std::mutex* initialization_mutex;

public:
  static OCL_FftPlanCacheClassic& getInstance() {

    static OCL_FftPlanCacheClassic* volatile instance = NULL;
    if (instance == NULL) {
      if (initialization_mutex == NULL)
        initialization_mutex = new std::mutex();
      std::scoped_lock lock(*initialization_mutex);
      if (instance == NULL)
        instance = new OCL_FftPlanCacheClassic();
    }
    return *instance;
  }

  cv::Ptr<OCL_FftPlanClassic> getFftPlanClassic(int dft_size, int depth) {
    int                                                  key = (dft_size << 16) | (depth & 0xFFFF);
    std::map<int, cv::Ptr<OCL_FftPlanClassic>>::iterator f   = planStorage.find(key);
    if (f != planStorage.end()) {
      return f->second;
    } else {
      cv::Ptr<OCL_FftPlanClassic> newPlan = cv::Ptr<OCL_FftPlanClassic>(new OCL_FftPlanClassic(dft_size, depth));
      planStorage[key]                    = newPlan;
      return newPlan;
    }
  }

  ~OCL_FftPlanCacheClassic() {
    planStorage.clear();
  }

  OCL_FftPlanCacheClassic() : planStorage() {
  }

protected:
  std::map<int, cv::Ptr<OCL_FftPlanClassic>> planStorage;
};

class FftMethod : public OpticFlowCalc {

private:
  std::mutex process_mutex;

  cv::Mat imCurrF, imPrevF;

  bool                    useOCL;
  bool                    useNewKernel;
  OCL_FftPlanCache        cache;
  OCL_FftPlanCacheClassic cacheClassic;
  std::string             cl_file_name;

  cv::UMat usrc1, usrc2;
  cv::UMat window1, window2;
  cv::UMat FFT1, FFT2, FFTR1, FFTR2, MUL, IFFTC, PCR, P, Pm, C, D, ML, T;
  cv::Mat  H_FFT1, H_FFT2, H_FFTR1, H_FFTR2, H_MUL, H_IFFTC, H_PCR, H_P, H_Pm, H_C, H_D, H_ML, H_T;
  cv::UMat L_SMEM, L_MAXVAL, L_MAXLOC;
  cv::UMat twiddles;

  cv::ocl::Queue mainQueue;

  int frameSize;
  int samplePointSize, samplePointSize_lr;

  int    imCenterX, imCenterY;  // center of original image
  int    xi, yi;                // frame corner coordinates
  double fx, fy;

  std::vector<cv::Point2d> speeds;

  int sqNum, sqNum_lr;

  int max_px_speed_lr, max_px_speed_sq_lr;

  cv::Point2d shift;
  cv::Point2d shift_raw;

  bool first;
  bool gotBoth;
  bool gotNth;
  int  Nreps;
  bool running;

  bool storeVideo;

  cv::VideoWriter outputVideo;

  void ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix);

  template <typename T>
  static void fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes);

  bool ocl_dft_rows(cv::InputArray _src, cv::OutputArray _dst, int nonzero_rows, int flags, int fftType);
  bool ocl_dft_cols(cv::InputArray _src, cv::OutputArray _dst, int nonzero_cols, int flags, int fftType);

  bool ocl_dft(cv::InputArray _src, cv::OutputArray _dst, int flags, int nonzero_rows);

  bool phaseCorrelate_ocl(cv::InputArray _src1, cv::InputArray _src2, std::vector<cv::Point2f>& out, int vec_rows, int vec_cols);

  bool phaseCorrelate_lr_ocl(cv::InputArray _src1, cv::InputArray _src2, std::vector<cv::Point2f>& out, int vec_rows, int vec_cols);

  std::vector<cv::Point2d> phaseCorrelateField(cv::Mat& _src1, cv::Mat& _src2, unsigned int X, unsigned int Y, CV_OUT double* response = 0);

  std::vector<cv::Point2d> phaseCorrelateFieldLongRange(cv::Mat& _src1, cv::Mat& _src2, unsigned int X, unsigned int Y, CV_OUT double* response = 0);

  void dft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags);
  void idft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags = 0);

  void mulSpectrums_special(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB);

  bool ocl_mulSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB);

public:
  FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
            bool i_tilt_corr_enable, std::string* videoPath, int videoFPS, std::string i_cl_file_name, bool i_useOCL);


  std::vector<cv::Point2d> processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center,
                                        std::vector<cv::Point2d>& raw_output, double i_fx, double i_fy);
  std::vector<cv::Point2d> processImageLongRange(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center,
                                                 std::vector<cv::Point2d>& raw_output, double i_fx, double i_fy);
};

#endif  // FFTMETHOD_H
