#include <FftMethod.h>

#define LONG_RANGE_RATIO 4

cv::String buildOptions;
cv::Mat    storageA, storageB, diffmap;

/* showFMat() //{ */

void showFMat(cv::InputOutputArray& M, const char* name = "ocv_debugshit") {

  cv::Mat mat_host;
  if (M.isUMat()) {
    cv::UMat UM = M.getUMat();
    mat_host    = UM.getMat(cv::ACCESS_READ);
  } else {
    mat_host = M.getMat();
  }

  cv::Mat catmat;
  if (mat_host.channels() > 1) {
    std::vector<cv::Mat> mulmats;
    cv::split(mat_host, mulmats);
    hconcat(mulmats, catmat);
  } else {
    catmat = mat_host;
  }

  double    minval;
  double    maxval;
  cv::Point minloc, maxloc;
  cv::minMaxLoc(catmat, &minval, &maxval, &minloc, &maxloc);

  /* std::cout << "IMMIN: " << minval << " IMMAX: " << maxval << std::endl; */
  /* std::cout << "MINLOC: " << minloc << " MAXLOC: " << maxloc << std::endl; */
  /* std::cout << "WIDTH: " << mat_host.cols << " HEIGHT: " << mat_host.rows << std::endl; */

  /* if (mat_host.type() == CV_32F) */
  /*   std::cout << "0:0: " << mat_host.at<float>(0,0) << " HEIGHT: " << mat_host.rows << std::endl; */

  /* maxval = std::min(maxval,7e6); */
  /* cv::convertScaleAbs(catmat, catmat, 255 / (maxval)); */
  cv::convertScaleAbs(catmat - minval, catmat, 255 / (maxval - minval));
  /* cv::minMaxIdx(usrc2, &min, &max); */
  /* std::cout << "IMMIN: " << min << " IMMAX: " << max << std::endl; */
  /* cv::convertScaleAbs(usrc2, catmat, 255 / max); */
  imshow(name, catmat);
}

//}

/* prep_ocl_kernel() //{ */

cv::ocl::ProgramSource prep_ocl_kernel(const char* filename) {

  /* std::cout << "Loading OpenCL kernel file \" " << filename << std::endl; */
  std::ifstream ist(filename, std::ifstream::in);
  std::string   str((std::istreambuf_iterator<char>(ist)), std::istreambuf_iterator<char>());

  if (str.empty())
    std::cerr << "Could not load the file. Aborting" << std::endl;

  return cv::ocl::ProgramSource(str.c_str());
}

//}

/* magSpectrums() //{ */

static void magSpectrums(cv::InputArray _src, cv::OutputArray _dst) {

  cv::Mat src   = _src.getMat();
  int     depth = src.depth(), cn = src.channels(), type = src.type();
  int     rows = src.rows, cols = src.cols;
  int     j, k;

  CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

  if (src.depth() == CV_32F)
    _dst.create(src.rows, src.cols, CV_32FC1);
  else
    _dst.create(src.rows, src.cols, CV_64FC1);

  cv::Mat dst = _dst.getMat();
  dst.setTo(0);  // Mat elements are not equal to zero by default!

  bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

  if (is_1d)
    cols = cols + rows - 1, rows = 1;

  int ncols = cols * cn;
  int j0    = cn == 1;
  int j1    = ncols - (cols % 2 == 0 && cn == 1);

  if (depth == CV_32F) {
    const float* dataSrc = src.ptr<float>();
    float*       dataDst = dst.ptr<float>();

    size_t stepSrc = src.step / sizeof(dataSrc[0]);
    size_t stepDst = dst.step / sizeof(dataDst[0]);

    if (!is_1d && cn == 1) {
      for (k = 0; k < (cols % 2 ? 1 : 2); k++) {
        if (k == 1)
          dataSrc += cols - 1, dataDst += cols - 1;
        dataDst[0] = dataSrc[0] * dataSrc[0];
        if (rows % 2 == 0)
          dataDst[(rows - 1) * stepDst] = dataSrc[(rows - 1) * stepSrc] * dataSrc[(rows - 1) * stepSrc];

        for (j = 1; j <= rows - 2; j += 2) {
          dataDst[j * stepDst] =
              (float)std::sqrt((double)dataSrc[j * stepSrc] * dataSrc[j * stepSrc] + (double)dataSrc[(j + 1) * stepSrc] * dataSrc[(j + 1) * stepSrc]);
        }

        if (k == 1)
          dataSrc -= cols - 1, dataDst -= cols - 1;
      }
    }

    for (; rows--; dataSrc += stepSrc, dataDst += stepDst) {
      if (is_1d && cn == 1) {
        dataDst[0] = dataSrc[0] * dataSrc[0];
        if (cols % 2 == 0)
          dataDst[j1] = dataSrc[j1] * dataSrc[j1];
      }

      for (j = j0; j < j1; j += 2) {
        dataDst[j] = (float)std::sqrt((double)dataSrc[j] * dataSrc[j] + (double)dataSrc[j + 1] * dataSrc[j + 1]);
      }
    }
  } else {
    const double* dataSrc = src.ptr<double>();
    double*       dataDst = dst.ptr<double>();

    size_t stepSrc = src.step / sizeof(dataSrc[0]);
    size_t stepDst = dst.step / sizeof(dataDst[0]);

    if (!is_1d && cn == 1) {
      for (k = 0; k < (cols % 2 ? 1 : 2); k++) {
        if (k == 1)
          dataSrc += cols - 1, dataDst += cols - 1;
        dataDst[0] = dataSrc[0] * dataSrc[0];
        if (rows % 2 == 0)
          dataDst[(rows - 1) * stepDst] = dataSrc[(rows - 1) * stepSrc] * dataSrc[(rows - 1) * stepSrc];

        for (j = 1; j <= rows - 2; j += 2) {
          dataDst[j * stepDst] = std::sqrt(dataSrc[j * stepSrc] * dataSrc[j * stepSrc] + dataSrc[(j + 1) * stepSrc] * dataSrc[(j + 1) * stepSrc]);
        }

        if (k == 1)
          dataSrc -= cols - 1, dataDst -= cols - 1;
      }
    }

    for (; rows--; dataSrc += stepSrc, dataDst += stepDst) {
      if (is_1d && cn == 1) {
        dataDst[0] = dataSrc[0] * dataSrc[0];
        if (cols % 2 == 0)
          dataDst[j1] = dataSrc[j1] * dataSrc[j1];
      }

      for (j = j0; j < j1; j += 2) {
        dataDst[j] = std::sqrt(dataSrc[j] * dataSrc[j] + dataSrc[j + 1] * dataSrc[j + 1]);
      }
    }
  }
}

//}

/* DFTFactorize() //{ */

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

//}

/* OCL_FftPlan::OCL_FftPlan() //{ */

OCL_FftPlan::OCL_FftPlan(int _size, int _depth, std::string i_cl_file_name) : dft_size(_size), dft_depth(_depth), status(true) {

  cl_file_name = i_cl_file_name;

  CV_Assert(dft_depth == CV_32F || dft_depth == CV_64F);

  int              min_radix;
  std::vector<int> radixes, blocks;
  ocl_getRadixes(dft_size, radixes, blocks, min_radix);
  double thread_count_deprecated = dft_size / min_radix;

  if (thread_count_deprecated > (int)cv::ocl::Device::getDefault().maxWorkGroupSize()) {
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

  twiddles.create(1, twiddle_size, CV_MAKE_TYPE(dft_depth, 2), cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  if (dft_depth == CV_32F)
    fillRadixTable<float>(twiddles, radixes);
  else
    fillRadixTable<double>(twiddles, radixes);


  std::string buildOptions_deprecated =
      cv::format("-D LOCAL_SIZE=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s", dft_size, min_radix, cv::ocl::typeToStr(dft_depth),
                 cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)), dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str());
}

//}

/* OCL_FftPlanClassic::prep_ocl_kernel() //{ */

cv::ocl::ProgramSource OCL_FftPlanClassic::prep_ocl_kernel(const char* filename) {

  /* std::cout << "Loading OpenCL kernel file \" " << filename << std::endl; */
  std::ifstream ist(filename, std::ifstream::in);
  std::string   str((std::istreambuf_iterator<char>(ist)), std::istreambuf_iterator<char>());

  if (str.empty())
    std::cerr << "Could not load the file. Aborting" << std::endl;

  return cv::ocl::ProgramSource(str.c_str());
}

//}

/* OCL_FftPlan::enqueueTransform() //{ */

bool OCL_FftPlan::enqueueTransform(cv::InputArray _src1, cv::InputArray _src2, cv::InputOutputArray _fft1, cv::InputOutputArray _fft2,
                                   cv::InputOutputArray _fftr1, cv::InputOutputArray _fftr2, cv::InputOutputArray _mul, cv::InputOutputArray _ifftc,
                                   cv::InputOutputArray _pcr, cv::InputOutputArray _dst, cv::InputArray _l_smem, cv::InputArray _l_maxval,
                                   cv::InputArray _l_maxloc, int rowsPerWI, int Xfields, int Yfields, std::vector<cv::Point2f>& output, int thread_count,
                                   int block_count, cv::ocl::Queue& mainQueue) {
  if (!status)
    return false;

  /* const cv::ocl::Device& dev = cv::ocl::Device::getDefault(); */

  cv::UMat src1  = _src1.getUMat();
  cv::UMat src2  = _src2.getUMat();
  cv::UMat fftr1 = _fftr1.getUMat();
  cv::UMat fftr2 = _fftr2.getUMat();
  cv::UMat fft1  = _fft1.getUMat();
  cv::UMat fft2  = _fft2.getUMat();
  cv::UMat mul   = _mul.getUMat();
  cv::UMat ifftc = _ifftc.getUMat();
  cv::UMat pcr   = _pcr.getUMat();
  cv::UMat dst   = _dst.getUMat();

  cv::UMat l_smem   = _l_smem.getUMat();
  cv::UMat l_maxval = _l_maxval.getUMat();
  cv::UMat l_maxloc = _l_maxloc.getUMat();

  size_t     globalsize[2];
  size_t     localsize[2];
  cv::String kernel_name;

  cv::String options = buildOptions;

  kernel_name = "phaseCorrelateField";

  globalsize[0] = thread_count;
  globalsize[1] = block_count;
  localsize[0]  = thread_count;
  localsize[1]  = 1;

  /* std::cout << "G0 " << globalsize[0] << " G1 " << globalsize[1] << " L0 " << localsize[0] << " L1 " << localsize[1] << std::endl; */

  options += " -D ROW_F_REAL_INPUT";
  options += " -D COL_F_COMPLEX_INPUT";
  options += " -D ROW_F_COMPLEX_OUTPUT";
  options += " -D COL_F_REAL_OUTPUT";
  options += " -D COL_I_REAL_INPUT";
  options += " -D COL_I_COMPLEX_OUTPUT";
  options += " -D ROW_I_COMPLEX_INPUT";
  options += " -D ROW_I_REAL_OUTPUT";
  /* options += " -D ROW_I_COMPLEX_OUTPUT"; */
  options += " -D NO_CONJUGATE";
  options += " -D MUL_CONJ";
  options += " -D NEED_MAXVAL";
  options += " -D NEED_MAXLOC";

  if ((block_count % 2) == 0)
    options += " -D EVEN";

  size_t wgs = thread_count;
  options += " -D WGS=" + std::to_string((int)wgs);

  int wgs2_aligned = 1;
  while (wgs2_aligned < (int)wgs)
    wgs2_aligned <<= 1;
  wgs2_aligned >>= 1;
  options += " -D WGS2_ALIGNED=";
  options += std::to_string(wgs2_aligned);

  /* std::cout << options << std::endl; */
  if (k_phase_corr.empty())
    k_phase_corr = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options);

  if (k_phase_corr.empty()) {
    return false;
  }


  int ki = 0;
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadOnly(src1));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadOnly(src2));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(fftr1));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(fftr2));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(fft1));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(fft2));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(mul));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(ifftc));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadWrite(pcr));
  ki     = k_phase_corr.set(ki, cv::ocl::KernelArg::PtrReadWrite(dst));
  /* ki = k_phase_corr.set(ki, cv::ocl::KernelArg::PtrWriteOnly(l_smem)); */
  /* ki = k_phase_corr.set(ki, cv::ocl::KernelArg::PtrWriteOnly(l_maxloc)); */
  /* ki = k_phase_corr.set(ki, cv::ocl::KernelArg::PtrWriteOnly(l_maxval)); */
  ki = k_phase_corr.set(ki, cv::ocl::KernelArg::ReadOnlyNoSize(twiddles));
  ki = k_phase_corr.set(ki, thread_count);
  ki = k_phase_corr.set(ki, rowsPerWI);
  ki = k_phase_corr.set(ki, Xfields);
  ki = k_phase_corr.set(ki, Yfields);
  ki = k_phase_corr.set(ki, block_count);

  /* k_phase_corr.args( */
  /*     cv::ocl::KernelArg::ReadOnly(src1), */
  /*     cv::ocl::KernelArg::ReadOnly(src2), */
  /*     cv::ocl::KernelArg::ReadWrite(fftr1), */
  /*     cv::ocl::KernelArg::ReadWrite(fftr2), */
  /*     cv::ocl::KernelArg::ReadWrite(fft1), */
  /*     cv::ocl::KernelArg::ReadWrite(fft2), */
  /*     cv::ocl::KernelArg::ReadWrite(mul), */
  /*     cv::ocl::KernelArg::ReadWrite(ifftc), */
  /*     cv::ocl::KernelArg::ReadWrite(pcr), */
  /*     cv::ocl::KernelArg::PtrReadWrite(dst), */
  /*     cv::ocl::KernelArg::PtrReadWrite(l_smem), */
  /*     cv::ocl::KernelArg::PtrReadWrite(l_maxval), */
  /*     cv::ocl::KernelArg::PtrReadWrite(l_maxloc), */
  /*     cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), */
  /*     thread_count, */
  /*     rowsPerWI, */
  /*     Xfields, */
  /*     Yfields */
  /*     ); */

  /* std::cout << "LMS USED: " << k_phase_corr.localMemSize() << std::endl; */
  /* std::cout << "LMS AVAIL: " << cv::ocl::Device::getDefault().localMemSize() << std::endl; */
  /* std::cout << "BS: " << cv::ocl::Device::getDefault().printfBufferSize() << std::endl; */

  bool partial = k_phase_corr.run(2, globalsize, localsize, true, mainQueue);

  /* ros::Duration(0.0000001).sleep(); */
  /* usleep(100000); */
  /* std::this_thread::sleep_for(std::chrono::milliseconds(1)); */
  /* return false; */

  /* showFMat(fft1); */

  cv::Mat dst_host = dst.getMat(cv::ACCESS_READ);

  /* float maxVal = -1; */
  int   maxLoc[2];
  float maxLocF[2];

  size_t index = 0;
  for (int j = 0; j < Yfields; j++) {
    for (int i = 0; i < Xfields; i++) {

      uint  index_max = std::numeric_limits<uint>::max();
      float maxval = std::numeric_limits<float>::min() > 0 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::min(), maxval2 = maxval;
      uint  maxloc = index_max;

      const float *maxptr = NULL, *maxptr2 = NULL;
      const uint*  maxlocptr = NULL;
      maxptr                 = (const float*)(dst_host.ptr() + index);
      index += sizeof(float) * block_count;
      index     = cv::alignSize(index, 8);
      maxlocptr = (const uint*)(dst_host.ptr() + index);
      index += sizeof(uint) * block_count;
      index = cv::alignSize(index, 8);

      /* for (int k = 0; k < pcr.rows; k++) */
      int k = 0;
      {
        /* std::cout << "maxptr[" << k << "] = " << maxptr[k] <<std::endl; */
        /* std::cout << "maxlocptr[" << k << "] = " << maxlocptr[k] <<std::endl; */
        if (maxptr && maxptr[k] >= maxval) {
          maxLocF[0] = maxptr[1];
          maxLocF[1] = maxptr[2];
          if (maxptr[k] == maxval) {
            if (maxlocptr)
              maxloc = std::min(maxlocptr[k], maxloc);
          } else {
            if (maxlocptr)
              maxloc = maxlocptr[k];
            maxval = maxptr[k];
          }
        }
        if (maxptr2 && maxptr2[k] > maxval2)
          maxval2 = maxptr2[k];
      }

      /* maxVal = (double)maxval; */
      /* std::cout << "MAXVAL: " << maxval <<std::endl; */
      /* std::cout << "MAXLOC: " << maxloc <<std::endl; */

      maxLoc[0] = (maxloc % block_count) - block_count / 2;
      maxLoc[1] = (maxloc / block_count) - block_count / 2;

      /* std::cout << "MAXLOC: " << maxLoc[0] << " : " << maxLoc[1] <<std::endl; */
      /* std::cout << "MAXLOC float: " << maxLocF[0] << " : " << maxLocF[1] <<std::endl; */

      if ((abs(maxLoc[0]) > block_count / 2) || (abs(maxLoc[1]) > block_count / 2)) {
        /* pcr.copyTo(storageB); */
        ROS_WARN("[OpticFlow]: LARGE SHIFT DETECTED!: %d:%d - %d:%d, block_count:%d", i, j, maxLoc[0], maxLoc[1], block_count);
        output[i + j * Xfields] = cv::Point2f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
      } else
        output[i + j * Xfields] = cv::Point2f(maxLocF[0], maxLocF[1]);


      /* output[i+j*Xfields]  = cv::Point2f(maxLoc[0],maxLoc[1]); */
      /* std::cout << "OUT: " << output[i+j*Xfields] <<std::endl; */
    }
  }
  /* std::cout << "OUT: " << output <<std::endl; */
  return partial;
}

//}

/* OCL_FftPlan::ocl_getRadixes() //{ */

void OCL_FftPlan::ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix) {

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
      min_radix = cv::min(min_radix, block * radix);
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
    min_radix = cv::min(min_radix, block * radix);
  }
}

//}

/* OCL_FftPlan::fillRadixTable() //{ */

template <typename T>
void OCL_FftPlan::fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes) {
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

//}

/* FftMethod::ocl_getRadixes() //{ */

void FftMethod::ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix) {

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
      min_radix = cv::min(min_radix, block * radix);
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
    min_radix = cv::min(min_radix, block * radix);
  }
}

//}

/* FftMethod::fillRadixTable() //{ */

template <typename T>
void FftMethod::fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes) {
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

//}

/* FftMethod::ocl_dft_rows() //{ */

bool FftMethod::ocl_dft_rows(cv::InputArray _src, cv::OutputArray _dst, int nonzero_rows, int flags, int fftType) {

  int                         type = _src.type(), depth = CV_MAT_DEPTH(type);
  cv::Ptr<OCL_FftPlanClassic> plan = cacheClassic.getFftPlanClassic(_src.cols(), depth);
  /* std::cout << "HEEEY " << plan->buildOptions_deprecated << std::endl; */
  return plan->enqueueTransformClassic(_src, _dst, nonzero_rows, flags, fftType, true);
}

//}

/* FftMethod::ocl_dft_cols() //{ */

bool FftMethod::ocl_dft_cols(cv::InputArray _src, cv::OutputArray _dst, int nonzero_cols, int flags, int fftType) {
  int                         type = _src.type(), depth = CV_MAT_DEPTH(type);
  cv::Ptr<OCL_FftPlanClassic> plan = cacheClassic.getFftPlanClassic(_src.rows(), depth);
  return plan->enqueueTransformClassic(_src, _dst, nonzero_cols, flags, fftType, false);
}

//}

/* FftMethod::ocl_dft() //{ */

bool FftMethod::ocl_dft(cv::InputArray _src, cv::OutputArray _dst, int flags, int nonzero_rows) {

  int      type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
  cv::Size ssize         = _src.size();
  bool     doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0;

  if (!((cn == 1 || cn == 2) && (depth == CV_32F || (depth == CV_64F && doubleSupport))))
    return false;

  // if is not a multiplication of prime numbers { 2, 3, 5 }
  if (ssize.area() != cv::getOptimalDFTSize(ssize.area()))
    return false;

  cv::UMat src            = _src.getUMat();
  int      complex_input  = cn == 2 ? 1 : 0;
  int      complex_output = (flags & cv::DFT_COMPLEX_OUTPUT) != 0;
  int      real_input     = cn == 1 ? 1 : 0;
  int      real_output    = (flags & cv::DFT_REAL_OUTPUT) != 0;
  bool     inv            = (flags & cv::DFT_INVERSE) != 0 ? 1 : 0;

  if (nonzero_rows <= 0 || nonzero_rows > _src.rows())
    nonzero_rows = _src.rows();

  // if output format is not specified
  if (complex_output + real_output == 0) {
    if (real_input)
      real_output = 1;
    else
      complex_output = 1;
  }

  FftType fftType = (FftType)(complex_input << 0 | complex_output << 1);

  // Forward Complex to CCS not supported
  if (fftType == C2R && !inv)
    fftType = C2C;

  // Inverse CCS to Complex not supported
  if (fftType == R2C && inv)
    fftType = R2R;

  cv::UMat output;
  if (fftType == C2C || fftType == R2C) {
    // complex output
    _dst.create(src.size(), CV_MAKETYPE(depth, 2));
    output = _dst.getUMat();
  } else {
    // real output
    _dst.create(src.size(), CV_MAKETYPE(depth, 1));
    output.create(src.size(), CV_MAKETYPE(depth, 2));
  }

  if (!inv) {

    if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType))
      return false;

    /* output(cv::Rect(0,0,samplePointSize/2,samplePointSize)).copyTo(storageB); */

    int nonzero_cols = fftType == R2R ? output.cols / 2 + 1 : output.cols;
    if (!ocl_dft_cols(output, _dst, nonzero_cols, flags, fftType))
      return false;

  } else {
    if (fftType == C2C) {
      // complex output
      if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType))
        return false;

      if (!ocl_dft_cols(output, output, output.cols, flags, fftType))
        return false;
    } else {
      int nonzero_cols = src.cols / 4 + 1;
      if (!ocl_dft_cols(src, output, nonzero_cols, flags, fftType))
        return false;

      /* output(cv::Rect(0,0,samplePointSize/2,samplePointSize)).copyTo(storageB); */

      if (!ocl_dft_rows(output, _dst, nonzero_rows, flags, fftType))
        return false;
    }
  }
  return true;
}

//}

/* FftMethod::phaseCorrelate_ocl() //{ */

bool FftMethod::phaseCorrelate_ocl(cv::InputArray _src1, cv::InputArray _src2, std::vector<cv::Point2f>& out, int vec_rows, int vec_cols) {

  int flags = 0;
  flags |= cv::DFT_REAL_OUTPUT;

  /* int nonzero_rows = samplePointSize; */

  int dft_size  = samplePointSize;
  int type      = _src1.type();
  int dft_depth = CV_MAT_DEPTH(type);

  CV_Assert(_src1.type() == _src2.type());
  CV_Assert(_src1.size() == _src2.size());
  CV_Assert(dft_depth == CV_32F || dft_depth == CV_64F);

  int              min_radix;
  std::vector<int> radixes, blocks;
  ocl_getRadixes(dft_size, radixes, blocks, min_radix);
  int thread_count = dft_size / min_radix;

  /* bool status; */

  if (thread_count > (int)cv::ocl::Device::getDefault().maxWorkGroupSize()) {
    /* status = false; */
    return false;
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

  char cvt[2][40];
  buildOptions = cv::format("-D LOCAL_SIZE=%d -D SEARCH_RADIUS=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s -D dstT=%s -D convertToDT=%s", dft_size,
                            55, min_radix, cv::ocl::typeToStr(dft_depth), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)),
                            dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str(), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, min_radix)),
                            cv::ocl::convertTypeStr(dft_depth, dft_depth, min_radix, cvt[0]));
  int      cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
  cv::Size ssize         = _src1.size();
  bool     doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0;

  if (!((cn == 1 || cn == 2) && (depth == CV_32F || (depth == CV_64F && doubleSupport))))
    return false;

  // if is not a multiplication of prime numbers { 2, 3, 5 }
  if (ssize.area() != cv::getOptimalDFTSize(ssize.area()))
    return false;

  /* cv::UMat src = _src1.getUMat(); */
  int  complex_input  = cn == 2 ? 1 : 0;
  int  complex_output = (flags & cv::DFT_COMPLEX_OUTPUT) != 0;
  int  real_input     = cn == 1 ? 1 : 0;
  int  real_output    = (flags & cv::DFT_REAL_OUTPUT) != 0;
  bool inv            = (flags & cv::DFT_INVERSE) != 0 ? 1 : 0;


  // if output format is not specified
  if (complex_output + real_output == 0) {
    if (real_input)
      real_output = 1;
    else
      complex_output = 1;
  }

  FftType fftType = (FftType)(complex_input << 0 | complex_output << 1);

  // Forward Complex to CCS not supported
  if (fftType == C2R && !inv)
    fftType = C2C;

  // Inverse CCS to Complex not supported
  if (fftType == R2C && inv)
    fftType = R2R;

  /* cv::UMat output; */
  /* if (fftType == C2C || fftType == R2C) */
  /* { */
  /*   // complex output */
  /*   _dst.create(src.size(), CV_MAKETYPE(depth, 2)); */
  /*   output = _dst.getUMat(); */
  /* } */
  /* else */
  /* { */
  /*   // real output */
  /*   _dst.create(src.size(), CV_MAKETYPE(depth, 1)); */
  /*   output.create(src.size(), CV_MAKETYPE(depth, 2)); */
  /* } */

  /* if (!inv) */
  /* { */

  /*   if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType)) */
  /*     return false; */

  /*   int nonzero_cols = fftType == R2R ? output.cols/2 + 1 : output.cols; */
  /*   if (!ocl_dft_cols(output, _dst, nonzero_cols, flags, fftType)) */
  /*     return false; */
  /* } */
  /* else */
  /* { */
  /*   if (fftType == C2C) */
  /*   { */
  /*     // complex output */
  /*     if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType)) */
  /*       return false; */


  /*     if (!ocl_dft_cols(output, output, output.cols, flags, fftType)) */
  /*       return false; */

  /*   } */
  /*   else */
  /*   { */
  /*     int nonzero_cols = src.cols/2 + 1; */
  /*     if (!ocl_dft_cols(src, output, nonzero_cols, flags, fftType)) */
  /*       return false; */


  /*     if (!ocl_dft_rows(output, _dst, nonzero_rows, flags, fftType)) */
  /*       return false; */
  /*   } */
  /* } */
  int rowsPerWI = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;

  cv::Ptr<OCL_FftPlan> plan = cache.getFftPlan(samplePointSize, depth, cl_file_name);
  return plan->enqueueTransform(_src1, _src2, FFT1, FFT2, FFTR1, FFTR2, MUL, IFFTC, PCR, ML, L_SMEM, L_MAXVAL, L_MAXLOC, rowsPerWI, vec_cols, vec_rows, out,
                                thread_count, samplePointSize, mainQueue);

  return false;
}

//}
//
/* FftMethod::phaseCorrelate_lr_ocl() //{ */

bool FftMethod::phaseCorrelate_lr_ocl(cv::InputArray _src1, cv::InputArray _src2, std::vector<cv::Point2f>& out, int vec_rows, int vec_cols) {

  int flags = 0;
  flags |= cv::DFT_REAL_OUTPUT;

  /* int nonzero_rows = samplePointSize; */

  int dft_size  = samplePointSize_lr;
  int type      = _src1.type();
  int dft_depth = CV_MAT_DEPTH(type);

  CV_Assert(_src1.type() == _src2.type());
  CV_Assert(_src1.size() == _src2.size());
  CV_Assert(dft_depth == CV_32F || dft_depth == CV_64F);

  int              min_radix;
  std::vector<int> radixes, blocks;
  ocl_getRadixes(dft_size, radixes, blocks, min_radix);
  int thread_count = dft_size / min_radix;

  /* bool status; */

  if (thread_count > (int)cv::ocl::Device::getDefault().maxWorkGroupSize()) {
    /* status = false; */
    return false;
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

  char cvt[2][40];
  buildOptions = cv::format("-D LOCAL_SIZE=%d -D SEARCH_RADIUS=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s -D dstT=%s -D convertToDT=%s", dft_size,
                            55, min_radix, cv::ocl::typeToStr(dft_depth), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)),
                            dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str(), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, min_radix)),
                            cv::ocl::convertTypeStr(dft_depth, dft_depth, min_radix, cvt[0]));
  int      cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
  cv::Size ssize         = _src1.size();
  bool     doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0;

  if (!((cn == 1 || cn == 2) && (depth == CV_32F || (depth == CV_64F && doubleSupport))))
    return false;

  // if is not a multiplication of prime numbers { 2, 3, 5 }
  if (ssize.area() != cv::getOptimalDFTSize(ssize.area()))
    return false;

  /* cv::UMat src = _src1.getUMat(); */
  int  complex_input  = cn == 2 ? 1 : 0;
  int  complex_output = (flags & cv::DFT_COMPLEX_OUTPUT) != 0;
  int  real_input     = cn == 1 ? 1 : 0;
  int  real_output    = (flags & cv::DFT_REAL_OUTPUT) != 0;
  bool inv            = (flags & cv::DFT_INVERSE) != 0 ? 1 : 0;


  // if output format is not specified
  if (complex_output + real_output == 0) {
    if (real_input)
      real_output = 1;
    else
      complex_output = 1;
  }

  FftType fftType = (FftType)(complex_input << 0 | complex_output << 1);

  // Forward Complex to CCS not supported
  if (fftType == C2R && !inv)
    fftType = C2C;

  // Inverse CCS to Complex not supported
  if (fftType == R2C && inv)
    fftType = R2R;

  int rowsPerWI = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;

  cv::Ptr<OCL_FftPlan> plan = cache.getFftPlan(samplePointSize_lr, depth, cl_file_name);
  return plan->enqueueTransform(_src1, _src2, FFT1, FFT2, FFTR1, FFTR2, MUL, IFFTC, PCR, ML, L_SMEM, L_MAXVAL, L_MAXLOC, rowsPerWI, vec_cols, vec_rows, out,
                                thread_count, samplePointSize_lr, mainQueue);

  return false;
}

//}

/* FftMethod::dft_special() //{ */

void FftMethod::dft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags) {

  ocl_dft(_src0, _dst, flags, 0);
}

//}

/* FftMethod::idft_special() //{ */

void FftMethod::idft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags) {

  ocl_dft(_src0, _dst, flags | cv::DFT_INVERSE, 0);
}

//}

/* FftMethod::ocl_mulSpectrums() //{ */

bool FftMethod::ocl_mulSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB) {

  int atype = _srcA.type();
  /* int btype = _srcB.type(); */
  int      rowsPerWI = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;
  cv::Size asize = _srcA.size(), bsize = _srcB.size();
  CV_Assert(asize == bsize);
  /* CV_Assert(atype == CV_32FC2 && btype == CV_32FC2); */

  if (flags != 0)
    return false;

  cv::UMat A = _srcA.getUMat(), B = _srcB.getUMat();
  CV_Assert(A.size() == B.size());

  _dst.create(A.size(), atype);
  cv::UMat dst = _dst.getUMat();

  cv::ocl::Kernel k("mulAndNormalizeSpectrums", prep_ocl_kernel(cl_file_name.c_str()), buildOptions + " -D CONJ ");
  if (k.empty())
    return false;

  k.args(cv::ocl::KernelArg::ReadOnlyNoSize(A), cv::ocl::KernelArg::ReadOnlyNoSize(B), cv::ocl::KernelArg::WriteOnly(dst), rowsPerWI);

  size_t globalsize[2] = {(size_t)asize.width, ((size_t)asize.height + rowsPerWI - 1) / rowsPerWI};
  return k.run(2, globalsize, NULL, false);
}

//}

/* FftMethod::mulSpectrums_special() //{ */

void FftMethod::mulSpectrums_special(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB) {

  int type = _srcA.type();

  CV_Assert(type == _srcB.type() && _srcA.size() == _srcB.size());
  CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

  ocl_mulSpectrums(_srcA, _srcB, _dst, flags, conjB);
}

//}

/* void divSpectrums() //{ */

static void divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB) {

  cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
  int     depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
  int     rows = srcA.rows, cols = srcA.cols;
  int     j, k;

  CV_Assert(type == srcB.type());
  CV_Assert(srcA.size() == srcB.size());
  CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
  CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

  _dst.create(srcA.rows, srcA.cols, type);
  cv::Mat dst = _dst.getMat();

  CV_Assert(dst.data != srcA.data);  // non-inplace check
  CV_Assert(dst.data != srcB.data);  // non-inplace check

  bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 && srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

  if (is_1d && !(flags & cv::DFT_ROWS))
    cols = cols + rows - 1, rows = 1;

  int ncols = cols * cn;
  int j0    = cn == 1;
  int j1    = ncols - (cols % 2 == 0 && cn == 1);

  if (depth == CV_32F) {
    const float* dataA = srcA.ptr<float>();
    const float* dataB = srcB.ptr<float>();
    float*       dataC = dst.ptr<float>();
    float        eps   = FLT_EPSILON;  // prevent div0 problems

    size_t stepA = srcA.step / sizeof(dataA[0]);
    size_t stepB = srcB.step / sizeof(dataB[0]);
    size_t stepC = dst.step / sizeof(dataC[0]);

    if (!is_1d && cn == 1) {
      for (k = 0; k < (cols % 2 ? 1 : 2); k++) {
        if (k == 1)
          dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
        dataC[0] = dataA[0] / (dataB[0] + eps);
        if (rows % 2 == 0)
          dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
        if (!conjB)
          for (j = 1; j <= rows - 2; j += 2) {
            double denom = (double)dataB[j * stepB] * dataB[j * stepB] + (double)dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + (double)eps;

            double re = (double)dataA[j * stepA] * dataB[j * stepB] + (double)dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

            double im = (double)dataA[(j + 1) * stepA] * dataB[j * stepB] - (double)dataA[j * stepA] * dataB[(j + 1) * stepB];

            dataC[j * stepC]       = (float)(re / denom);
            dataC[(j + 1) * stepC] = (float)(im / denom);
          }
        else
          for (j = 1; j <= rows - 2; j += 2) {

            double denom = (double)dataB[j * stepB] * dataB[j * stepB] + (double)dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + (double)eps;

            double re = (double)dataA[j * stepA] * dataB[j * stepB] - (double)dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

            double im = (double)dataA[(j + 1) * stepA] * dataB[j * stepB] + (double)dataA[j * stepA] * dataB[(j + 1) * stepB];

            dataC[j * stepC]       = (float)(re / denom);
            dataC[(j + 1) * stepC] = (float)(im / denom);
          }
        if (k == 1)
          dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
      }
    }

    for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC) {
      if (is_1d && cn == 1) {
        dataC[0] = dataA[0] / (dataB[0] + eps);
        if (cols % 2 == 0)
          dataC[j1] = dataA[j1] / (dataB[j1] + eps);
      }

      if (!conjB)
        for (j = j0; j < j1; j += 2) {
          double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
          double re    = (double)(dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1]);
          double im    = (double)(dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1]);
          dataC[j]     = (float)(re / denom);
          dataC[j + 1] = (float)(im / denom);
        }
      else
        for (j = j0; j < j1; j += 2) {
          double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
          double re    = (double)(dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1]);
          double im    = (double)(dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1]);
          dataC[j]     = (float)(re / denom);
          dataC[j + 1] = (float)(im / denom);
        }
    }
  } else {
    const double* dataA = srcA.ptr<double>();
    const double* dataB = srcB.ptr<double>();
    double*       dataC = dst.ptr<double>();
    double        eps   = DBL_EPSILON;  // prevent div0 problems

    size_t stepA = srcA.step / sizeof(dataA[0]);
    size_t stepB = srcB.step / sizeof(dataB[0]);
    size_t stepC = dst.step / sizeof(dataC[0]);

    if (!is_1d && cn == 1) {
      for (k = 0; k < (cols % 2 ? 1 : 2); k++) {
        if (k == 1)
          dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
        dataC[0] = dataA[0] / (dataB[0] + eps);
        if (rows % 2 == 0)
          dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
        if (!conjB)
          for (j = 1; j <= rows - 2; j += 2) {
            double denom = dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + eps;

            double re = dataA[j * stepA] * dataB[j * stepB] + dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

            double im = dataA[(j + 1) * stepA] * dataB[j * stepB] - dataA[j * stepA] * dataB[(j + 1) * stepB];

            dataC[j * stepC]       = re / denom;
            dataC[(j + 1) * stepC] = im / denom;
          }
        else
          for (j = 1; j <= rows - 2; j += 2) {
            double denom = dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + eps;

            double re = dataA[j * stepA] * dataB[j * stepB] - dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

            double im = dataA[(j + 1) * stepA] * dataB[j * stepB] + dataA[j * stepA] * dataB[(j + 1) * stepB];

            dataC[j * stepC]       = re / denom;
            dataC[(j + 1) * stepC] = im / denom;
          }
        if (k == 1)
          dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
      }
    }

    for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC) {
      if (is_1d && cn == 1) {
        dataC[0] = dataA[0] / (dataB[0] + eps);
        if (cols % 2 == 0)
          dataC[j1] = dataA[j1] / (dataB[j1] + eps);
      }

      if (!conjB)
        for (j = j0; j < j1; j += 2) {
          double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
          double re    = dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1];
          double im    = dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1];
          dataC[j]     = re / denom;
          dataC[j + 1] = im / denom;
        }
      else
        for (j = j0; j < j1; j += 2) {
          double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
          double re    = dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1];
          double im    = dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1];
          dataC[j]     = re / denom;
          dataC[j + 1] = im / denom;
        }
    }
  }
}

//}

/* void fftShift() //{ */

static void fftShift(cv::InputOutputArray _out) {

  cv::Mat out = _out.getMat();

  if (out.rows == 1 && out.cols == 1) {
    // trivially shifted.
    return;
  }

  std::vector<cv::Mat> planes;
  split(out, planes);

  int xMid = out.cols >> 1;
  int yMid = out.rows >> 1;

  bool is_1d = xMid == 0 || yMid == 0;

  if (is_1d) {
    int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
    xMid       = xMid + yMid;

    for (size_t i = 0; i < planes.size(); i++) {
      cv::Mat tmp;
      cv::Mat half0(planes[i], cv::Rect(0, 0, xMid + is_odd, 1));
      cv::Mat half1(planes[i], cv::Rect(xMid + is_odd, 0, xMid, 1));

      half0.copyTo(tmp);
      half1.copyTo(planes[i](cv::Rect(0, 0, xMid, 1)));
      tmp.copyTo(planes[i](cv::Rect(xMid, 0, xMid + is_odd, 1)));
    }
  } else {
    int isXodd = out.cols % 2 == 1;
    int isYodd = out.rows % 2 == 1;
    for (size_t i = 0; i < planes.size(); i++) {
      // perform quadrant swaps...
      cv::Mat q0(planes[i], cv::Rect(0, 0, xMid + isXodd, yMid + isYodd));
      cv::Mat q1(planes[i], cv::Rect(xMid + isXodd, 0, xMid, yMid + isYodd));
      cv::Mat q2(planes[i], cv::Rect(0, yMid + isYodd, xMid + isXodd, yMid));
      cv::Mat q3(planes[i], cv::Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

      if (!(isXodd || isYodd)) {
        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
      } else {
        cv::Mat tmp0, tmp1, tmp2, tmp3;
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

//}

/* weightedCentroid() //{ */

/*
static cv::Point2d weightedCentroid(cv::InputArray _src, int fragmentSize, int Xvec, int Yvec, cv::Point peakLocation, cv::Size weightBoxSize,
                                    double* response) {
  cv::Mat src = _src.getMat();

  int type = src.type();
  CV_Assert(type == CV_32FC1 || type == CV_64FC1);

  int minr = Yvec * fragmentSize + peakLocation.y - (weightBoxSize.height >> 1);
  int maxr = Yvec * fragmentSize + peakLocation.y + (weightBoxSize.height >> 1);
  int minc = Xvec * fragmentSize + peakLocation.x - (weightBoxSize.width >> 1);
  int maxc = Xvec * fragmentSize + peakLocation.x + (weightBoxSize.width >> 1);

  cv::Point2d centroid;
  double      sumIntensity = 0.0;

  // clamp the values to min and max if needed.
  if (minr < 0) {
    minr = 0;
  }

  if (minc < 0) {
    minc = 0;
  }

  if (maxr > src.rows - 1) {
    maxr = src.rows - 1;
  }

  if (maxc > src.cols - 1) {
    maxc = src.cols - 1;
  }

  if (type == CV_32FC1) {
    const float* dataIn = src.ptr<float>();
    dataIn += minr * src.cols;
    for (int y = minr; y <= maxr; y++) {
      for (int x = minc; x <= maxc; x++) {
        centroid.x += (double)x * dataIn[x];
        centroid.y += (double)y * dataIn[x];
        sumIntensity += (double)dataIn[x];
      }

      dataIn += src.cols;
    }
  }
  if (response)
    *response = sumIntensity;

  sumIntensity += DBL_EPSILON;  // prevent div0 problems...

  centroid.x /= sumIntensity;
  centroid.y /= sumIntensity;

  return centroid;
}
*/

//}

/* FftMethod::phaseCorrelateField //{ */

std::vector<cv::Point2d> FftMethod::phaseCorrelateField(cv::Mat& _src1, cv::Mat& _src2, unsigned int X, unsigned int Y, double* response) {

  CV_Assert(_src1.type() == _src2.type());
  CV_Assert(_src1.type() == CV_32FC1 || _src1.type() == CV_64FC1);
  CV_Assert(_src1.size() == _src2.size());

  useNewKernel = true;

  /* double elapsedTimeI, elapsedTime1, elapsedTime2, elapsedTime3, elapsedTime4, elapsedTime5, elapsedTime6, elapsedTimeO; */
  /* elapsedTimeI = 0; */
  /* elapsedTime1 = 0; */
  /* elapsedTime2 = 0; */
  /* elapsedTime3 = 0; */
  /* elapsedTime4 = 0; */
  /* elapsedTime5 = 0; */
  /* elapsedTime6 = 0; */
  /* elapsedTimeO = 0; */

  /* clock_t begin, end, begin_overall; */
  /* begin         = std::clock(); */
  /* begin_overall = std::clock(); */

  std::vector<cv::Point2d> output;

  _src1.copyTo(usrc1);
  _src2.copyTo(usrc2);

  cv::Mat src1, src2, h_window1, h_window2;
  _src1.copyTo(src1);
  _src2.copyTo(src2);

  /* usrc1 = _src1.getUMat(cv::ACCESS_READ); */
  /* usrc2 = _src2.getUMat(cv::ACCESS_READ); */

  /* usrc1. */
  /* usrc2.setTo(_src2); */

  /* int M = cv::getOptimalDFTSize(samplePointSize); */
  /* int N = cv::getOptimalDFTSize(samplePointSize); */
  /* ROS_INFO("[OpticFlow]: M: %d",M); */

  cv::Rect roi;

  /* end          = std::clock(); */

  /* elapsedTimeI = double(end - begin) / CLOCKS_PER_SEC; */
  /* ROS_INFO("[OpticFlow]: INITIALIZATION: %f s, %f Hz", elapsedTimeI , 1.0 / elapsedTimeI); */

  /* cv::Mat showhost; */
  std::vector<cv::Point2f> peakLocs;
  peakLocs.resize(sqNum * sqNum);
  if (useNewKernel) {
    phaseCorrelate_ocl(usrc1, usrc2, peakLocs, Y, X);
    /* PCR(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(showhost); */
    /* for (int i = 0; i < ((int)(peakLocs.size())); i++) { */
    /*   std::cout << "out " << i << " = " << peakLocs[i] << std::endl; */
    /* } */
  }
  for (unsigned int j = 0; j < Y; j++) {
    for (unsigned int i = 0; i < X; i++) {

      /* begin = std::clock(); */

      if (!useNewKernel) {
        /* if ((!useNewKernel) || ((j==(Y-2)) && (i==(X-1)))){ */
        /* if ((!useNewKernel) || ((j==0) && (i==0))){ */
        xi  = i * samplePointSize;
        yi  = j * samplePointSize;
        roi = cv::Rect(xi, yi, samplePointSize, samplePointSize);


        /* if (useOCL) { */
        /*   FFT1 = FFT1_field[j][i]; */
        /*   FFT2 = FFT2_field[j][i]; */
        /* } */

        /* if (!useOCL) { */

        if (useOCL) {
          window1 = usrc1(roi);
          window2 = usrc2(roi);
          /* FFTR1(cv::Rect(0,0,samplePointSize/4,samplePointSize)).copyTo(storageA); */
          /* FFTR1(cv::Rect(0,0,samplePointSize/2,samplePointSize)).copyTo(storageA); */
          /* FFT1(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(storageA); */
          dft_special(window1, FFT1, cv::DFT_REAL_OUTPUT);
          dft_special(window2, FFT2, cv::DFT_REAL_OUTPUT);
          mulSpectrums(FFT1, FFT2, P, 0, true);
          magSpectrums(P, Pm);
          divSpectrums(P, Pm, C, 0, false);  // FF* / |FF*| (phase correlation equation completed here...)
          idft_special(C, C);                // gives us the nice peak shift location...
          fftShift(C);                       // shift the energy to the center of the frame.
          C(cv::Rect(0, 0, samplePointSize, samplePointSize)).copyTo(storageB);
          PCR(cv::Rect(0, 0, samplePointSize, samplePointSize)).copyTo(storageA);
          /* diffmap = (storageA) - (storageB); */
          /* ros::Duration(0.5).sleep(); */
          /* s(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(storage); */
        } else {
          h_window1 = src1(roi);
          h_window2 = src2(roi);
          /* FFT1(cv::Rect(0,0,samplePointSize/2,samplePointSize)).copyTo(storageA); */
          dft(h_window1, H_FFT1, cv::DFT_REAL_OUTPUT);
          /* FFT1.copyTo(storageB); */
          dft(h_window2, H_FFT2, cv::DFT_REAL_OUTPUT);
          mulSpectrums(H_FFT1, H_FFT2, H_P, 0, true);
          magSpectrums(H_P, H_Pm);
          divSpectrums(H_P, H_Pm, H_D, 0, false);  // FF* / |FF*| (phase correlation equation completed here...)
          idft(H_D, H_C);                          // gives us the nice peak shift location...
          fftShift(H_C);                           // shift the energy to the center of the frame.
          /* FFT1(cv::Rect(i*samplePointSize,j*samplePointSize,samplePointSize,samplePointSize)).copyTo(storageA); */
          /* MUL.copyTo(storageA); */
          /* IFFTC(cv::Rect(0,0,frameSize,frameSize/2)).copyTo(storageA); */
          /* IFFTC.copyTo(storageA); */
          PCR.copyTo(storageA);
          /* FFTR1(cv::Rect(i*samplePointSize,j*samplePointSize,samplePointSize/2,samplePointSize)).copyTo(storageA); */
          /* FFTR1(cv::Rect(i*samplePointSize,j*samplePointSize,samplePointSize/2,samplePointSize)).copyTo(storageA); */
          /* H_FFT1(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(storageB); */
          H_C(cv::Rect(0, 0, samplePointSize, samplePointSize)).copyTo(storageB);
        }
        /* cv::Mat storageData; */
        /* PCR.copyTo(storageData); */
        /* std::cout<< "Starting to vomit the output data array: " << std::endl; */

        /* for (int n=0; n<16*samplePointSize*2; n++){ */
        /* std::cout << n << ": " << storageData.at<uint>(n) << " | "; */
        /* } */

        /* std::cout<< "Ending the vomitting."<< std::endl; */


        /* if ((j==0) && (i==0)){ */
        /* cv::Mat diffmap = (storageB) - (storageA(cv::Rect(i*samplePointSize,j*samplePointSize,samplePointSize,samplePointSize))); */
        /* showFMat(diffmap); */
        /* showFMat(storageB, "OLD"); */
        /* showFMat(storageData, "PCR"); */
        /* showFMat(storageA(cv::Rect(i*samplePointSize,j*samplePointSize,samplePointSize,samplePointSize)),"NEW"); */
        /* showFMat(storageA,"FULL"); */
        /* showFMat(ML,"DATA"); */
        /* } */
      }
      /* PCR.copyTo(storageA); */
      /*   showFMat(storageA,"ocv_NEW"); */
      /* IFFTC.copyTo(storageB); */
      /* showFMat(storageB,"ocv_NEW"); */

      /* PCR(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(storageA); */
      /*     showFMat(storageA,"NEW"); */

      // locate the highest peak
      /* minMaxLoc(C, NULL, NULL, NULL, &(peakLocs[i+j*sqNum])); */

      // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
      cv::Point2d t;
      /* t = weightedCentroid(C,i,j, samplePointSize, peakLocs[i+j*sqNum], cv::Size(5, 5), response); */
      t = cv::Point2d(peakLocs[i + j * sqNum]);

      // max response is M*N (not exactly, might be slightly larger due to rounding errors)
      if (response)
        *response /= samplePointSize * samplePointSize;

      /* if (i==0 && j==0) */
      /* std::cout << "RAW: " << peakLocs[i+j*sqNum] << " SUBPIX: " << t.x << ":" << t.y <<  std::endl; */

      /* end         = std::clock(); */
      /* elapsedTime = double(end - begin) / CLOCKS_PER_SEC; */
      /* ROS_INFO("[OpticFlow]: Step 4: %f s, %f Hz", elapsedTime , 1.0 / elapsedTime); */
      /* begin = std::clock(); */

      // adjust shift relative to image center...
      /* cv::Point2d center((double)samplePointSize / 2.0, samplePointSize / 2.0); */

      output.push_back(t);
      /* output.push_back(cv::Point(0,0)); */
    }
  }
  /* ROS_INFO("[OpticFlow]: Step 1: %f s, %f Hz", elapsedTime1 , 1.0 / elapsedTime1); */
  /* ROS_INFO("[OpticFlow]: Step 2: %f s, %f Hz", elapsedTime2 , 1.0 / elapsedTime2); */
  /* ROS_INFO("[OpticFlow]: Step 3: %f s, %f Hz", elapsedTime3 , 1.0 / elapsedTime3); */
  /* ROS_INFO("[OpticFlow]: Step 4: %f s, %f Hz", elapsedTime4 , 1.0 / elapsedTime4); */
  /* ROS_INFO("[OpticFlow]: Step 5: %f s, %f Hz", elapsedTime5 , 1.0 / elapsedTime5); */
  /* ROS_INFO("[OpticFlow]: Step 6: %f s, %f Hz", elapsedTime6 , 1.0 / elapsedTime6); */
  /* end          = std::clock(); */
  /* elapsedTimeO = double(end - begin_overall) / CLOCKS_PER_SEC; */
  /* ROS_INFO("[OpticFlow]: OVERALL: %f s, %f Hz", elapsedTimeO , 1.0 / elapsedTimeO); */
  return output;
}

//}
//
/* FftMethod::phaseCorrelateFieldLongRange //{ */

std::vector<cv::Point2d> FftMethod::phaseCorrelateFieldLongRange(cv::Mat& _src1, cv::Mat& _src2, unsigned int X, unsigned int Y, double* response) {

  CV_Assert(_src1.type() == _src2.type());
  CV_Assert(_src1.type() == CV_32FC1 || _src1.type() == CV_64FC1);
  CV_Assert(_src1.size() == _src2.size());

  useNewKernel = true;

  std::vector<cv::Point2d> output;

  _src1.copyTo(usrc1);
  _src2.copyTo(usrc2);

  cv::Mat src1, src2, h_window1, h_window2;
  _src1.copyTo(src1);
  _src2.copyTo(src2);


  cv::Rect roi;


  /* cv::Mat showhost; */
  std::vector<cv::Point2f> peakLocs;
  peakLocs.resize(sqNum_lr * sqNum_lr);
  if (useNewKernel) {
    phaseCorrelate_lr_ocl(usrc1, usrc2, peakLocs, Y, X);
    /* PCR(cv::Rect(0,0,samplePointSize,samplePointSize)).copyTo(showhost); */
    /* for (int i = 0; i < ((int)(peakLocs.size())); i++) { */
    /*   std::cout << "out " << i << " = " << peakLocs[i] << std::endl; */
    /* } */
  }
  for (unsigned int j = 0; j < Y; j++) {
    for (unsigned int i = 0; i < X; i++) {

      /* begin = std::clock(); */

      if (!useNewKernel) {
        /* if ((!useNewKernel) || ((j==(Y-2)) && (i==(X-1)))){ */
        /* if ((!useNewKernel) || ((j==0) && (i==0))){ */
        xi  = i * samplePointSize_lr;
        yi  = j * samplePointSize_lr;
        roi = cv::Rect(xi, yi, samplePointSize_lr, samplePointSize_lr);


        /* if (useOCL) { */
        /*   FFT1 = FFT1_field[j][i]; */
        /*   FFT2 = FFT2_field[j][i]; */
        /* } */

        /* if (!useOCL) { */

        if (useOCL) {
          window1 = usrc1(roi);
          window2 = usrc2(roi);
          dft_special(window1, FFT1, cv::DFT_REAL_OUTPUT);
          dft_special(window2, FFT2, cv::DFT_REAL_OUTPUT);
          mulSpectrums(FFT1, FFT2, P, 0, true);
          magSpectrums(P, Pm);
          divSpectrums(P, Pm, C, 0, false);  // FF* / |FF*| (phase correlation equation completed here...)
          idft_special(C, C);                // gives us the nice peak shift location...
          fftShift(C);                       // shift the energy to the center of the frame.
          C(cv::Rect(0, 0, samplePointSize_lr, samplePointSize_lr)).copyTo(storageB);
          PCR(cv::Rect(0, 0, samplePointSize_lr, samplePointSize_lr)).copyTo(storageA);
        } else {
          h_window1 = src1(roi);
          h_window2 = src2(roi);
          dft(h_window1, H_FFT1, cv::DFT_REAL_OUTPUT);
          /* FFT1.copyTo(storageB); */
          dft(h_window2, H_FFT2, cv::DFT_REAL_OUTPUT);
          mulSpectrums(H_FFT1, H_FFT2, H_P, 0, true);
          magSpectrums(H_P, H_Pm);
          divSpectrums(H_P, H_Pm, H_D, 0, false);  // FF* / |FF*| (phase correlation equation completed here...)
          idft(H_D, H_C);                          // gives us the nice peak shift location...
          fftShift(H_C);                           // shift the energy to the center of the frame.
          PCR.copyTo(storageA);
          H_C(cv::Rect(0, 0, samplePointSize_lr, samplePointSize_lr)).copyTo(storageB);
        }
      }

      // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
      cv::Point2d t;
      t = cv::Point2d(peakLocs[i + j * sqNum_lr]);

      // max response is M*N (not exactly, might be slightly larger due to rounding errors)
      if (response)
        *response /= samplePointSize_lr * samplePointSize_lr;


      output.push_back(t);
      /* output.push_back(cv::Point(0,0)); */
    }
  }
  return output;
}

//}

/* FftMethod::FftMethod() //{ */

FftMethod::FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
                     bool i_tilt_corr_enable, std::string* videoPath, int videoFPS, std::string i_cl_file_name, bool i_useOCL) {

  frameSize          = i_frameSize;
  samplePointSize    = i_samplePointSize;
  samplePointSize_lr = 1 * i_samplePointSize;
  max_px_speed_sq    = pow(max_px_speed_t, 2);
  max_px_speed_lr    = 1 * max_px_speed_t;
  max_px_speed_sq_lr = pow(max_px_speed_lr, 2);
  ;

  cl_file_name = i_cl_file_name;
  useOCL       = i_useOCL;

  storeVideo = i_storeVideo;
  if (storeVideo) {
#ifdef ROS_MELODIC
    outputVideo.open(*videoPath, CV_FOURCC('M', 'P', 'E', 'G'), videoFPS, cv::Size(frameSize, frameSize), false);
#endif
#ifdef ROS_NOETIC
    outputVideo.open(*videoPath, cv::VideoWriter::fourcc('M', 'P', 'E', 'G'), videoFPS, cv::Size(frameSize, frameSize), false);
#endif
    if (!outputVideo.isOpened())
      ROS_ERROR("[OpticFlow]: Could not open output video file: %s", videoPath->c_str());
  }

  if ((frameSize % 2) == 1) {
    frameSize--;
  }
  if ((frameSize % samplePointSize) != 0) {
    ROS_WARN(
        "FS: %d, SPS: %d - Oh, what kind of setting for OpticFlow is this? Frame size must be a multiple of SamplePointSize! Forcing FrameSize = "
        "SamplePointSize (i.e. one "
        "window)..",
        frameSize, samplePointSize);
    samplePointSize = frameSize;
  }


  sqNum    = frameSize / samplePointSize;
  sqNum_lr = sqNum / LONG_RANGE_RATIO;


  usrc1.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  usrc2.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  /* window1.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  /* window2.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  FFT1.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  FFT2.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  FFTR1.create(frameSize, frameSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  FFTR2.create(frameSize, frameSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  MUL.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  IFFTC.create(frameSize, frameSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  /* PCR.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  PCR.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  /* C.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  D.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  T.create(frameSize, frameSize, CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  ML.create(1, sqNum * sqNum * samplePointSize * (sizeof(uint) + sizeof(float)), CV_32FC1, cv::USAGE_ALLOCATE_DEVICE_MEMORY);

  L_SMEM.create(1, samplePointSize * samplePointSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  L_MAXVAL.create(1, samplePointSize * samplePointSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);
  L_MAXLOC.create(1, samplePointSize * samplePointSize, CV_32FC2, cv::USAGE_ALLOCATE_DEVICE_MEMORY);


  mainQueue.create(cv::ocl::Context::getDefault(false), cv::ocl::Device::getDefault());

  H_FFT1.create(samplePointSize, samplePointSize, CV_32FC1);
  H_FFT2.create(samplePointSize, samplePointSize, CV_32FC1);
  H_FFTR1.create(samplePointSize, samplePointSize, CV_32FC2);
  H_FFTR2.create(samplePointSize, samplePointSize, CV_32FC2);
  H_MUL.create(samplePointSize, samplePointSize, CV_32FC1);
  H_IFFTC.create(samplePointSize, samplePointSize, CV_32FC2);
  /* PCR.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  H_PCR.create(samplePointSize, samplePointSize, CV_32FC1);
  /* C.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY); */
  H_D.create(samplePointSize, samplePointSize, CV_32FC1);
  H_T.create(samplePointSize, samplePointSize, CV_32FC1);
  H_ML.create(1, sqNum * sqNum * samplePointSize * (sizeof(uint) + sizeof(float)) * 2, CV_32FC1);


  first = true;
  /* gotBoth = false; */
  running = false;
  /* Nreps = 0; */
  /* gotNth = false; */
}

//}

/* FftMethod::FftMethod() //{ */

std::vector<cv::Point2d> FftMethod::processImage(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center,
                                                 std::vector<cv::Point2d>& raw_output, double i_fx, double i_fy) {

  if (running)
    return std::vector<cv::Point2d>();
  running = true;

  /* ROS_INFO("[OpticFlow]: FX:%f, FY%f",i_fx,i_fy); */

  fx = i_fx;
  fy = i_fy;


  // save image for GUI
  if (gui || storeVideo) {
    imView = imCurr.clone();
  }

  // copy first to second
  if (first) {
    imCurr.copyTo(imPrev);
  }

  if (debug) {
    ROS_INFO("[OpticFlow]: Curr type: %d prev type: %d", imCurr.type(), imPrev.type());
  }

  // convert images to float images
  /* if (Nreps > 30) */
  /*   gotNth = true; */

  /* if (!gotNth){ */
  /* std::cout <<" Here A" <<std::endl; */
  imCurr.convertTo(imCurrF, CV_32FC1);
  imPrev.convertTo(imPrevF, CV_32FC1);
  /* Nreps++; */
  /* } */

  /* if (!first) { */
  /*   gotBoth = true; */
  /* } */

  // clear the vector with speeds
  speeds.clear();

  /* double midX = imCurr.cols / 2; */
  /* double midY = imCurr.rows / 2; */

  /* double distX, distY;  // distance from middle */
  /* double corrX, corrY;  // yaw corrections */

  // calculate correlation for each window and store it if it doesn't exceed the limit
  if (useOCL)
    speeds = phaseCorrelateField(imCurrF, imPrevF, sqNum, sqNum);
  else
    speeds.resize(sqNum * sqNum);

  for (int j = 0; j < sqNum; j++) {
    for (int i = 0; i < sqNum; i++) {
      xi = i * samplePointSize;
      yi = j * samplePointSize;
      if (useOCL)
        shift = speeds[i + sqNum * j];
      else
        shift = -cv::phaseCorrelate(imCurrF(cv::Rect(xi, yi, samplePointSize, samplePointSize)), imPrevF(cv::Rect(xi, yi, samplePointSize, samplePointSize)));

      shift_raw = shift;

      bool valid = true;
      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
          absd(shift.y) > ((double)samplePointSize / 2)) {
        ROS_WARN("[OpticFlow]: FFT - shift is too large (%f; %f) in window x %d y %d", shift.x, shift.y, i, j);
        valid = false;
      }
      if ((isnan(shift.x)) || (isnan(shift.y))) {
        ROS_WARN("[OpticFlow]: FFT - NaN optical flow response in window x %d y %d", i, j);
        valid = false;
      }

      if (!valid) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        speeds[i + j * sqNum] = cv::Point2d(nan(""), nan(""));
      } else {
        speeds[i + j * sqNum] = cv::Point2d(shift.x, shift.y);
      }

      // draw nice lines if gui is enabled
      if (gui || storeVideo) {
        if (valid)
          cv::line(imView, cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2),
                   cv::Point2i(xi + samplePointSize / 2, yi + samplePointSize / 2) + cv::Point2i((int)(shift.x * 5.0), (int)(shift.y * 5.0)), cv::Scalar(255),
                   valid ? 5 : 1);
      }
    }
  }

  /* cv::circle(imView, cv::Point2i(rot_center), 5, cv::Scalar(255),5); */
  /* cv::line(imView, cv::Point2i(imView.size() / 2), */
  /*          cv::Point2i(imView.size() / 2) + cv::Point2i(tan(tiltCorr_dynamic.x) * fx * 5, tan(tiltCorr_dynamic.y) * fy * 5), cv::Scalar(155), 5); */

  imPrev = imCurr.clone();

  if (gui) {
    /* std::vector<cv::Mat> mulmats; */
    /* cv::Mat catmat; */
    /* cv::split(FFT1.getMat(cv::ACCESS_READ), mulmats); */
    /* hconcat(mulmats,catmat); */

    /* double min; */
    /* double max; */
    /* cv::minMaxIdx(catmat, &min, &max); */
    /* std::cout << "IMMIN: " << min << " IMMAX: " << max << std::endl; */
    /* cv::convertScaleAbs(catmat, catmat, 255 / max); */
    /* /1* cv::minMaxIdx(usrc2, &min, &max); *1/ */
    /* /1* std::cout << "IMMIN: " << min << " IMMAX: " << max << std::endl; *1/ */
    /* /1* cv::convertScaleAbs(usrc2, catmat, 255 / max); *1/ */
    /* imshow("debugshit",catmat); */
    /* imshow("debugshit",usrc1); */
    /* ROS_INFO("[%s]: Showing image", ros::this_node::getName().c_str()); */
    cv::imshow("ocv_optic_flow", imView);
    cv::waitKey(1);
  }

  if (storeVideo) {
    outputVideo << imView;
  }

  running = false;
  first   = false;

  return speeds;
}

std::vector<cv::Point2d> FftMethod::processImageLongRange(cv::Mat imCurr, bool gui, bool debug, cv::Point midPoint_t, double yaw_angle, cv::Point2d rot_center,
                                                          std::vector<cv::Point2d>& raw_output, double i_fx, double i_fy) {

  if (running)
    return std::vector<cv::Point2d>();
  running = true;

  /* ROS_INFO("[OpticFlow]: FX:%f, FY%f",i_fx,i_fy); */

  fx = i_fx;
  fy = i_fy;


  // save image for GUI
  // copy first to second
  if (first) {
    imCurr.copyTo(imPrev);
  }

  if (debug) {
    ROS_INFO("[OpticFlow]: Curr type: %d prev type: %d", imCurr.type(), imPrev.type());
  }

  ROS_INFO("[OpticFlow]: Using long range mode");

  cv::Mat imCurrD, imPrevD;
  cv::resize(imCurr, imCurrD, cv::Size(), 1.0 / LONG_RANGE_RATIO, 1.0 / LONG_RANGE_RATIO);
  cv::resize(imPrev, imPrevD, cv::Size(), 1.0 / LONG_RANGE_RATIO, 1.0 / LONG_RANGE_RATIO);

  if (gui || storeVideo) {
    imView = imCurrD.clone();
  }


  imCurrD.convertTo(imCurrF, CV_32FC1);
  imPrevD.convertTo(imPrevF, CV_32FC1);
  speeds.clear();

  if (useOCL) {
    /* ROS_WARN("TODO!"); */
    speeds = phaseCorrelateFieldLongRange(imCurrF, imPrevF, sqNum_lr, sqNum_lr);
  } else
    speeds.resize(sqNum_lr * sqNum_lr);

  for (int j = 0; j < sqNum_lr; j++) {
    for (int i = 0; i < sqNum_lr; i++) {
      xi = i * samplePointSize_lr;
      yi = j * samplePointSize_lr;
      /* ROS_INFO_STREAM("HERE A: sqNum: " << sqNum_lr << " SPS: " << samplePointSize_lr << " xi: " << xi << " yi: " << yi); */
      if (useOCL)
        shift = speeds[i + sqNum_lr * j];
      else
        shift = -cv::phaseCorrelate(imCurrF(cv::Rect(xi, yi, samplePointSize_lr, samplePointSize_lr)),
                                    imPrevF(cv::Rect(xi, yi, samplePointSize_lr, samplePointSize_lr)));

      /* ROS_INFO("HERE B"); */
      shift_raw = shift;

      bool valid = true;
      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq_lr || absd(shift.x) > ((double)samplePointSize_lr / 2) ||
          absd(shift.y) > ((double)samplePointSize_lr / 2)) {
        ROS_WARN("[OpticFlow]: FFT - shift is too large (%f; %f) in window x %d y %d", shift.x, shift.y, i, j);
        valid = false;
      }
      if ((isnan(shift.x)) || (isnan(shift.y))) {
        ROS_WARN("[OpticFlow]: FFT - NaN optical flow response in window x %d y %d", i, j);
        valid = false;
      }

      if (!valid) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        speeds[i + j * sqNum_lr] = cv::Point2d(nan(""), nan(""));
      } else {
        speeds[i + j * sqNum_lr] = cv::Point2d(shift.x, shift.y);
      }

      // draw nice lines if gui is enabled
      if (gui || storeVideo) {
        if (valid)
          cv::line(imView, cv::Point2i(xi + samplePointSize_lr / 2, yi + samplePointSize_lr / 2),
                   cv::Point2i(xi + samplePointSize_lr / 2, yi + samplePointSize_lr / 2) + cv::Point2i((int)(shift.x * 5.0), (int)(shift.y * 5.0)),
                   cv::Scalar(255), valid ? 5 : 1);
      }
    }
  }


  imPrev = imCurr.clone();

  if (gui) {
    cv::imshow("ocv_optic_flow", imView);
    cv::waitKey(1);
  }

  if (storeVideo) {
    outputVideo << imView;
  }

  running = false;
  first   = false;

  return speeds;
}

//}
