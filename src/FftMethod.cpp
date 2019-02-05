#include "../include/mrs_optic_flow/FftMethod.h"


cv::ocl::ProgramSource prep_ocl_kernel(const char* filename){
  std::cout << "Loading OpenCL kernel file \" " << filename << std::endl;
  std::ifstream ist(filename, std::ifstream::in);
  std::string str((std::istreambuf_iterator<char>(ist)),
      std::istreambuf_iterator<char>());

  if (str.empty())
    std::cerr << "Could not load the file. Aborting" << std::endl;

  return cv::ocl::ProgramSource(str.c_str());
}

enum FftType
{
    R2R = 0, // real to CCS in case forward transform, CCS to real otherwise
    C2R = 1, // complex to real in case inverse transform
    R2C = 2, // real to complex in case forward transform
    C2C = 3  // complex to complex
};

static int
DFTFactorize( int n, int* factors )
{
    int nf = 0, f, i, j;

    if( n <= 5 )
    {
        factors[0] = n;
        return 1;
    }

    f = (((n - 1)^n)+1) >> 1;
    if( f > 1 )
    {
        factors[nf++] = f;
        n = f == n ? 1 : n/f;
    }

    for( f = 3; n > 1; )
    {
        int d = n/f;
        if( d*f == n )
        {
            factors[nf++] = f;
            n = d;
        }
        else
        {
            f += 2;
            if( f*f > n )
                break;
        }
    }

    if( n > 1 )
        factors[nf++] = n;

    f = (factors[0] & 1) == 0;
    for( i = f; i < (nf+f)/2; i++ )
        CV_SWAP( factors[i], factors[nf-i-1+f], j );

    return nf;
}


OCL_FftPlan::OCL_FftPlan(int _size, int _depth, std::string i_cl_file_name) : dft_size(_size), dft_depth(_depth), status(true)
    {
      cl_file_name = i_cl_file_name;

        CV_Assert( dft_depth == CV_32F || dft_depth == CV_64F );

        int min_radix;
        std::vector<int> radixes, blocks;
        ocl_getRadixes(dft_size, radixes, blocks, min_radix);
        double thread_count_deprecated = dft_size / min_radix;

        if (thread_count_deprecated > (int) cv::ocl::Device::getDefault().maxWorkGroupSize())
        {
            status = false;
            return;
        }

        // generate string with radix calls
        cv::String radix_processing;
        int n = 1, twiddle_size = 0;
        for (size_t i=0; i<radixes.size(); i++)
        {
            int radix = radixes[i], block = blocks[i];
            if (block > 1)
                radix_processing += cv::format("fft_radix%d_B%d(smem,twiddles+%d,ind,%d,%d);", radix, block, twiddle_size, n, dft_size/radix);
            else
                radix_processing += cv::format("fft_radix%d(smem,twiddles+%d,ind,%d,%d);", radix, twiddle_size, n, dft_size/radix);
            twiddle_size += (radix-1)*n;
            n *= radix;
        }

        twiddles.create(1, twiddle_size, CV_MAKE_TYPE(dft_depth, 2));
        if (dft_depth == CV_32F)
            fillRadixTable<float>(twiddles, radixes);
        else
            fillRadixTable<double>(twiddles, radixes);


        std::string buildOptions = cv::format("-D LOCAL_SIZE=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s",
                              dft_size, min_radix, cv::ocl::typeToStr(dft_depth), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)),
                              dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str());

    }

    /* bool OCL_FftPlan::enqueueTransform(cv::InputArray _src, cv::OutputArray _dst, int num_dfts, int flags, int fftType, bool rows) */
    /* { */
    /*     if (!status) */
    /*         return false; */

    /*     cv::UMat src = _src.getUMat(); */
    /*     cv::UMat dst = _dst.getUMat(); */

    /*     size_t globalsize[2]; */
    /*     size_t localsize[2]; */
    /*     cv::String kernel_name; */

    /*     bool inv = (flags & cv::DFT_INVERSE) != 0; */
    /*     cv::String options = buildOptions; */

    /*     if (rows) */
    /*     { */
    /*         globalsize[0] = thread_count; */
    /*         globalsize[1] = src.rows; */
    /*         localsize[0] = thread_count; */
    /*         localsize[1] = 1; */
    /*         kernel_name = !inv ? "fft_multi_radix_rows" : "ifft_multi_radix_rows"; */
    /*         if ((inv) && (flags & cv::DFT_SCALE)) */
    /*             options += " -D DFT_SCALE"; */
    /*     } */
    /*     else */
    /*     { */
    /*         globalsize[1] = num_dfts; */
    /*         globalsize[0] = thread_count; */
    /*         localsize[1] = 1; */
    /*         localsize[0] = thread_count; */
    /*         kernel_name = !inv ? "fft_multi_radix_cols" : "ifft_multi_radix_cols"; */
    /*         if (flags & cv::DFT_SCALE) */
    /*             options += " -D DFT_SCALE"; */
    /*     } */

    /*     options += src.channels() == 1 ? " -D REAL_INPUT" : " -D COMPLEX_INPUT"; */
    /*     if (flags & cv::DFT_REAL_OUTPUT){ */
    /*       options += " -D REAL_OUTPUT"; */
    /*     } */
    /*     else { */
    /*       options += " -D COMPLEX_OUTPUT"; */
    /*     } */

    /*     if (!inv) */
    /*     { */
    /*         if ((src.channels() == 1) || (rows && (fftType == R2R))) */
    /*             options += " -D NO_CONJUGATE"; */
    /*     } */
    /*     else */
    /*     { */
    /*         if (rows && (fftType == C2R || fftType == R2R)) */
    /*             options += " -D NO_CONJUGATE"; */
    /*         if (dst.cols % 2 == 0) */
    /*             options += " -D EVEN"; */
    /*     } */


    /*     if (rows) */
    /*     { */
    /*       if (inv){ */
    /*         if (k_fft_inv_row.empty()) */
    /*           k_fft_inv_row = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options); */
    /*         if (k_fft_inv_row.empty()){ */
    /*           return false; */
    /*         } */
    /*         k_fft_inv_row.args(cv::ocl::KernelArg::ReadOnly(src), cv::ocl::KernelArg::WriteOnly(dst), cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts); */
    /*         return k_fft_inv_row.run(2, globalsize, localsize, true); */
    /*       } */
    /*       else{ */
    /*         if (k_fft_forw_row.empty()) */
    /*           k_fft_forw_row = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options); */
    /*         if (k_fft_forw_row.empty()){ */
    /*           return false; */
    /*         } */
    /*         k_fft_forw_row.args(cv::ocl::KernelArg::ReadOnly(src), cv::ocl::KernelArg::WriteOnly(dst), cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts); */
    /*         return k_fft_forw_row.run(2, globalsize, localsize, true); */
    /*       } */
    /*     }else{ */
    /*       if (inv){ */
    /*         if (k_fft_inv_col.empty()) */
    /*           k_fft_inv_col = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options); */
    /*         if (k_fft_inv_col.empty()){ */
    /*           return false; */
    /*         } */
    /*         k_fft_inv_col.args(cv::ocl::KernelArg::ReadOnly(src), cv::ocl::KernelArg::WriteOnly(dst), cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts); */
    /*         return k_fft_inv_col.run(2, globalsize, localsize, true); */
    /*       } */
    /*       else{ */
    /*         if (k_fft_forw_col.empty()) */
    /*           k_fft_forw_col = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options); */
    /*         if (k_fft_forw_col.empty()){ */
    /*           return false; */
    /*         } */
    /*         k_fft_forw_col.args(cv::ocl::KernelArg::ReadOnly(src), cv::ocl::KernelArg::WriteOnly(dst), cv::ocl::KernelArg::ReadOnlyNoSize(twiddles), thread_count, num_dfts); */
    /*         return k_fft_forw_col.run(2, globalsize, localsize, true); */
    /*       } */
    /*     } */


    /* } */

    bool OCL_FftPlan::enqueueTransform(cv::InputArray _src1, cv::InputArray _src2, cv::InputOutputArray _fft1, cv::InputArray _fft2, cv::InputArray _mul, cv::InputArray _pcr, cv::OutputArray _dst, int num_dfts,int Xfields,int Yfields, std::vector<cv::Point> &output,int thread_count,int block_count)
    {
      if (!status)
        return false;

      const cv::ocl::Device & dev = cv::ocl::Device::getDefault();

      cv::UMat src1 = _src1.getUMat();
      cv::UMat src2 = _src2.getUMat();
      cv::UMat fft1 = _fft1.getUMat();
      cv::UMat fft2 = _fft2.getUMat();
      cv::UMat mul = _mul.getUMat();
      cv::UMat pcr = _pcr.getUMat();
      cv::UMat dst = _dst.getUMat();

      size_t globalsize[2];
      size_t localsize[2];
      cv::String kernel_name;

      cv::String options = buildOptions;

      kernel_name = "phaseCorrelateField";

      globalsize[0] = thread_count;
      globalsize[1] = block_count;
      localsize[0] = thread_count;
      localsize[1] = 1;

      options += " -D ROW_F_REAL_INPUT";
      options += " -D COL_F_COMPLEX_INPUT";
      options += " -D ROW_F_COMPLEX_OUTPUT";
      options += " -D COL_F_COMPLEX_OUTPUT";
      options += " -D ROW_I_COMPLEX_INPUT";
      options += " -D COL_I_COMPLEX_INPUT";
      options += " -D ROW_I_COMPLEX_OUTPUT";
      options += " -D COL_I_REAL_OUTPUT";
      /* options += " -D NO_CONJUGATE"; */

      if (dst.cols % 2 == 0)
        options += " -D EVEN";

      size_t wgs = dev.maxWorkGroupSize();
      options += " -D WGS="+std::to_string((int)wgs);

      int wgs2_aligned = 1;
      while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
      wgs2_aligned >>= 1;
      options += " -D WGS2_ALIGNED=";
      options += std::to_string(wgs2_aligned);

      std::cout << options << std::endl;
      if (k_phase_corr.empty())
        k_phase_corr = cv::ocl::Kernel(kernel_name.c_str(), prep_ocl_kernel(cl_file_name.c_str()), options);

      if (k_phase_corr.empty()){
        return false;
      }

      k_phase_corr.args(
          cv::ocl::KernelArg::ReadOnly(src1),
          cv::ocl::KernelArg::ReadOnly(src2),
          cv::ocl::KernelArg::ReadWrite(fft1),
          cv::ocl::KernelArg::ReadWrite(fft2),
          cv::ocl::KernelArg::ReadWrite(mul),
          cv::ocl::KernelArg::ReadWrite(pcr),
          cv::ocl::KernelArg::PtrWriteOnly(dst),
          cv::ocl::KernelArg::ReadOnlyNoSize(twiddles),
          thread_count,
          num_dfts,
          Xfields,
          Yfields
          );

      bool partial = k_phase_corr.run(2, globalsize, localsize, true);

      cv::Mat fft1_host = fft1.getMat(cv::ACCESS_READ);
    std::vector<cv::Mat> mulmats;
    cv::Mat catmat;
    cv::split(fft1_host, mulmats);
    hconcat(mulmats,catmat);

    double min;
    double max;
    cv::minMaxIdx(catmat, &min, &max);
    std::cout << "IMMIN: " << min << " IMMAX: " << max << std::endl;
    cv::convertScaleAbs(catmat, catmat, 255 / max);
    /* cv::minMaxIdx(usrc2, &min, &max); */
    /* std::cout << "IMMIN: " << min << " IMMAX: " << max << std::endl; */
    /* cv::convertScaleAbs(usrc2, catmat, 255 / max); */
    imshow("debugshit",catmat);

      cv::Mat dst_host = dst.getMat(cv::ACCESS_READ);

      float  maxVal = -1;
      int  maxLoc[2];

      size_t index = 0;
      for (int j=0;j<Yfields;j++){
        for (int i=0;i<Xfields;i++){

          uint index_max = std::numeric_limits<uint>::max();
          float maxval = std::numeric_limits<float>::min() > 0 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::min(), maxval2 = maxval;
          uint maxloc = index_max;

          const float * maxptr = NULL, * maxptr2 = NULL;
          const uint * maxlocptr = NULL;
          maxptr = (const float *)(dst_host.ptr() + index);
          index += sizeof(float) * dev.maxComputeUnits();
          index = cv::alignSize(index, 8);
          maxlocptr = (const uint *)(dst_host.ptr() + index);
          index += sizeof(uint) * dev.maxComputeUnits();
          index = cv::alignSize(index, 8);

          for (int i = 0; i < dev.maxComputeUnits(); i++)
          {
            if (maxptr && maxptr[i] >= maxval)
            {
              if (maxptr[i] == maxval)
              {
                if (maxlocptr)
                  maxloc = std::min(maxlocptr[i], maxloc);
              }
              else
              {
                if (maxlocptr)
                  maxloc = maxlocptr[i];
                maxval = maxptr[i];
              }
            }
            if (maxptr2 && maxptr2[i] > maxval2)
              maxval2 = maxptr2[i];
          }

          maxVal = (double)maxval;

          maxLoc[0] = maxloc / fft1.cols;
          maxLoc[1] = maxloc % fft1.cols;

          for (int i=0;i<=1;i++){
            if (maxLoc[i] > fft1.cols/2)
              maxLoc[i]=maxLoc[i] - fft1.cols/2;
            else
              maxLoc[i]=maxLoc[i] + fft1.cols/2;
          }

          output[i+j*Xfields]  = cv::Point2i(maxLoc[0],maxLoc[1]);
    }}
      /* std::cout << "OUT: " << output <<std::endl; */
      return partial;
    }

    void OCL_FftPlan::ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix)
    {
        int factors[34];
        int nf = DFTFactorize(cols, factors);

        int n = 1;
        int factor_index = 0;
        min_radix = INT_MAX;

        // 2^n transforms
        if ((factors[factor_index] & 1) == 0)
        {
            for( ; n < factors[factor_index];)
            {
                int radix = 2, block = 1;
                if (8*n <= factors[0])
                    radix = 8;
                else if (4*n <= factors[0])
                {
                    radix = 4;
                    if (cols % 12 == 0)
                        block = 3;
                    else if (cols % 8 == 0)
                        block = 2;
                }
                else
                {
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
                min_radix = cv::min(min_radix, block*radix);
                n *= radix;
            }
            factor_index++;
        }

        // all the other transforms
        for( ; factor_index < nf; factor_index++)
        {
            int radix = factors[factor_index], block = 1;
            if (radix == 3)
            {
                if (cols % 12 == 0)
                    block = 4;
                else if (cols % 9 == 0)
                    block = 3;
                else if (cols % 6 == 0)
                    block = 2;
            }
            else if (radix == 5)
            {
                if (cols % 10 == 0)
                    block = 2;
            }
            radixes.push_back(radix);
            blocks.push_back(block);
            min_radix = cv::min(min_radix, block*radix);
        }
    }

    template <typename T>
    void OCL_FftPlan::fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes)
    {
      cv::Mat tw = twiddles.getMat(cv::ACCESS_WRITE);
        T* ptr = tw.ptr<T>();
        int ptr_index = 0;

        int n = 1;
        for (size_t i=0; i<radixes.size(); i++)
        {
            int radix = radixes[i];
            n *= radix;

            for (int j=1; j<radix; j++)
            {
                double theta = -CV_2PI*j/n;

                for (int k=0; k<(n/radix); k++)
                {
                    ptr[ptr_index++] = (T) cos(k*theta);
                    ptr[ptr_index++] = (T) sin(k*theta);
                }
            }
        }
    }


void FftMethod::ocl_getRadixes(int cols, std::vector<int>& radixes, std::vector<int>& blocks, int& min_radix)
{
  int factors[34];
  int nf = DFTFactorize(cols, factors);

  int n = 1;
  int factor_index = 0;
  min_radix = INT_MAX;

  // 2^n transforms
  if ((factors[factor_index] & 1) == 0)
  {
    for( ; n < factors[factor_index];)
    {
      int radix = 2, block = 1;
      if (8*n <= factors[0])
        radix = 8;
      else if (4*n <= factors[0])
      {
        radix = 4;
        if (cols % 12 == 0)
          block = 3;
        else if (cols % 8 == 0)
          block = 2;
      }
      else
      {
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
      min_radix = cv::min(min_radix, block*radix);
      n *= radix;
    }
    factor_index++;
  }

  // all the other transforms
  for( ; factor_index < nf; factor_index++)
  {
    int radix = factors[factor_index], block = 1;
    if (radix == 3)
    {
      if (cols % 12 == 0)
        block = 4;
      else if (cols % 9 == 0)
        block = 3;
      else if (cols % 6 == 0)
        block = 2;
    }
    else if (radix == 5)
    {
      if (cols % 10 == 0)
        block = 2;
    }
    radixes.push_back(radix);
    blocks.push_back(block);
    min_radix = cv::min(min_radix, block*radix);
  }
}

  template <typename T>
void FftMethod::fillRadixTable(cv::UMat twiddles, const std::vector<int>& radixes)
{
  cv::Mat tw = twiddles.getMat(cv::ACCESS_WRITE);
  T* ptr = tw.ptr<T>();
  int ptr_index = 0;

  int n = 1;
  for (size_t i=0; i<radixes.size(); i++)
  {
    int radix = radixes[i];
    n *= radix;

    for (int j=1; j<radix; j++)
    {
      double theta = -CV_2PI*j/n;

      for (int k=0; k<(n/radix); k++)
      {
        ptr[ptr_index++] = (T) cos(k*theta);
        ptr[ptr_index++] = (T) sin(k*theta);
      }
    }
  }
}


/* bool FftMethod::ocl_dft_rows(cv::InputArray _src, cv::OutputArray _dst, int nonzero_rows, int flags, int fftType) */
/* { */
/*     int type = _src.type(), depth = CV_MAT_DEPTH(type); */
/*     cv::Ptr<OCL_FftPlan> plan = cache.getFftPlan(_src.cols(), depth, cl_file_name); */
/*     return plan->enqueueTransform(_src, _dst, nonzero_rows, flags, fftType, true); */
/* } */

/* bool FftMethod::ocl_dft_cols(cv::InputArray _src, cv::OutputArray _dst, int nonzero_cols, int flags, int fftType) */
/* { */
/*     int type = _src.type(), depth = CV_MAT_DEPTH(type); */
/*     cv::Ptr<OCL_FftPlan> plan = cache.getFftPlan(_src.rows(), depth, cl_file_name); */
/*     return plan->enqueueTransform(_src, _dst, nonzero_cols, flags, fftType, false); */
/* } */

/* bool FftMethod::ocl_dft(cv::InputArray _src, cv::OutputArray _dst, int flags, int nonzero_rows) */
/* { */
/*     int type = _src.type(), cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type); */
/*     cv::Size ssize = _src.size(); */
/*     bool doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0; */

/*     if ( !((cn == 1 || cn == 2) && (depth == CV_32F || (depth == CV_64F && doubleSupport))) ) */
/*         return false; */

/*     // if is not a multiplication of prime numbers { 2, 3, 5 } */
/*     if (ssize.area() != cv::getOptimalDFTSize(ssize.area())) */
/*         return false; */

/*     cv::UMat src = _src.getUMat(); */
/*     int complex_input = cn == 2 ? 1 : 0; */
/*     int complex_output = (flags & cv::DFT_COMPLEX_OUTPUT) != 0; */
/*     int real_input = cn == 1 ? 1 : 0; */
/*     int real_output = (flags & cv::DFT_REAL_OUTPUT) != 0; */
/*     bool inv = (flags & cv::DFT_INVERSE) != 0 ? 1 : 0; */

/*     if( nonzero_rows <= 0 || nonzero_rows > _src.rows() ) */
/*         nonzero_rows = _src.rows(); */

/*     // if output format is not specified */
/*     if (complex_output + real_output == 0) */
/*     { */
/*         if (real_input) */
/*             real_output = 1; */
/*         else */
/*             complex_output = 1; */
/*     } */

/*     FftType fftType = (FftType)(complex_input << 0 | complex_output << 1); */

/*     // Forward Complex to CCS not supported */
/*     if (fftType == C2R && !inv) */
/*         fftType = C2C; */

/*     // Inverse CCS to Complex not supported */
/*     if (fftType == R2C && inv) */
/*         fftType = R2R; */

/*     cv::UMat output; */
/*     if (fftType == C2C || fftType == R2C) */
/*     { */
/*         // complex output */
/*         _dst.create(src.size(), CV_MAKETYPE(depth, 2)); */
/*         output = _dst.getUMat(); */
/*     } */
/*     else */
/*     { */
/*         // real output */
/*       _dst.create(src.size(), CV_MAKETYPE(depth, 1)); */
/*       output.create(src.size(), CV_MAKETYPE(depth, 2)); */
/*     } */

/*     if (!inv) */
/*     { */

/*       if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType)) */
/*         return false; */

/*       int nonzero_cols = fftType == R2R ? output.cols/2 + 1 : output.cols; */
/*       if (!ocl_dft_cols(output, _dst, nonzero_cols, flags, fftType)) */
/*         return false; */
/*     } */
/*     else */
/*     { */
/*       if (fftType == C2C) */
/*       { */
/*         // complex output */
/*         if (!ocl_dft_rows(src, output, nonzero_rows, flags, fftType)) */
/*           return false; */

/*         if (!ocl_dft_cols(output, output, output.cols, flags, fftType)) */
/*           return false; */
/*       } */
/*       else */
/*       { */
/*         int nonzero_cols = src.cols/2 + 1; */
/*         if (!ocl_dft_cols(src, output, nonzero_cols, flags, fftType)) */
/*           return false; */


/*         if (!ocl_dft_rows(output, _dst, nonzero_rows, flags, fftType)) */
/*           return false; */
/*       } */
/*     } */
/*     return true; */
/* } */


bool FftMethod::phaseCorrelate_ocl(cv::InputArray _src1,cv::InputArray _src2, std::vector<cv::Point2i> &out, int vec_rows, int vec_cols)
{

  int flags = 0;
  flags |= cv::DFT_REAL_OUTPUT;

  int nonzero_rows = _src1.cols();

  int dft_size = samplePointSize;
  int type = _src1.type();
  int dft_depth = CV_MAT_DEPTH(type);

  CV_Assert( _src1.type() == _src2.type());
  CV_Assert( _src1.size() == _src2.size());
  CV_Assert( dft_depth == CV_32F || dft_depth == CV_64F );

  int min_radix;
  std::vector<int> radixes, blocks;
  ocl_getRadixes(dft_size, radixes, blocks, min_radix);
  int thread_count = dft_size / min_radix;

  bool status;

  if (thread_count > (int) cv::ocl::Device::getDefault().maxWorkGroupSize())
  {
    status = false;
    return false;
  }

  // generate string with radix calls
  cv::String radix_processing;
  int n = 1, twiddle_size = 0;
  for (size_t i=0; i<radixes.size(); i++)
  {
    int radix = radixes[i], block = blocks[i];
    if (block > 1)
      radix_processing += cv::format("fft_radix%d_B%d(smem,twiddles+%d,ind,%d,%d);", radix, block, twiddle_size, n, dft_size/radix);
    else
      radix_processing += cv::format("fft_radix%d(smem,twiddles+%d,ind,%d,%d);", radix, twiddle_size, n, dft_size/radix);
    twiddle_size += (radix-1)*n;
    n *= radix;
  }

  twiddles.create(1, twiddle_size, CV_MAKE_TYPE(dft_depth, 2));
  if (dft_depth == CV_32F)
    fillRadixTable<float>(twiddles, radixes);
  else
    fillRadixTable<double>(twiddles, radixes);

  buildOptions = cv::format("-D LOCAL_SIZE=%d -D kercn=%d -D FT=%s -D CT=%s%s -D RADIX_PROCESS=%s",
      dft_size, min_radix, cv::ocl::typeToStr(dft_depth), cv::ocl::typeToStr(CV_MAKE_TYPE(dft_depth, 2)),
      dft_depth == CV_64F ? " -D DOUBLE_SUPPORT" : "", radix_processing.c_str());
  int cn = CV_MAT_CN(type), depth = CV_MAT_DEPTH(type);
  cv::Size ssize = _src1.size();
  bool doubleSupport = cv::ocl::Device::getDefault().doubleFPConfig() > 0;

  if ( !((cn == 1 || cn == 2) && (depth == CV_32F || (depth == CV_64F && doubleSupport))) )
    return false;

  // if is not a multiplication of prime numbers { 2, 3, 5 }
  if (ssize.area() != cv::getOptimalDFTSize(ssize.area()))
    return false;

  cv::UMat src = _src1.getUMat();
  int complex_input = cn == 2 ? 1 : 0;
  int complex_output = (flags & cv::DFT_COMPLEX_OUTPUT) != 0;
  int real_input = cn == 1 ? 1 : 0;
  int real_output = (flags & cv::DFT_REAL_OUTPUT) != 0;
  bool inv = (flags & cv::DFT_INVERSE) != 0 ? 1 : 0;


  // if output format is not specified
  if (complex_output + real_output == 0)
  {
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

  cv::Ptr<OCL_FftPlan> plan = cache.getFftPlan(samplePointSize, depth, cl_file_name);
  return plan->enqueueTransform(_src1, _src2,  FFT1, FFT2, MUL, PCR,  ML, nonzero_rows, vec_cols, vec_rows, out, thread_count,samplePointSize);


  return true;
}

/* void FftMethod::dft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags) */
/* { */
/*   ocl_dft(_src0, _dst, flags,0); */
/* } */
/* void FftMethod::idft_special(cv::InputArray _src0, cv::OutputArray _dst, int flags) */
/* { */
/*   ocl_dft(_src0, _dst, flags | cv::DFT_INVERSE,0); */
/* } */

bool FftMethod::ocl_mulSpectrums( cv::InputArray _srcA, cv::InputArray _srcB,
                              cv::OutputArray _dst, int flags, bool conjB )
{
    int atype = _srcA.type(), btype = _srcB.type(),
            rowsPerWI = cv::ocl::Device::getDefault().isIntel() ? 4 : 1;
    cv::Size asize = _srcA.size(), bsize = _srcB.size();
    CV_Assert(asize == bsize);
    /* CV_Assert(atype == CV_32FC2 && btype == CV_32FC2); */

    if (flags != 0 )
        return false;

    cv::UMat A = _srcA.getUMat(), B = _srcB.getUMat();
    CV_Assert(A.size() == B.size());

    _dst.create(A.size(), atype);
    cv::UMat dst = _dst.getUMat();

    cv::ocl::Kernel k("mulAndNormalizeSpectrums",
                  prep_ocl_kernel(cl_file_name.c_str()),
                  buildOptions+" -D CONJ ");
    if (k.empty())
        return false;

    k.args(cv::ocl::KernelArg::ReadOnlyNoSize(A), cv::ocl::KernelArg::ReadOnlyNoSize(B),
          cv::ocl::KernelArg::WriteOnly(dst), rowsPerWI);

    size_t globalsize[2] = { (size_t)asize.width, ((size_t)asize.height + rowsPerWI - 1) / rowsPerWI };
    return k.run(2, globalsize, NULL, false);
}

void FftMethod::mulSpectrums_special( cv::InputArray _srcA, cv::InputArray _srcB,
                       cv::OutputArray _dst, int flags, bool conjB )
{
  int type = _srcA.type();

  CV_Assert( type == _srcB.type() && _srcA.size() == _srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

  ocl_mulSpectrums(_srcA, _srcB, _dst, flags, conjB);
}

static void divSpectrums( cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB)
{
  cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type());
    CV_Assert(  srcA.size() == srcB.size() );
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

static cv::Point2d weightedCentroid(cv::InputArray _src,int fragmentSize, int Xvec,int Yvec, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
  cv::Mat src = _src.getMat();

    int type = src.type();
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    int minr = Yvec*fragmentSize+peakLocation.y - (weightBoxSize.height >> 1);
    int maxr = Yvec*fragmentSize+peakLocation.y + (weightBoxSize.height >> 1);
    int minc = Xvec*fragmentSize+peakLocation.x - (weightBoxSize.width  >> 1);
    int maxc = Xvec*fragmentSize+peakLocation.x + (weightBoxSize.width  >> 1);

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

  useOCL = true;
  useNewKernel = true;


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


  _src1.copyTo(usrc1);
  _src2.copyTo(usrc2);


  /* usrc1 = _src1.getUMat(cv::ACCESS_READ); */
  /* usrc2 = _src2.getUMat(cv::ACCESS_READ); */

  /* usrc1. */
  /* usrc2.setTo(_src2); */



  /* int M = cv::getOptimalDFTSize(samplePointSize); */
  /* int N = cv::getOptimalDFTSize(samplePointSize); */
  /* ROS_INFO("M: %d",M); */

    cv::Rect roi;

    end         = std::clock();
    elapsedTimeI = double(end - begin) / CLOCKS_PER_SEC;
    ROS_INFO("INITIALIZATION: %f s, %f Hz", elapsedTimeI , 1.0 / elapsedTimeI);


    std::vector<cv::Point2i> peakLocs;
    peakLocs.resize(sqNum*sqNum);
    if (useNewKernel){
      phaseCorrelate_ocl(usrc1,usrc1, peakLocs, Y,X);
    }
    for (int i = 0; i < X; i++) {
      for (int j = 0; j < Y; j++) {
        begin = std::clock();


        if (!useNewKernel) {
          xi    = i * samplePointSize;
          yi    = j * samplePointSize;
          roi = cv::Rect(xi,yi,samplePointSize,samplePointSize);


          /* if (useOCL) { */
          /*   FFT1 = FFT1_field[j][i]; */
          /*   FFT2 = FFT2_field[j][i]; */
          /* } */

          /* if (!useOCL) { */
          window1 = usrc1(roi);
          window2 = usrc2(roi);
          /* } */
          /* ROS_INFO_ONCE("padded size: %dx%d",padded2.rows,padded2.cols); */

          end         = std::clock();
          elapsedTime1 += double(end - begin) / CLOCKS_PER_SEC;
          begin = std::clock();

          // execute phase correlation equation
          // Reference: http://en.wikipedia.org/wiki/Phase_correlation
          /* dft(usrc1(roi), FFT1, cv::DFT_REAL_OUTPUT); */
          /* dft(usrc2(roi), FFT2, cv::DFT_REAL_OUTPUT); */
          if (useOCL){
            /* dft_special(window1, FFT1, cv::DFT_REAL_OUTPUT); */
            /* dft_special(window2, FFT2, cv::DFT_REAL_OUTPUT); */
            /* dft_special(window1, FFT1, cv::DFT_REAL_OUTPUT); */
            /* dft_special(window2, FFT2, cv::DFT_REAL_OUTPUT); */
          }
          else {
            dft(window1, FFT1, cv::DFT_REAL_OUTPUT);
            dft(window2, FFT2, cv::DFT_REAL_OUTPUT);
          }

          /* ROS_INFO("[%d]: FFT TYPE", FFT1.type()); */

          /* end         = std::clock(); */
          /* elapsedTime2 += double(end - begin) / CLOCKS_PER_SEC; */
          /* begin = std::clock(); */

          if (useOCL){
            mulSpectrums_special(FFT1, FFT2, C, 0, true);
          }
          else{
            mulSpectrums(FFT1, FFT2, P, 0, true);
          }
          /* cv::Mat tempView; */
          /* cv::normalize(FFT1, tempView, 255,0, cv::NORM_MINMAX, CV_8UC1); */

          /* if (i==0 && j==0) */
          /*   cv::imshow("fft1",tempView); */
          /* ROS_INFO("DEPTH: %d, CHANNELS: %d", FFT1.depth(), FFT1.channels()); */

          /* end         = std::clock(); */
          /* elapsedTime3 += double(end - begin) / CLOCKS_PER_SEC; */
          /* begin = std::clock(); */

          /* magSpectrums(P, Pm); */

          /* end         = std::clock(); */
          /* elapsedTime4 += double(end - begin) / CLOCKS_PER_SEC; */
          /* begin = std::clock(); */

          /* /1* divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...) *1/ */

          /* end         = std::clock(); */
          /* elapsedTime5 += double(end - begin) / CLOCKS_PER_SEC; */
          /* begin = std::clock(); */


          if (useOCL){
            /* idft_special(C, C); // gives us the nice peak shift location... */
          }
          else {
            idft(C, C); // gives us the nice peak shift location...
          }

          end         = std::clock();
          elapsedTime6 += double(end - begin) / CLOCKS_PER_SEC;
          begin = std::clock();

          fftShift(C); // shift the energy to the center of the frame.


          // locate the highest peak
          minMaxLoc(C, NULL, NULL, NULL, &(peakLocs[i+j*sqNum]));

        }
        // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
        cv::Point2d t;
        t = weightedCentroid(C,i,j, samplePointSize, peakLocs[i+j*sqNum], cv::Size(5, 5), response);

        // max response is M*N (not exactly, might be slightly larger due to rounding errors)
        if(response)
          *response /= samplePointSize*samplePointSize;

        /* if (i==0 && j==0) */
          /* std::cout << "RAW: " << peakLocs[i+j*sqNum] << " SUBPIX: " << t.x << ":" << t.y <<  std::endl; */

        /* end         = std::clock(); */
        /* elapsedTime = double(end - begin) / CLOCKS_PER_SEC; */
        /* ROS_INFO("Step 4: %f s, %f Hz", elapsedTime , 1.0 / elapsedTime); */
        /* begin = std::clock(); */

        // adjust shift relative to image center...
        cv::Point2d center((double)samplePointSize / 2.0, samplePointSize / 2.0);

        output.push_back(t-center);
        /* output.push_back(cv::Point(0,0)); */
      }
    }
    /* ROS_INFO("Step 1: %f s, %f Hz", elapsedTime1 , 1.0 / elapsedTime1); */
    /* ROS_INFO("Step 2: %f s, %f Hz", elapsedTime2 , 1.0 / elapsedTime2); */
    /* ROS_INFO("Step 3: %f s, %f Hz", elapsedTime3 , 1.0 / elapsedTime3); */
    /* ROS_INFO("Step 4: %f s, %f Hz", elapsedTime4 , 1.0 / elapsedTime4); */
    /* ROS_INFO("Step 5: %f s, %f Hz", elapsedTime5 , 1.0 / elapsedTime5); */
    /* ROS_INFO("Step 6: %f s, %f Hz", elapsedTime6 , 1.0 / elapsedTime6); */
    end         = std::clock();
    elapsedTimeO = double(end - begin_overall) / CLOCKS_PER_SEC;
    ROS_INFO("OVERALL: %f s, %f Hz", elapsedTimeO , 1.0 / elapsedTimeO);
    return output;
}
FftMethod::FftMethod(int i_frameSize, int i_samplePointSize, double max_px_speed_t, bool i_storeVideo, bool i_raw_enable, bool i_rot_corr_enable,
            bool i_tilt_corr_enable, std::string *videoPath, int videoFPS, std::string i_cl_file_name) {
  frameSize       = i_frameSize;
  samplePointSize = i_samplePointSize;
  max_px_speed_sq = pow(max_px_speed_t, 2);

  cl_file_name = i_cl_file_name;

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
      const cv::ocl::Device & dev = cv::ocl::Device::getDefault();

    usrc1.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    usrc2.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    window1.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    window2.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    FFT1.create(samplePointSize, samplePointSize, CV_32FC2,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    FFT2.create(samplePointSize, samplePointSize, CV_32FC2,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    MUL.create(samplePointSize, samplePointSize, CV_32FC2,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    PCR.create(frameSize, frameSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    C.create(samplePointSize, samplePointSize, CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);
    ML.create(1, dev.getDefault().maxWorkGroupSize()*sqNum*sqNum*(sizeof(uint)+sizeof(float)), CV_32FC1,cv::USAGE_ALLOCATE_DEVICE_MEMORY);

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
      /* shift = cv::phaseCorrelate(imPrevF(cv::Rect(xi, yi, samplePointSize, samplePointSize)), imCurrF(cv::Rect(xi, yi, samplePointSize, samplePointSize))); */
      shift = speeds[j+sqNum*i];
      shift_raw = shift;

      bool valid=true;
      if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) ||
          absd(shift.y) > ((double)samplePointSize / 2)) {
        ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j);
        valid=false;
      }

      /* if (raw_enable) { */
      /*   // push without correction first */
      /*   if (pow(shift.x, 2) + pow(shift.y, 2) > max_px_speed_sq || absd(shift.x) > ((double)samplePointSize / 2) || */
      /*       absd(shift.y) > ((double)samplePointSize / 2)) { */
      /*     ROS_WARN("[OpticFlow]: FFT - invalid correlation in window x %d y %d", i, j); */
      /*     speeds[j*sqNum+i] = cv::Point2d(nan(""), nan("")); */
      /*   } else { */
      /*     // ROS_WARN("[OpticFlow]: Hacks going on in raw...");  // hack for Gazebo Mobius */
      /*     // speeds.push_back(cv::Point2f(-shift.x,-shift.y)); */
      /*     speeds[j*sqNum+i] = cv::Point2d(shift.x, shift.y); */
      /*   } */
      /* } */


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
        speeds[j+i*sqNum]=cv::Point2d(nan(""), nan(""));
      } else {
        speeds[j+i*sqNum]=cv::Point2d(shift.x, shift.y);
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
    cv::imshow("mrs_optic_flow", imView);
    cv::waitKey(1);
  }

  if (storeVideo) {
    outputVideo << imView;
  }

  return speeds;
}
