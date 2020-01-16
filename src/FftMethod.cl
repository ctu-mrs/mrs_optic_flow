// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
#define DEBUG false

#define SQRT_2 0.707106781188f
#define sin_120 0.866025403784f
#define fft5_2  0.559016994374f
#define fft5_3 -0.951056516295f
#define fft5_4 -1.538841768587f
#define fft5_5  0.363271264002f


__attribute__((always_inline))
CT mul_complex(CT a, CT b) {
    return (CT)(fma(a.x, b.x, -a.y * b.y), fma(a.x, b.y, a.y * b.x));
}

__attribute__((always_inline))
CT twiddle(CT a) {
    return (CT)(a.y, -a.x);
}

__attribute__((always_inline))
void butterfly2(CT a0, CT a1, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x & (block_size - 1);
    a1 = mul_complex(twiddles[k], a1);
    const int dst_ind = (x << 1) - k;

    smem[dst_ind] = a0 + a1;
    smem[dst_ind+block_size] = a0 - a1;
}

__attribute__((always_inline))
void butterfly4(CT a0, CT a1, CT a2, CT a3, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x & (block_size - 1);
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k + block_size], a2);
    a3 = mul_complex(twiddles[k + 2*block_size], a3);

    const int dst_ind = ((x - k) << 2) + k;

    CT b0 = a0 + a2;
    a2 = a0 - a2;
    CT b1 = a1 + a3;
    a3 = twiddle(a1 - a3);

    smem[dst_ind]                = b0 + b1;
    smem[dst_ind + block_size]   = a2 + a3;
    smem[dst_ind + 2*block_size] = b0 - b1;
    smem[dst_ind + 3*block_size] = a2 - a3;
}

__attribute__((always_inline))
void butterfly3(CT a0, CT a1, CT a2, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x % block_size;
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k+block_size], a2);
    const int dst_ind = ((x - k) * 3) + k;

    CT b1 = a1 + a2;
    a2 = twiddle(sin_120*(a1 - a2));
    CT b0 = a0 - (CT)(0.5f)*b1;

    smem[dst_ind] = a0 + b1;
    smem[dst_ind + block_size] = b0 + a2;
    smem[dst_ind + 2*block_size] = b0 - a2;
}

__attribute__((always_inline))
void butterfly5(CT a0, CT a1, CT a2, CT a3, CT a4, __local CT* smem, __global const CT* twiddles,
                const int x, const int block_size)
{
    const int k = x % block_size;
    a1 = mul_complex(twiddles[k], a1);
    a2 = mul_complex(twiddles[k + block_size], a2);
    a3 = mul_complex(twiddles[k+2*block_size], a3);
    a4 = mul_complex(twiddles[k+3*block_size], a4);

    const int dst_ind = ((x - k) * 5) + k;
    __local CT* dst = smem + dst_ind;

    CT b0, b1, b5;

    b1 = a1 + a4;
    a1 -= a4;

    a4 = a3 + a2;
    a3 -= a2;

    a2 = b1 + a4;
    b0 = a0 - (CT)0.25f * a2;

    b1 = fft5_2 * (b1 - a4);
    a4 = fft5_3 * (CT)(-a1.y - a3.y, a1.x + a3.x);
    b5 = (CT)(a4.x - fft5_5 * a1.y, a4.y + fft5_5 * a1.x);

    a4.x += fft5_4 * a3.y;
    a4.y -= fft5_4 * a3.x;

    a1 = b0 + b1;
    b0 -= b1;

    dst[0] = a0 + a2;
    dst[block_size] = a1 + a4;
    dst[2 * block_size] = b0 + b5;
    dst[3 * block_size] = b0 - b5;
    dst[4 * block_size] = a1 - a4;
}

__attribute__((always_inline))
void fft_radix2(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1;

    if (x < t)
    {
        a0 = smem[x];
        a1 = smem[x+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly2(a0, a1, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x1 + 2*t/3;
    CT a0, a1, a2, a3, a4, a5;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B4(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/4;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x1 < t/4)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
        a6 = smem[x4]; a7 = smem[x4+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/4)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
        butterfly2(a6, a7, smem, twiddles, x4, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix2_B5(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/5;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    const int x5 = x1 + 4*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    if (x1 < t/5)
    {
        a0 = smem[x1]; a1 = smem[x1+t];
        a2 = smem[x2]; a3 = smem[x2+t];
        a4 = smem[x3]; a5 = smem[x3+t];
        a6 = smem[x4]; a7 = smem[x4+t];
        a8 = smem[x5]; a9 = smem[x5+t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/5)
    {
        butterfly2(a0, a1, smem, twiddles, x1, block_size);
        butterfly2(a2, a3, smem, twiddles, x2, block_size);
        butterfly2(a4, a5, smem, twiddles, x3, block_size);
        butterfly2(a6, a7, smem, twiddles, x4, block_size);
        butterfly2(a8, a9, smem, twiddles, x5, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1, a2, a3;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x+t]; a2 = smem[x+2*t]; a3 = smem[x+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly4(a0, a1, a2, a3, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t];
        a4 = smem[x2]; a5 = smem[x2+t]; a6 = smem[x2+2*t]; a7 = smem[x2+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly4(a0, a1, a2, a3, smem, twiddles, x1, block_size);
        butterfly4(a4, a5, a6, a7, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix4_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x2 + t/3;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t];
        a4 = smem[x2]; a5 = smem[x2+t]; a6 = smem[x2+2*t]; a7 = smem[x2+3*t];
        a8 = smem[x3]; a9 = smem[x3+t]; a10 = smem[x3+2*t]; a11 = smem[x3+3*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly4(a0, a1, a2, a3, smem, twiddles, x1, block_size);
        butterfly4(a4, a5, a6, a7, smem, twiddles, x2, block_size);
        butterfly4(a8, a9, a10, a11, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix8(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    CT a0, a1, a2, a3, a4, a5, a6, a7;

    if (x < t)
    {
        int tw_ind = block_size / 8;

        a0 = smem[x];
        a1 = mul_complex(twiddles[k], smem[x + t]);
        a2 = mul_complex(twiddles[k + block_size],smem[x+2*t]);
        a3 = mul_complex(twiddles[k+2*block_size],smem[x+3*t]);
        a4 = mul_complex(twiddles[k+3*block_size],smem[x+4*t]);
        a5 = mul_complex(twiddles[k+4*block_size],smem[x+5*t]);
        a6 = mul_complex(twiddles[k+5*block_size],smem[x+6*t]);
        a7 = mul_complex(twiddles[k+6*block_size],smem[x+7*t]);

        CT b0, b1, b6, b7;

        b0 = a0 + a4;
        a4 = a0 - a4;
        b1 = a1 + a5;
        a5 = a1 - a5;
        a5 = (CT)(SQRT_2) * (CT)(a5.x + a5.y, -a5.x + a5.y);
        b6 = twiddle(a2 - a6);
        a2 = a2 + a6;
        b7 = a3 - a7;
        b7 = (CT)(SQRT_2) * (CT)(-b7.x + b7.y, -b7.x - b7.y);
        a3 = a3 + a7;

        a0 = b0 + a2;
        a2 = b0 - a2;
        a1 = b1 + a3;
        a3 = twiddle(b1 - a3);
        a6 = a4 - b6;
        a4 = a4 + b6;
        a7 = twiddle(a5 - b7);
        a5 = a5 + b7;

    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
    {
        const int dst_ind = ((x - k) << 3) + k;
        __local CT* dst = smem + dst_ind;

        dst[0] = a0 + a1;
        dst[block_size] = a4 + a5;
        dst[2 * block_size] = a2 + a3;
        dst[3 * block_size] = a6 + a7;
        dst[4 * block_size] = a0 - a1;
        dst[5 * block_size] = a4 - a5;
        dst[6 * block_size] = a2 - a3;
        dst[7 * block_size] = a6 - a7;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    CT a0, a1, a2;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x+t]; a2 = smem[x+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly3(a0, a1, a2, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/2;
    CT a0, a1, a2, a3, a4, a5;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B3(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1 + t/3;
    const int x3 = x2 + t/3;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8;

    if (x1 < t/3)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
        a6 = smem[x3]; a7 = smem[x3+t]; a8 = smem[x3+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/3)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
        butterfly3(a6, a7, a8, smem, twiddles, x3, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix3_B4(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int thread_block = t/4;
    const int x2 = x1 + thread_block;
    const int x3 = x1 + 2*thread_block;
    const int x4 = x1 + 3*thread_block;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11;

    if (x1 < t/4)
    {
        a0 = smem[x1]; a1 = smem[x1+t]; a2 = smem[x1+2*t];
        a3 = smem[x2]; a4 = smem[x2+t]; a5 = smem[x2+2*t];
        a6 = smem[x3]; a7 = smem[x3+t]; a8 = smem[x3+2*t];
        a9 = smem[x4]; a10 = smem[x4+t]; a11 = smem[x4+2*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/4)
    {
        butterfly3(a0, a1, a2, smem, twiddles, x1, block_size);
        butterfly3(a3, a4, a5, smem, twiddles, x2, block_size);
        butterfly3(a6, a7, a8, smem, twiddles, x3, block_size);
        butterfly3(a9, a10, a11, smem, twiddles, x4, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix5(__local CT* smem, __global const CT* twiddles, const int x, const int block_size, const int t)
{
    const int k = x % block_size;
    CT a0, a1, a2, a3, a4;

    if (x < t)
    {
        a0 = smem[x]; a1 = smem[x + t]; a2 = smem[x+2*t]; a3 = smem[x+3*t]; a4 = smem[x+4*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < t)
        butterfly5(a0, a1, a2, a3, a4, smem, twiddles, x, block_size);

    barrier(CLK_LOCAL_MEM_FENCE);
}

__attribute__((always_inline))
void fft_radix5_B2(__local CT* smem, __global const CT* twiddles, const int x1, const int block_size, const int t)
{
    const int x2 = x1+t/2;
    CT a0, a1, a2, a3, a4, a5, a6, a7, a8, a9;

    if (x1 < t/2)
    {
        a0 = smem[x1]; a1 = smem[x1 + t]; a2 = smem[x1+2*t]; a3 = smem[x1+3*t]; a4 = smem[x1+4*t];
        a5 = smem[x2]; a6 = smem[x2 + t]; a7 = smem[x2+2*t]; a8 = smem[x2+3*t]; a9 = smem[x2+4*t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x1 < t/2)
    {
        butterfly5(a0, a1, a2, a3, a4, smem, twiddles, x1, block_size);
        butterfly5(a5, a6, a7, a8, a9, smem, twiddles, x2, block_size);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

#ifdef DFT_SCALE
#define SCALE_VAL(x, scale) x*scale
#else
#define SCALE_VAL(x, scale) x
#endif

__attribute__((always_inline))
void fft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global const CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz,
                                   int Xvec,int Yvec, int samplePointSize,
                                   __local CT* smem)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    if (y < nz)
    {
        /* __global CT smem[LOCAL_SIZE]; */
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;
        FT scale = (FT) 1/(samplePointSize*samplePointSize);

#ifdef ROW_F_COMPLEX_INPUT
        __global const CT* src = (__global const CT*)(src_ptr + mad24(mad24(Yvec,samplePointSize,y), src_step, mad24(mad24(Xvec,samplePointSize,x), (int)sizeof(CT), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = src[i*block_size];
#else
        /* __global const FT* src = (__global const FT*)(src_ptr + mad24(y, src_step, mad24(x, (int)sizeof(FT), src_offset))); */
        __global const FT* src = (__global const FT*)(src_ptr + mad24(mad24(Yvec,samplePointSize,y), src_step, mad24(mad24(Xvec,samplePointSize,x), (int)sizeof(FT), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = (CT)(src[i*block_size], 0.f);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        /* if (x == 0) */
        /*   printf("smem[0]: %f", smem[0]); */

        RADIX_PROCESS;

#ifdef ROW_F_COMPLEX_OUTPUT
#ifdef NO_CONJUGATE
        const int cols = samplePointSize/2 + 1;
#else
        const int cols = samplePointSize;
#endif

        __global CT* dst = (__global CT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step,mad24(Xvec,samplePointSize*((int)sizeof(CT)),dst_offset) ));
        #pragma unroll
        for (int i=x; i<cols; i+=block_size)
            dst[i] = SCALE_VAL(smem[i], scale);

#else
        // pack row to CCS
        __global FT* smem_1cn = (__global FT*) smem;
        __global FT* dst = (__global FT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(Xvec,samplePointSize*((int)sizeof(FT)),dst_offset)));
        for (int i=x; i<samplePointSize-1; i+=block_size)
            dst[i+1] = SCALE_VAL(smem_1cn[i+2], scale);
        if (x == 0)
            dst[0] = SCALE_VAL(smem_1cn[0], scale);
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef ROW_F_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(Xvec,samplePointSize*((int)sizeof(CT)),dst_offset)));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(Xvec,samplePointSize*((int)sizeof(FT)),dst_offset)));
#endif
        #pragma unroll
        for (int i=x; i<samplePointSize; i+=block_size)
            dst[i] = 0.f;
    }
}

__attribute__((always_inline))
void fft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global const CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz,
                                   int Xvec, int Yvec, int samplePointSize,
                                   __local CT* smem)
{
    const int x = get_group_id(1);
    const int y = get_global_id(0);

    /* if (x == 0) printf("N y:%d\n",y); */

    if (x < nz)
    {
        /* __global CT smem[LOCAL_SIZE]; */
        __global const uchar* src = src_ptr + mad24(mad24(Yvec,samplePointSize,y), src_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), src_offset));
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;
        FT scale = 1.f/(samplePointSize*samplePointSize);
        /* if ((x == 0) && (y ==0)) */
        /*   printf("NEW scale = %f", scale); */

        /* #pragma unroll */
        /* for (int i=0; i<kercn; i++){ */
        /*   if (x< (nz-1)) */
        /*     *(__global FT*)(dstn + i*block_size*dst_step) = *(__global const FT*)(src + i*block_size*src_step); */
        /*     *(__global FT*)(dstn + i*block_size*dst_step + (int)sizeof(FT)) = *(__global const FT*)(src + i*block_size*src_step+ (int)sizeof(FT)); */
        /* } */

        /* return; */

        #pragma unroll
        for (int i=0; i<kercn; i++){
            smem[y+i*block_size] = *((__global const CT*)(src + i*block_size*src_step));
        }

        /* __global uchar* dstn = (__global uchar*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)sizeof(FT), dst_offset))); */
        /* barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE); */
        /* #pragma unroll */
        /* for (int i=0; i<kercn; i++){ */

        /*   if (x<(nz-1)){ */
        /*     *(__global FT*)(dstn + i*block_size*dst_step) = smem[y+i*block_size].x; */
        /*     *(__global FT*)(dstn + i*block_size*dst_step + (nz-1)*(int)sizeof(FT)) = smem[y+i*block_size].y; */
        /*     /1* *((__global FT*)(dstn+i*block_size*dst_step))                        = *((__global const FT*)(src + i*block_size*src_step)); *1/ */
        /*     /1* *((__global FT*)(dstn+i*block_size*dst_step+(nz-1)*(int)sizeof(FT))) = *((__global const FT*)(src + i*block_size*src_step+(int)sizeof(FT))); *1/ */
        /*     } */
        /* } */
        /* return; */

        /* barrier(CLK_LOCAL_MEM_FENCE); */
        /* return; */

        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        /* for (int i=0; i<kercn; i++){ */
        /*   if (x<(nz-1)){ */
        /*     *((__global FT*)(dstn+i*block_size*dst_step)) =smem[y+i*block_size].x; */
        /*     *((__global FT*)(dstn+i*block_size*dst_step+(nz-1)*(int)sizeof(FT))) = smem[y+i*block_size].y; */
        /*   } */
        /* } */
        /* barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE); */
        /* return; */

        RADIX_PROCESS;

        /* __global uchar* dstn = dst_ptr + mad24(x, (int)sizeof(FT)*2, mad24(y, dst_step, dst_offset - (int)sizeof(FT))); */
        /* for (int i=0; i<kercn; i++){ */
        /* if ((x == 0) && (y ==0) && (i == 1)){ */
        /*   printf("NEW val = %f\n",*((__global const FT*)(src + i*block_size*src_step))); */
        /*   printf("NEW src start %#020x\n",src); */
        /* } */
        /*         vstore2(SCALE_VAL(smem[y+i*block_size], scale), 0, (__global FT*) (dstn+i*block_size*dst_step)); */
        /* } */
        
        /* return; */

#ifdef COL_F_COMPLEX_OUTPUT
        /* if ((x == 0) && (y ==0) ){ */
        /*   printf("NEW C val = %f\n",smem[1]); */
        /* } */
        __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            *((__global CT*)(dst + i*block_size*dst_step)) = SCALE_VAL(smem[y + i*block_size], scale);
#else
        /* if ((x == 0) && (y ==0) ){ */
        /*   printf("NEW R val = %f\n",smem[1]); */
        /* } */
        if (x == 0)
        {
            // pack first column to CCS
            __local FT* smem_1cn = (__local FT*) smem;
            __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y)+1, dst_step, mad24(Xvec,samplePointSize*((int)sizeof(FT)),dst_offset));
            for (int i=y; i<samplePointSize-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + mad24(mul24(Yvec,samplePointSize),dst_step, mad24(Xvec,samplePointSize*((int)sizeof(FT)),dst_offset))) ) = SCALE_VAL(smem_1cn[0], scale);
        }
        else if (x == (samplePointSize+1)/2)
        {
            // pack last column to CCS (if needed)
            __local FT* smem_1cn = (__local FT*) smem;

            __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y)+1,  dst_step,  mad24(mad24(Xvec,samplePointSize,samplePointSize-1), ((int)sizeof(FT)), dst_offset));
            for (int i=y; i<samplePointSize-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + mad24(Yvec,  mul24(samplePointSize,dst_step),  mad24(Xvec, samplePointSize*(int)sizeof(FT), (samplePointSize-1)*((int)sizeof(FT))+dst_offset))) ) = SCALE_VAL(smem_1cn[0], scale);
        }
        else
        {
            __global uchar* dst = dst_ptr + mad24(mad24(Xvec,samplePointSize,x*2), (int)sizeof(FT), mad24(mad24(Yvec,samplePointSize,y), dst_step, dst_offset - (int)sizeof(FT)));
            #pragma unroll
            for (int i=y; i<(samplePointSize); i+=block_size, dst+=block_size*dst_step)
                vstore2(SCALE_VAL(smem[i], scale), 0, (__global FT*) dst);
                /* vstore2(SCALE_VAL((float2)(0.0f,0.0f), scale), 0, (__global FT*) dst); */
                /* *((__global FT*) dst) = 0.0f; */
                /* *((__global FT*) (dst+((int)sizeof(FT)))) = 0.0f; */
        }
#endif
    }
}

__attribute__((always_inline))
void ifft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                    __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                    __global CT* const twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz,
                                    int Xvec, int Yvec, int samplePointSize,
                                    __local CT* smem)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    const FT scale = (FT) 1.0f/(LOCAL_SIZE*LOCAL_SIZE);

    if (y < nz)
    {
      if ((y > (SEARCH_RADIUS)) && (y < (LOCAL_SIZE-SEARCH_RADIUS))){
        __global FT* dst = (__global FT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,(y<LOCAL_SIZE/2)?(y+LOCAL_SIZE/2):(y-LOCAL_SIZE/2)), dst_step, dst_offset));

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
          int shift = mad24(i,block_size,x);
          dst[mad24(Xvec,samplePointSize,shift)] = 0.0f;
        }

      }
     else{

      /* if ((Xvec % 2) == 0) */
      /*   return; */
        /* __global CT smem[LOCAL_SIZE]; */
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;

#if defined(ROW_I_COMPLEX_INPUT) && !defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(mad24(Yvec, samplePointSize, y), src_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            smem[x+i*block_size].x =  src[i*block_size].x;
            smem[x+i*block_size].y = -src[i*block_size].y;
        }
#else

    #if !defined(ROW_I_REAL_INPUT) && defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(mad24(Yvec, samplePointSize, y), src_step, mad24(mad24(Xvec,samplePointSize,2), (int)sizeof(CT), src_offset)));

        #pragma unroll
        for (int i=x; i<(LOCAL_SIZE-1)/2; i+=block_size)
        {
            smem[i+1].x = src[i].x;
            smem[i+1].y = -src[i].y;
            smem[LOCAL_SIZE-i-1] = src[i];
        }
    #else

        #pragma unroll
        for (int i=x; i<(LOCAL_SIZE-1)/2; i+=block_size)
        {
            CT src = vload2(0, (__global const FT*)(src_ptr + mad24(Yvec, samplePointSize, y), src_step, mad24(mad24(Xvec,samplePointSize,2*i+1), (int)sizeof(FT), src_offset)));

            smem[i+1].x = src.x;
            smem[i+1].y = -src.y;
            smem[LOCAL_SIZE-i-1] = src;
        }

    #endif

        if (x==0)
        {
            smem[0].x = *(__global const FT*)(src_ptr + mad24(mad24(Yvec, samplePointSize, y), src_step, src_offset));
            smem[0].y = 0.f;

            if(LOCAL_SIZE % 2 ==0)
            {
                #if !defined(ROW_I_REAL_INPUT) && defined(NO_CONJUGATE)
                smem[LOCAL_SIZE/2].x = src[LOCAL_SIZE/2-1].x;
                #else
                smem[LOCAL_SIZE/2].x = *(__global const FT*)(src_ptr + mad24(mad24(Yvec, samplePointSize, y), src_step, mad24(mad24(Xvec,samplePointSize,LOCAL_SIZE-1), (int)sizeof(FT), src_offset)));
                #endif
                smem[LOCAL_SIZE/2].y = 0.f;
            }
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
#ifdef ROW_I_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(mad24(Yvec, samplePointSize, y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            dst[i*block_size].x = SCALE_VAL(smem[x + i*block_size].x, scale);
            dst[i*block_size].y = SCALE_VAL(-smem[x + i*block_size].y, scale);
        }
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,(y<LOCAL_SIZE/2)?(y+LOCAL_SIZE/2):(y-LOCAL_SIZE/2)), dst_step, mad24(0, (int)(sizeof(FT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
          int shift = mad24(i,block_size,x);
          if ((shift > (SEARCH_RADIUS)) && (shift < (LOCAL_SIZE-SEARCH_RADIUS)))
            dst[mad24(Xvec,samplePointSize,(shift<(LOCAL_SIZE/2))?(shift+(LOCAL_SIZE/2)):(shift-(LOCAL_SIZE/2)))] = 0.0f;
          else
            dst[mad24(Xvec,samplePointSize,(shift<(LOCAL_SIZE/2))?(shift+(LOCAL_SIZE/2)):(shift-(LOCAL_SIZE/2)))] = SCALE_VAL(smem[shift].x, scale);
        }
     }
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef ROW_I_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(mad24(Yvec, samplePointSize, y), dst_step, dst_offset));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(mad24(Yvec, samplePointSize, y), dst_step, dst_offset));
#endif
        #pragma unroll
        for (int i=mad24(Xvec,samplePointSize,x); i<LOCAL_SIZE; i+=block_size)
            dst[i] = 0.f;
    }
}

__attribute__((always_inline))
void ifft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __global CT* const twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz,
                              int Xvec, int Yvec, int samplePointSize,
                              __local CT* smem)
{
    const int x = get_group_id(1);
    const int y = get_global_id(0);

#ifdef COL_I_COMPLEX_INPUT
    if (x < nz)
    {
        /* __global CT smem[LOCAL_SIZE]; */
        __global const uchar* src = src_ptr + mad24(mad24(Yvec,samplePointSize,y), src_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), src_offset));
        __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset));
        /* __global uchar* dst = dst_ptr + mad24(mad24(Yvec,dst_rows,y), dst_step, mad24(x, (int)(sizeof(CT)), mad24(Xvec,dst_cols,src_offset))); */
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            CT temp = *((__global const CT*)(src + i*block_size*src_step));
            smem[y+i*block_size].x =  temp.x;
            smem[y+i*block_size].y =  -temp.y;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
           __global CT* res = (__global CT*)(dst + i*block_size*dst_step);
            res[0].x = smem[y + i*block_size].x;
            res[0].y = -smem[y + i*block_size].y;
            /* if (x<5) */
            /* printf("X: %d, Y: %d i: %d | smem: %f:%f. \n", x,y, i, smem[y+i*block_size].x,-smem[y+i*block_size].y); */
        }
    }
#else
    if (x < nz)
    {
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;

        /* __global CT smem[LOCAL_SIZE]; */
#ifdef EVEN
        if (x!=0 && (x!=(nz-1)))
#else
        if (x!=0)
#endif
        {
            __global const uchar* src = src_ptr + mad24(mad24(Yvec,samplePointSize,y), src_step, mad24(mad24(Xvec,samplePointSize,2*x-1), (int)sizeof(FT), src_offset));
            #pragma unroll
            for (int i=0; i<kercn; i++)
            {
                CT temp = vload2(0, (__global const FT*)(src + i*block_size*src_step));
                smem[y+i*block_size].x = temp.x;
                smem[y+i*block_size].y = -temp.y;
            }
        }
        else
        {
            int ind = x==0 ? 0: 2*x-1;
            __global const FT* src = (__global const FT*)(src_ptr + mad24(1, src_step, mad24(mad24(Xvec,samplePointSize,ind), (int)sizeof(FT), src_offset)));
            int step = src_step/(int)sizeof(FT);

            #pragma unroll
            for (int i=y; i<(LOCAL_SIZE-1)/2; i+=block_size)
            {
                smem[i+1].x = src[2*i*step];
                smem[i+1].y = -src[(2*i+1)*step];

                smem[LOCAL_SIZE-i-1].x = src[2*i*step];;
                smem[LOCAL_SIZE-i-1].y = src[(2*i+1)*step];
            }
            if (y==0)
            {
                smem[0].x = *(__global const FT*)(src_ptr + mad24(mad24(Xvec,samplePointSize,ind), (int)sizeof(FT), src_offset));
                smem[0].y = 0.f;

                if(LOCAL_SIZE % 2 ==0)
                {
                    smem[LOCAL_SIZE/2].x = src[(LOCAL_SIZE-2)*step];
                    smem[LOCAL_SIZE/2].y = 0.f;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
        __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset));

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            __global CT* res = (__global CT*)(dst + i*block_size*dst_step);
            res[0].x =  smem[y + i*block_size].x;
            res[0].y = -smem[y + i*block_size].y;
        }
    }
    /* else */
    /* { */
    /*     const int block_size = LOCAL_SIZE/kercn; */
    /*     __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset)); */
    /*     #pragma unroll */
    /*     for (int i=0; i<kercn; i++) */
    /*     { */
    /*         __global CT* dst_c = (__global CT*)(dst + i*block_size*dst_step); */
    /*         *dst_c = (CT)(0.0f,0.0f); */
    /*     } */
    /* } */
#endif
}

inline float2 cmulf(float2 a, float2 b)
{
    return (float2)(mad(a.x, b.x, - a.y * b.y), mad(a.x, b.y, a.y * b.x));
}

inline float2 cmulnormf(float2 a, float2 b)
{
   float2 mul = (float2)(mad(a.x, b.x, - a.y * b.y), mad(a.x, b.y, a.y * b.x));
   float denom = rsqrt(mad(mul.x,mul.x,mul.y*mul.y + FLT_EPSILON));
     return (float2)(mul*denom);
   /* return mul; */
}

inline float2 conjf(float2 a)
{
    return (float2)(a.x, - a.y);
}

void mulAndNormalizeSpectrums(
                                   __global const uchar * src1_ptr, int src1_step, int src1_offset,
                                   __global const uchar * src2_ptr, int src2_step, int src2_offset,
                                   __global uchar * dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   int Xvec, int Yvec, int samplePointSize)
{
    const int y = get_global_id(0);
    const int x = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    const int fstep = (int)sizeof(FT);

    /* int x = get_group_id(0); */
    /* int y0 = get_global_id(0) * rowsPerWI; */

      /* printf("dst_cols: %d", dst_cols); */
    /* int threads = get_global_size(0); */
    /* int reps = (dst_cols-1/threads)+1; */
    if (x < LOCAL_SIZE)
    {
      /* int src1_index = mad24(y0, src1_step, mad24(x, (int)sizeof(float2), src1_offset)); */
      /* int src2_index = mad24(y0, src2_step, mad24(x, (int)sizeof(float2), src2_offset)); */
      /* int dst_index = mad24(y0, dst_step, mad24(x, (int)sizeof(float2), dst_offset)); */

      //---------------------------------------

#ifdef COL_F_REAL_OUTPUT

      __global const uchar* src1 = (__global const uchar*)(src1_ptr + mad24(mad24(Yvec,samplePointSize,y), src1_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(FT)), src1_offset)));
      __global const uchar* src2 = (__global const uchar*)(src2_ptr + mad24(mad24(Yvec,samplePointSize,y), src2_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(FT)), src2_offset)));
      __global uchar* dst = dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(FT)), dst_offset));

 
        if ( (x == 0) || (x == LOCAL_SIZE-1) )
        {
          for (int i=0; i<kercn; i++){
          /* for (int i=0; i<1; i++){ */
            if (((y+i*block_size) == 0) || ((y+i*block_size) == LOCAL_SIZE-1)){
      /* printf("MUL: lx: %d ly: %d gx: %d gy: %d\n",(int)(get_local_size(0)),(int)(get_local_size(1)),(int)(get_global_size(0)), (int)(get_global_size(1))); */
              /* printf("Y: %d\n",y+i*block_size); */
              /* printf("x: %d y: %d i: %d\n",x,y,i); */
              FT  temp1 = *(__global const FT*)(src1 + (i*block_size)*src1_step);
              FT  temp2 = *(__global const FT*)(src2 + (i*block_size)*src2_step);
              *(__global FT*)(dst + (i*block_size)*dst_step) = 1.0f/(temp1*temp2);
              /* *(__global FT*)(dst + (i*block_size)*dst_step) = (temp1*temp2); */
            }
            if (((y+i*block_size)%2) ==0)
              continue;
            if ((y+i*block_size) < (LOCAL_SIZE-2)){

              float2  temp1 = (float2)(*(__global const FT*)(src1 + (i*block_size)*src1_step), *(__global const FT*)(src1 + (1+i*block_size)*src1_step));
              float2  temp2 = (float2)(*(__global const FT*)(src2 + (i*block_size)*src2_step), *(__global const FT*)(src2 + (1+i*block_size)*src2_step));

#ifdef MUL_CONJ
              float2 v = cmulnormf(temp1, conjf(temp2));
#else
              float2 v = cmulnormf(temp1, temp2);
#endif
              *(__global FT*)(dst + (i*block_size)*dst_step) = v.x;
              *(__global FT*)(dst + (1+(i*block_size))*dst_step) = v.y;
              /* printf("!: %d, y: %d i: %d kercn: %d \n", y+i*block_size, y, i, kercn); */
            }
            /* else */
              /* printf("?: %d", y+i*block_size); */

          }
        }
        /* else if (x == (dst_cols+1)/2) */
        /* { */
        /*     // pack last column to CCS (if needed) */
        /*     __global FT* smem_1cn = (__global FT*) smem; */
        /*     __global uchar* dst = dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), mad24(y+1, dst_step, dst_offset)); */
        /*     for (int i=y; i<dst_rows-1; i+=block_size, dst+=dst_step*block_size) */
        /*         *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale); */
        /*     if (y == 0) */
        /*         *((__global FT*) (dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), dst_offset))) = SCALE_VAL(smem_1cn[0], scale); */
        /* } */
        else
        {
          if ((x%2) ==0)
            return;
          if (x>LOCAL_SIZE-3){
            return;
          }


          for (int i=0; i<kercn; i++){
            if (((y+i*block_size) < ((LOCAL_SIZE)))){

              float2  temp1 = (float2)(*(__global const FT*)(src1 + (i*block_size)*src1_step), *(__global const FT*)(src1 + fstep + (i*block_size)*src1_step));
              float2  temp2 = (float2)(*(__global const FT*)(src2 + (i*block_size)*src2_step), *(__global const FT*)(src2 + fstep + (i*block_size)*src2_step));

#ifdef MUL_CONJ
              float2 v = cmulnormf(temp1, conjf(temp2));
#else
              float2 v = cmulnormf(temp1, temp2);
#endif
              *(__global FT*)(dst + (i*block_size)*dst_step) = v.x;
              *(__global FT*)(dst + fstep + (i*block_size)*dst_step) = v.y;
              /* if ((i==0) && (y==3)) */
                /* printf("vx: %f, vy: %f\n",v.x,v.y); */
            }

          }
        }



#else

      //-----------------------------------------
      __global const CT* src1 = (__global const CT*)(src1_ptr + mad24(mad24(Yvec,samplePointSize,y), src1_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), src1_offset)));
      __global const CT* src2 = (__global const CT*)(src2_ptr + mad24(mad24(Yvec,samplePointSize,y), src2_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), src2_offset)));
      __global CT* dst = (__global CT*)(dst_ptr + mad24(mad24(Yvec,samplePointSize,y), dst_step, mad24(mad24(Xvec,samplePointSize,x), (int)(sizeof(CT)), dst_offset)));

#pragma unroll
      for (int i=0; i<kercn; i++){
        float2  temp1 = src1[i*block_size];
        float2  temp2 = src2[i*block_size];
#ifdef MUL_CONJ
        float2 v = cmulnormf(temp1, conjf(temp2));
#else
        float2 v = cmulnormf(temp1, temp2);
#endif
        dst[i*block_size] = v;

      }
#endif
    }
}
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.


#define MINMAX_STRUCT_ALIGNMENT 8

static inline int align(int pos)
{
    return (pos + (MINMAX_STRUCT_ALIGNMENT - 1)) & (~(MINMAX_STRUCT_ALIGNMENT - 1));
}

#define MIN_VAL (-FLT_MAX)

#define noconvert
#define INDEX_MAX UINT_MAX

#define MIN_ABS(a) fabs(a)
#define MIN_ABS2(a, b) fabs(a - b)
#define MIN(a, b) fmin(a, b)
#define MAX(a, b) fmax(a, b)

#if kercn != 3
#define loadpix(addr) *(__global const float *)(addr)
#define srcTSIZE (int)sizeof(float)
#else
#define loadpix(addr) vload3(0, (__global const float *)(addr))
#define srcTSIZE ((int)sizeof(float) * 3)
#endif

#undef srcTSIZE
#define srcTSIZE (int)sizeof(float)


#define CALC_MAX(p, inc) \
    if (maxval < *(__global const float *)(srcptr + src_index+p*srcTSIZE)) \
    { \
        maxval = *(__global const float *)(srcptr + src_index+p*srcTSIZE); \
        maxloc = id + inc; \
    }



void minmaxloc(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                        int total, int searchRows,int searchCols, int groupnum, __global uchar * dstptr, int index,
                        __local float* localmem_max, __local uint* localmem_maxloc,
                        int Xvec,int Yvec,int samplePointSize
                        )
{
    int lid = get_local_id(0);
    int gid = get_group_id(1);
    int  id = (gid*cols)+ lid *kercn;
    /* printf("id start: %d\n",id); */
    const int repetitions = ceil((float)(LOCAL_SIZE/(float)(kercn*get_local_size(0))));
  /* if ((lid == 0) && (gid == 60)) */

    srcptr += src_offset;

    float maxval = MIN_VAL;
    /* __global float localmem_max[WGS2_ALIGNED]; */
    /* __global uint localmem_maxloc[WGS2_ALIGNED]; */
    uint maxloc = INDEX_MAX;

    int src_index;

    dstT temp;

    for (int i=0; i<repetitions; i++){
    /* for (int grain = groupnum * WGS * kercn ; id < total; id += grain)  */
        {
          src_index = mad24(mad24(Yvec,samplePointSize,gid), src_step, mul24(mad24(Xvec,samplePointSize,mad24(lid,kercn,(kercn*WGS)*i)),srcTSIZE));

          /* temp = (dstT)(*(__global const float *)(srcptr + src_index),*(__global const float *)(srcptr + src_index+srcTSIZE),0,0,0,0,0,0); */

    /* if ((gid==60)&&(lid==0)) */
    /*   printf("central value: %f",loadpix(srcptr + mad24(60, src_step, mad24(1, srcTSIZE,0)))); */

#if kercn == 1
            if (maxval < temp)
            {
                maxval = temp;
                maxloc = id;
            }
#elif kercn >= 2
            CALC_MAX(0, 0)
            CALC_MAX(1, 1)
#if kercn >= 3
            CALC_MAX(2, 2)
#if kercn >= 4
            CALC_MAX(3, 3)
#if kercn >= 8
            CALC_MAX(4, 4)
            CALC_MAX(5, 5)
            CALC_MAX(6, 6)
            CALC_MAX(7, 7)
#if kercn == 16
            CALC_MAX(8, 8)
            CALC_MAX(9, 9)
            CALC_MAX(A, 10)
            CALC_MAX(B, 11)
            CALC_MAX(C, 12)
            CALC_MAX(D, 13)
            CALC_MAX(E, 14)
            CALC_MAX(F, 15)
#endif
#endif
#endif
#endif
#endif
        }
    }

    if (lid < WGS2_ALIGNED)
    {
        localmem_max[lid] = maxval;
        localmem_maxloc[lid] = maxloc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        int lid3 = lid - WGS2_ALIGNED;
              /* if (gid == 60) */
              /*   printf("before: LID3: %d V:%f, L:%d - maxloc: %d, maxval: %f\n",  lid3, localmem_max[lid3],localmem_maxloc[lid3], maxloc, maxval); */
        if (localmem_max[lid3] <= maxval)
        {
            if (localmem_max[lid3] == maxval)
                localmem_maxloc[lid3] = min(localmem_maxloc[lid3], maxloc);
            else
                localmem_maxloc[lid3] = maxloc,
            localmem_max[lid3] = maxval;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
              /* if ((lid == 0) && (gid == 60)) */
              /*   printf("lsize: %d, wgs2_al: %d\n", lsize, WGS2_ALIGNED); */
            int lid2 = lsize + lid;

            if (localmem_max[lid] <= localmem_max[lid2])
            {

                if (localmem_max[lid] == localmem_max[lid2])
                    localmem_maxloc[lid] = min(localmem_maxloc[lid2], localmem_maxloc[lid]);
                else
                    localmem_maxloc[lid] = localmem_maxloc[lid2],
                localmem_max[lid] = localmem_max[lid2];

              /* if (gid == 60) */
              /*   printf("after: V:%f, L:%d\n", localmem_max[lid],localmem_maxloc[lid]); */
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    /* if (gid == 60) */
    /*     printf("GID: %d: lid: %d maxloc: %d, maxval: %f\n", gid,lid,localmem_maxloc[lid],localmem_max[lid]); */
    int valstep = align((int)(sizeof(float))*groupnum);
    int fullstep = align(mad24(groupnum,(int)(sizeof(uint)),valstep));
    if (lid == 0)
    {
      int lposv = mad24(index, fullstep, mul24((int)(sizeof(float)),gid));
      int lposl = mad24(index, fullstep, mad24((int)(sizeof(uint)),gid,valstep));
        *(__global float *)(dstptr +lposv) = localmem_max[0];
        *(__global uint *)(dstptr + lposl) = localmem_maxloc[0];
    /* if (gid == 60) */
    /*     printf("maxval_GPU: %f\n", localmem_max[0]); */
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if ((lid == 0) && (gid == 0))
    {
      /* printf("valstep: %d, fullstep: %d\n", valstep, fullstep); */
        int lposvZ = mad24(index, fullstep, mul24((int)(sizeof(float)),0));
        int lposlZ = mad24(index, fullstep, mad24((int)(sizeof(uint)),0,valstep));
      /* printf("lposlZ A: %d\n", lposlZ); */
      for (int i=1; i<groupnum; i++){
        int lposv = mad24(index, fullstep, mul24((int)(sizeof(float)),i));
        int lposl = mad24(index, fullstep, mad24((int)(sizeof(uint)),i,valstep));

        if (*(__global float *)(dstptr + lposv) > *(__global float *)(dstptr + lposvZ)){
          *(__global float *)(dstptr + lposvZ) = *(__global float *)(dstptr + lposv);
          *(__global uint *)(dstptr + lposlZ) = *(__global uint *)(dstptr + lposl);
        }
      }
      if (DEBUG)
        printf("A: lposlZ: %d, value: %d\n", lposlZ, *(__global uint *)(dstptr + lposlZ));
    }
}

void refine(__global uchar * srcptr, int src_step, int src_offset, int cols, int rows,
    int groupnum, __global uchar * dstptr,int index, int Xvec,int Yvec, int samplePointSize, int radius)
{
    int lid = get_local_id(0);
    int gid = get_group_id(1);
  if ((gid == 0) && (lid == 0)) {
    int valstep = align((int)(sizeof(float))*groupnum);
    int fullstep = align(mad24(groupnum,(int)(sizeof(uint)),valstep));
    int lposvZ = mad24(index, fullstep, 0);
    int lposlZ = mad24(index, fullstep, valstep);
    int indexMax = *(__global uint *)(dstptr + lposlZ);
    if (DEBUG)
      printf("B: lposlZ: %d, indexMax: %d\n", lposlZ, indexMax);
    int xc = indexMax % cols;
    int yc = indexMax / cols;
    if (DEBUG)
      printf("xc: %d, yc: %d\n", xc,yc);
    float valMax = *(__global float *)(dstptr + lposvZ);
      /* printf("valMax: %f\n", valMax); */
    int xmin = mad24(Xvec,samplePointSize,(((xc-radius)>=0)?xc-radius:0));
    int xmax = mad24(Xvec,samplePointSize,(((xc+radius)<cols)?xc+radius:cols-1));
    int ymin = mad24(Yvec,samplePointSize,(((yc-radius)>=0)?yc-radius:0));
    int ymax = mad24(Yvec,samplePointSize,(((yc+radius)<rows)?yc+radius:rows-1));
    if (DEBUG)
      printf("xmin: %d, xmax: %d, ymin: %d, ymax: %d\n", xmin, xmax, ymin, ymax);

    float centroidX = 0.0f;
    float centroidY = 0.0f;
    float sumIntensity = FLT_EPSILON;

    __global float* dataIn = (__global float*)(srcptr + mad24(ymin,src_step, src_offset));
    for(int y = ymin; y <= ymax; y++) {
      for(int x = xmin; x <= xmax; x++) {
        if (dataIn[x]>0.0f){
          centroidX   += x*dataIn[x];
          centroidY   += y*dataIn[x];
          sumIntensity += dataIn[x];
        }
      /* printf("dataIn[x]: %f \n", dataIn[x]); */
      /* printf("&dataIn[x]: %d \n", &(dataIn[x]) - &(dataIn)); */
      }
      dataIn += src_step/((int)sizeof(FT));
    }
      /* printf("CX: %f, CY: %f, SI: %f \n", centroidX, centroidY, sumIntensity); */

    /* printf("X: %f, Y: %f\n", centroidX, centroidY); */

    centroidX /= sumIntensity;
    centroidY /= sumIntensity;

    if (DEBUG)
      printf("X: %f, Y: %f\n", centroidX, centroidY);

    int lposXZ = mad24(index, fullstep, mul24((int)(sizeof(float)),1));
    int lposYZ = mad24(index, fullstep, mul24((int)(sizeof(float)),2));

      /* printf("lposXZ: %d, lposYZ: %d\n", lposXZ, lposYZ); */
    *(__global float *)(dstptr + lposXZ) = centroidX-mad24(Xvec,samplePointSize,(float)(cols>>1));
    *(__global float *)(dstptr + lposYZ) = centroidY-mad24(Yvec,samplePointSize,(float)(rows>>1));
    /* *(__global float *)(dstptr + lposXZ) = (xc)-(cols>>1); */
    /* *(__global float *)(dstptr + lposYZ) = (yc)-(rows>>1); */


  }
}

__kernel void phaseCorrelateField(__global const uchar* src1_ptr, int src1_step, int src1_offset, int src1_rows, int src1_cols,
                                  __global const uchar* src2_ptr, int src2_step, int src2_offset, int src2_rows, int src2_cols,
                                  __global uchar* fftr1_ptr, int fftr1_step, int fftr1_offset, int fftr1_rows, int fftr1_cols,
                                  __global uchar* fftr2_ptr, int fftr2_step, int fftr2_offset, int fftr2_rows, int fftr2_cols,
                                  __global uchar* fft1_ptr, int fft1_step, int fft1_offset, int fft1_rows, int fft1_cols,
                                  __global uchar* fft2_ptr, int fft2_step, int fft2_offset, int fft2_rows, int fft2_cols,
                                  __global uchar* mul_ptr, int mul_step, int mul_offset, int mul_rows, int mul_cols,
                                  __global uchar* ifftc_ptr, int ifftc_step, int ifftc_offset, int ifftc_rows, int ifftc_cols,
                                  __global uchar* pcr_ptr, int pcr_step, int pcr_offset, int pcr_rows, int pcr_cols,
                                  __global uchar * dstptr,// int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                  /* __global uchar * l_smemptr, __global uchar * l_maxvalptr, __global uchar * l_maxlocptr, */
                                  __global const CT * twiddles_ptr, int twiddles_step, int twiddles_offset,
                                   const int t, int rowsPerWI, int Xfields, int Yfields, int samplePointSize){

  __local CT smem[LOCAL_SIZE];
  __local float localmem_max[WGS2_ALIGNED];
  __local uint localmem_maxloc[WGS2_ALIGNED];

  /* __global CT* smem = (__global CT*)(l_smemptr + fft1_cols*get_group_id(1)*((int)sizeof(CT))); */
  /* __global CT* localmem_max = (__global CT*)(l_maxvalptr + WGS2_ALIGNED*get_group_id(1)*((int)sizeof(float))); */
  /* __global CT* localmem_maxloc = (__global CT*)(l_maxlocptr + WGS2_ALIGNED*get_group_id(1)*((int)sizeof(uint))); */

  /* int i = Xfields-2; */
  /*   int j = Yfields-1; */
  /*   { */
  /*   { */
  for (int j=0; j<Yfields; j++){
    for (int i=0; i<Xfields; i++){
      int index = i+Xfields*j;

      fft_multi_radix_rows(
          src1_ptr, src1_step, src1_offset, src1_rows, src1_cols,
          fftr1_ptr, fftr1_step, fftr1_offset, fftr1_rows, fftr1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize,
          i,j, samplePointSize,
          smem
          );
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      fft_multi_radix_rows(
          src2_ptr, src2_step, src2_offset, src2_rows, src2_cols,
          fftr2_ptr, fftr2_step, fftr2_offset, fftr2_rows, fftr2_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize,
          i,j,samplePointSize,
          smem
          );
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

          /* src1_ptr, src1_step, src1_offset, src1_rows, src1_cols, */
          /* fftr1_ptr, fftr1_step, fftr1_offset, fftr1_rows, fftr1_cols, */
      fft_multi_radix_cols(
          fftr1_ptr, fftr1_step, fftr1_offset, fftr1_rows, fftr1_cols,
          fft1_ptr, fft1_step, fft1_offset, fft1_rows, fft1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize/2+1,
          i,j, samplePointSize,
          smem
          );

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      fft_multi_radix_cols(
          fftr2_ptr, fftr2_step, fftr2_offset, fftr2_rows, fftr2_cols,
          fft2_ptr, fft2_step, fft2_offset, fft2_rows, fft2_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize/2+1,
          i,j,samplePointSize,
          smem);

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      mulAndNormalizeSpectrums(
          fft1_ptr, fft1_step, fft1_offset,fft2_ptr, fft2_step, fft2_offset,
          mul_ptr, mul_step, mul_offset, mul_rows, mul_cols,
          i,j,samplePointSize);

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      ifft_multi_radix_cols(
          mul_ptr, mul_step, mul_offset, mul_rows, mul_cols,
          ifftc_ptr, ifftc_step, ifftc_offset, ifftc_rows, ifftc_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize/2+1,
          i,j,samplePointSize,
          smem
          );
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      ifft_multi_radix_rows(
          ifftc_ptr, ifftc_step, ifftc_offset, ifftc_rows, ifftc_cols,
          pcr_ptr, pcr_step, pcr_offset, pcr_rows, pcr_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, samplePointSize,
          i,j,samplePointSize,
          smem
          );
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      minmaxloc(pcr_ptr, pcr_step, pcr_offset, LOCAL_SIZE, LOCAL_SIZE*LOCAL_SIZE, fft1_rows, fft1_cols, get_num_groups(0)*get_num_groups(1), dstptr,index,localmem_max, localmem_maxloc,
          i,j,samplePointSize);

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

      refine(pcr_ptr, pcr_step, pcr_offset, LOCAL_SIZE, LOCAL_SIZE, get_num_groups(0)*get_num_groups(1), dstptr,index, i, j, samplePointSize,3);

    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
  }


}
  
