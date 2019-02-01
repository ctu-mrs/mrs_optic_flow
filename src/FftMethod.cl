// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

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

__kernel void fft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    if (y < nz)
    {
        __local CT smem[LOCAL_SIZE];
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;
        FT scale = (FT) 1/(dst_cols*dst_rows);

#ifdef ROW_F_COMPLEX_INPUT
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = src[i*block_size];
#else
        __global const FT* src = (__global const FT*)(src_ptr + mad24(y, src_step, mad24(x, (int)sizeof(FT), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[x+i*block_size] = (CT)(src[i*block_size], 0.f);
#endif
        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

#ifdef ROW_F_COMPLEX_OUTPUT
        const int cols = dst_cols;

        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
        #pragma unroll
        for (int i=x; i<cols; i+=block_size)
            dst[i] = SCALE_VAL(smem[i], scale);
#ifdef ROW_F_REAL_INPUT
#ifdef ROW_F_COMPLEX_OUTPUT
#endif
#endif
#else
        // pack row to CCS
        __local FT* smem_1cn = (__local FT*) smem;
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
        for (int i=x; i<dst_cols-1; i+=block_size)
            dst[i+1] = SCALE_VAL(smem_1cn[i+2], scale);
        if (x == 0)
            dst[0] = SCALE_VAL(smem_1cn[0], scale);
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef ROW_F_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#endif
        #pragma unroll
        for (int i=x; i<dst_cols; i+=block_size)
            dst[i] = 0.f;
    }
}

__kernel void fft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                   __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                   __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_group_id(1);
    const int y = get_global_id(0);

    if (x < nz)
    {
        __local CT smem[LOCAL_SIZE];
        __global const uchar* src = src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset));
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;
        FT scale = 1.f/(dst_rows*dst_cols);

        #pragma unroll
        for (int i=0; i<kercn; i++)
            smem[y+i*block_size] = *((__global const CT*)(src + i*block_size*src_step));

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

#ifdef COL_F_COMPLEX_OUTPUT
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));
        #pragma unroll
        for (int i=0; i<kercn; i++)
            *((__global CT*)(dst + i*block_size*dst_step)) = SCALE_VAL(smem[y + i*block_size], scale);
#else
        if (x == 0)
        {
            // pack first column to CCS
            __local FT* smem_1cn = (__local FT*) smem;
            __global uchar* dst = dst_ptr + mad24(y+1, dst_step, dst_offset);
            for (int i=y; i<dst_rows-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + dst_offset)) = SCALE_VAL(smem_1cn[0], scale);
        }
        else if (x == (dst_cols+1)/2)
        {
            // pack last column to CCS (if needed)
            __local FT* smem_1cn = (__local FT*) smem;
            __global uchar* dst = dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), mad24(y+1, dst_step, dst_offset));
            for (int i=y; i<dst_rows-1; i+=block_size, dst+=dst_step*block_size)
                *((__global FT*) dst) = SCALE_VAL(smem_1cn[i+2], scale);
            if (y == 0)
                *((__global FT*) (dst_ptr + mad24(dst_cols-1, (int)sizeof(FT), dst_offset))) = SCALE_VAL(smem_1cn[0], scale);
        }
        else
        {
            __global uchar* dst = dst_ptr + mad24(x, (int)sizeof(FT)*2, mad24(y, dst_step, dst_offset - (int)sizeof(FT)));
            #pragma unroll
            for (int i=y; i<dst_rows; i+=block_size, dst+=block_size*dst_step)
                vstore2(SCALE_VAL(smem[i], scale), 0, (__global FT*) dst);
        }
#endif
    }
}

__kernel void ifft_multi_radix_rows(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                                    __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                    __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);
    const int block_size = LOCAL_SIZE/kercn;
    const FT scale = (FT) 1/(dst_cols*dst_rows);

    if (y < nz)
    {
        __local CT smem[LOCAL_SIZE];
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = x;

#if defined(ROW_I_COMPLEX_INPUT) && !defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            smem[x+i*block_size].x =  src[i*block_size].x;
            smem[x+i*block_size].y = -src[i*block_size].y;
        }
#else

    #if !defined(ROW_I_REAL_INPUT) && defined(NO_CONJUGATE)
        __global const CT* src = (__global const CT*)(src_ptr + mad24(y, src_step, mad24(2, (int)sizeof(FT), src_offset)));

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
            CT src = vload2(0, (__global const FT*)(src_ptr + mad24(y, src_step, mad24(2*i+1, (int)sizeof(FT), src_offset))));

            smem[i+1].x = src.x;
            smem[i+1].y = -src.y;
            smem[LOCAL_SIZE-i-1] = src;
        }

    #endif

        if (x==0)
        {
            smem[0].x = *(__global const FT*)(src_ptr + mad24(y, src_step, src_offset));
            smem[0].y = 0.f;

            if(LOCAL_SIZE % 2 ==0)
            {
                #if !defined(ROW_I_REAL_INPUT) && defined(NO_CONJUGATE)
                smem[LOCAL_SIZE/2].x = src[LOCAL_SIZE/2-1].x;
                #else
                smem[LOCAL_SIZE/2].x = *(__global const FT*)(src_ptr + mad24(y, src_step, mad24(LOCAL_SIZE-1, (int)sizeof(FT), src_offset)));
                #endif
                smem[LOCAL_SIZE/2].y = 0.f;
            }
        }
#endif

        barrier(CLK_LOCAL_MEM_FENCE);

        RADIX_PROCESS;

        // copy data to dst
#ifdef ROW_I_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            dst[i*block_size].x = SCALE_VAL(smem[x + i*block_size].x, scale);
            dst[i*block_size].y = SCALE_VAL(-smem[x + i*block_size].y, scale);
        }
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(FT)), dst_offset)));
        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            dst[i*block_size] = SCALE_VAL(smem[x + i*block_size].x, scale);
        }
#endif
    }
    else
    {
        // fill with zero other rows
#ifdef ROW_I_COMPLEX_OUTPUT
        __global CT* dst = (__global CT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#else
        __global FT* dst = (__global FT*)(dst_ptr + mad24(y, dst_step, dst_offset));
#endif
        #pragma unroll
        for (int i=x; i<dst_cols; i+=block_size)
            dst[i] = 0.f;
    }
}

__kernel void ifft_multi_radix_cols(__global const uchar* src_ptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar* dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __global CT* twiddles_ptr, int twiddles_step, int twiddles_offset, const int t, const int nz)
{
    const int x = get_group_id(1);
    const int y = get_global_id(0);

#ifdef COL_I_COMPLEX_INPUT
    if (x < nz)
    {
        __local CT smem[LOCAL_SIZE];
        __global const uchar* src = src_ptr + mad24(y, src_step, mad24(x, (int)(sizeof(CT)), src_offset));
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));
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
        }
    }
#else
    if (x < nz)
    {
        __global const CT* twiddles = (__global const CT*)(twiddles_ptr + twiddles_offset);
        const int ind = y;
        const int block_size = LOCAL_SIZE/kercn;

        __local CT smem[LOCAL_SIZE];
#ifdef EVEN
        if (x!=0 && (x!=(nz-1)))
#else
        if (x!=0)
#endif
        {
            __global const uchar* src = src_ptr + mad24(y, src_step, mad24(2*x-1, (int)sizeof(FT), src_offset));
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
            __global const FT* src = (__global const FT*)(src_ptr + mad24(1, src_step, mad24(ind, (int)sizeof(FT), src_offset)));
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
                smem[0].x = *(__global const FT*)(src_ptr + mad24(ind, (int)sizeof(FT), src_offset));
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
        __global uchar* dst = dst_ptr + mad24(y, dst_step, mad24(x, (int)(sizeof(CT)), dst_offset));

        #pragma unroll
        for (int i=0; i<kercn; i++)
        {
            __global CT* res = (__global CT*)(dst + i*block_size*dst_step);
            res[0].x =  smem[y + i*block_size].x;
            res[0].y = -smem[y + i*block_size].y;
        }
    }
#endif
}

inline float2 cmulf(float2 a, float2 b)
{
    return (float2)(mad(a.x, b.x, - a.y * b.y), mad(a.x, b.y, a.y * b.x));
}

inline float2 cmulnormf(float2 a, float2 b)
{
   float2 mul = (float2)(mad(a.x, b.x, - a.y * b.y), mad(a.x, b.y, a.y * b.x));
   float denom = rsqrt(mad(mul.x,mul.x,mul.y*mul.y));
   return (float2)(mul*denom);
}

inline float2 conjf(float2 a)
{
    return (float2)(a.x, - a.y);
}

__kernel void mulAndNormalizeSpectrums(__global const uchar * src1ptr, int src1_step, int src1_offset,
                                   __global const uchar * src2ptr, int src2_step, int src2_offset,
                                   __global uchar * dstptr, int dst_step, int dst_offset,
                                   int dst_rows, int dst_cols, int rowsPerWI)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        int src1_index = mad24(y0, src1_step, mad24(x, (int)sizeof(float2), src1_offset));
        int src2_index = mad24(y0, src2_step, mad24(x, (int)sizeof(float2), src2_offset));
        int dst_index = mad24(y0, dst_step, mad24(x, (int)sizeof(float2), dst_offset));

        for (int y = y0, y1 = min(dst_rows, y0 + rowsPerWI); y < y1; ++y,
            src1_index += src1_step, src2_index += src2_step, dst_index += dst_step)
        {
            float2 src0 = *(__global const float2 *)(src1ptr + src1_index);
            float2 src1 = *(__global const float2 *)(src2ptr + src2_index);
            __global float2 * dst = (__global float2 *)(dstptr + dst_index);

#ifdef CONJ
            float2 v = cmulnormf(src0, conjf(src1));
#else
            float2 v = cmulnormf(src0, src1);
#endif
            dst[0] = v;
        }
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

#define CALC_MIN(p, inc)

#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
#define CALC_MAX(p, inc) \
    if (maxval < temp.p) \
    { \
        maxval = temp.p; \
        maxloc = id + inc; \
    }
#else
#define CALC_MAX(p, inc) \
    maxval = MAX(maxval, temp.p);
#endif
#else
#define CALC_MAX(p, inc)
#endif

#define CALC_MAX2(p)

#define CALC_P(p, inc) \
    CALC_MIN(p, inc) \
    CALC_MAX(p, inc) \
    CALC_MAX2(p)

__kernel void minmaxloc(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                        int total, int groupnum, __global uchar * dstptr
                        )
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0)
    * kercn;

    srcptr += src_offset;

#ifdef NEED_MAXVAL
    float maxval = MIN_VAL;
    __local float localmem_max[WGS2_ALIGNED];
#ifdef NEED_MAXLOC
    __local uint localmem_maxloc[WGS2_ALIGNED];
    uint maxloc = INDEX_MAX;
#endif
#endif

    int src_index;

    float temp;

    for (int grain = groupnum * WGS
        * kercn
        ; id < total; id += grain)
    {
        {
          src_index = mad24(id / cols, src_step, mul24(id % cols, srcTSIZE));
          temp = (loadpix(srcptr + src_index));


#if kercn == 1
#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
            if (maxval < temp)
            {
                maxval = temp;
                maxloc = id;
            }
#else
            maxval = MAX(maxval, temp);
#endif
#endif
#elif kercn >= 2
            CALC_P(s0, 0)
            CALC_P(s1, 1)
#if kercn >= 3
            CALC_P(s2, 2)
#if kercn >= 4
            CALC_P(s3, 3)
#if kercn >= 8
            CALC_P(s4, 4)
            CALC_P(s5, 5)
            CALC_P(s6, 6)
            CALC_P(s7, 7)
#if kercn == 16
            CALC_P(s8, 8)
            CALC_P(s9, 9)
            CALC_P(sA, 10)
            CALC_P(sB, 11)
            CALC_P(sC, 12)
            CALC_P(sD, 13)
            CALC_P(sE, 14)
            CALC_P(sF, 15)
#endif
#endif
#endif
#endif
#endif
        }
    }

    if (lid < WGS2_ALIGNED)
    {
#ifdef NEED_MAXVAL
        localmem_max[lid] = maxval;
#endif
#ifdef NEED_MAXLOC
        localmem_maxloc[lid] = maxloc;
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        int lid3 = lid - WGS2_ALIGNED;
#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
        if (localmem_max[lid3] <= maxval)
        {
            if (localmem_max[lid3] == maxval)
                localmem_maxloc[lid3] = min(localmem_maxloc[lid3], maxloc);
            else
                localmem_maxloc[lid3] = maxloc,
            localmem_max[lid3] = maxval;
        }
#else
        localmem_max[lid3] = MAX(localmem_max[lid3], maxval);
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;

#ifdef NEED_MAXVAL
#ifdef NEED_MAXLOC
            if (localmem_max[lid] <= localmem_max[lid2])
            {
                if (localmem_max[lid] == localmem_max[lid2])
                    localmem_maxloc[lid] = min(localmem_maxloc[lid2], localmem_maxloc[lid]);
                else
                    localmem_maxloc[lid] = localmem_maxloc[lid2],
                localmem_max[lid] = localmem_max[lid2];
            }
#else
            localmem_max[lid] = MAX(localmem_max[lid], localmem_max[lid2]);
#endif
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        int pos = 0;
#ifdef NEED_MAXVAL
        *(__global float *)(dstptr + mad24(gid, (int)sizeof(float), pos)) = localmem_max[0];
        pos = mad24(groupnum, (int)sizeof(float), pos);
        pos = align(pos);
#endif
#ifdef NEED_MAXLOC
        *(__global uint *)(dstptr + mad24(gid, (int)sizeof(uint), pos)) = localmem_maxloc[0];
#endif
    }
}

__kernel void phaseCorrelateField(__global const uchar* src1_ptr, int src1_step, int src1_offset, int src1_rows, int src1_cols,
                                  __global const uchar* src2_ptr, int src2_step, int src2_offset, int src2_rows, int src2_cols,
                                  __global uchar* fft1_ptr, int fft1_step, int fft1_offset, int fft1_rows, int fft1_cols,
                                  __global uchar* fft2_ptr, int fft2_step, int fft2_offset, int fft2_rows, int fft2_cols,
                                  __global uchar* mul_ptr, int mul_step, int mul_offset, int mul_rows, int mul_cols,
                                  __global uchar* pcr_ptr, int pcr_step, int pcr_offset, int pcr_rows, int pcr_cols,
                                  __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                  __global CT * twiddles_ptr, int twiddles_step, int twiddles_offset,
                                   const int t, int rowsPerWI, int Xfields, int Yfields){

  /* for (int j=0; j<Yfields; j++){ */
  /*   for (int i=0; i<Yfields; i++){ */


      fft_multi_radix_rows(src1_ptr, src1_step, src1_offset, src1_rows, src1_cols,
          fft1_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_rows
          );
      fft_multi_radix_cols(fft1_ptr, src2_step, src2_offset, src2_rows, src2_cols,
          fft1_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_cols
          );

      fft_multi_radix_rows(src1_ptr, src1_step, src1_offset, src1_rows, src1_cols,
          fft2_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_rows
          );
      fft_multi_radix_cols(fft2_ptr, src2_step, src2_offset, src2_rows, src2_cols,
          fft2_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_cols
          );

      mulAndNormalizeSpectrums(fft1_ptr, src1_step, src1_offset,fft2_ptr, src2_step, src2_offset,
          mul_ptr, mul_step, mul_offset, mul_rows, mul_cols, rowsPerWI);

      ifft_multi_radix_rows(mul_ptr, src1_step, src1_offset, src1_rows, src1_cols,
          fft1_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_rows
          );
      ifft_multi_radix_cols(fft1_ptr, src2_step, src2_offset, src2_rows, src2_cols,
          pcr_ptr, src1_cols, 0, src1_rows, src1_cols,
          twiddles_ptr, twiddles_step, twiddles_offset, t, src1_cols
          );

      FFTSHIFT HERE

      minmaxloc(pcr_ptr, pcr_step, pcr_step, pcr_cols, pcr_cols*pcr_cols, get_num_groups(0)*get_num_groups(1), dstptr);
    /* } } */


}
  
