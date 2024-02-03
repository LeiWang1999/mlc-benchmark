#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// nvcc -gencode arch=compute_80,code=sm_80 -lineinfo -O3 1.evaluate_i4_gemm_with_ladder.cu -o evaluate ; ./evaluate
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#define TVM_ENABLE_L2_PREFETCH 1
__global__ void __launch_bounds__(128) Fused(half* __restrict__ input0, int8_t* __restrict__ input1, half* __restrict__ output0) {
  
  half mediate1_shared_warp[256];
  __shared__ half input0_shared[16384];
  __shared__ signed char input1_shared[4096];
  __shared__ half mediate0_shared[4096];
  signed char input1_shared_local[4];
  half mediate0_local[8];
  half input0_shared_warp[64];
  half mediate0_shared_warp[32];
  signed char input1_shared_local_1[4];
  half mediate0_local_1[8];
  half input0_shared_warp_1[64];
  half mediate0_shared_warp_1[32];

  const int MAX_BLOCK_N = 10;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int i_2_init = 0; i_2_init < 8; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 4; ++j_2_init) {
      for (int i = 0; i < 8; ++i) {
mediate1_shared_warp[((i_2_init * 32) + (j_2_init * 8)) + i] = 0.0;}
;
    }
  }
  #pragma unroll
  for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 8; ++ax0_ax1_ax2_ax3_0_fused_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(input0_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(input0_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(input0 + (((((((int)blockIdx.y) * 4194304) + (ax0_ax1_ax2_ax3_0_fused_0 * 524288)) + (((int)threadIdx.y) * 262144)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)))), "n"(16)
    );
  }
  }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(input1_shared + (((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(input1_shared + (((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(input1 + (((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.z) * 524288)) + (((int)threadIdx.y) * 262144)) + ((((int)threadIdx.x) >> 4) * 131072)) + ((((int)threadIdx.x) & 15) * 16)))), "n"(16)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int k_0 = 0; k_0 < 511; ++k_0) {
    __syncthreads();
    #pragma unroll
    for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 8; ++ax0_ax1_ax2_ax3_0_fused_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(input0_shared + (((((((k_0 + 1) & 1) * 8192) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(input0_shared + (((((((k_0 + 1) & 1) * 8192) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(input0 + (((((((((int)blockIdx.y) * 4194304) + (ax0_ax1_ax2_ax3_0_fused_0_1 * 524288)) + (((int)threadIdx.y) * 262144)) + (k_0 * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8)) + 512))), "n"(16)
    );
  }
    }

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(input1_shared + ((((((k_0 + 1) & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(input1_shared + ((((((k_0 + 1) & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(input1 + (((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.z) * 524288)) + (((int)threadIdx.y) * 262144)) + ((((int)threadIdx.x) >> 4) * 131072)) + (k_0 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 256))), "n"(16)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_ax1_ax2_ax3_0_fused_0_2 = 0; ax0_ax1_ax2_ax3_0_fused_0_2 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_2) {
      *(int*)(input1_shared_local + 0) = *(int*)(input1_shared + ((((((k_0 & 1) * 2048) + (ax0_ax1_ax2_ax3_0_fused_0_2 * 512)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)));
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        mediate0_local[ax0] = ((half)((input1_shared_local[(ax0 >> 1)] >> ((signed char)((ax0 & 1) * 4))) & (signed char)15));
      }
      *(uint4*)(mediate0_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_2 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(mediate0_local + 0);
    }
    __syncthreads();
    for (int k_1 = 0; k_1 < 2; ++k_1) {
      for (int ax0_1 = 0; ax0_1 < 8; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(input0_shared[(((((k_0 & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_1 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(input0_shared[(((((k_0 & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_1 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(input0_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(input0_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(input0_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(input0_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(mediate0_shared[(((((int)threadIdx.z) * 2048) + (ax0_2 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(mediate0_shared[(((((int)threadIdx.z) * 2048) + (ax0_2 * 512)) + (k_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(mediate0_shared_warp + (ax0_2 * 8)))[0]), "=r"(((unsigned *)(mediate0_shared_warp + (ax0_2 * 8)))[1]), "=r"(((unsigned *)(mediate0_shared_warp + (ax0_2 * 8)))[2]), "=r"(((unsigned *)(mediate0_shared_warp + (ax0_2 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(mediate1_shared_warp + ((i_2 * 32) + (j_2 * 8))))[0]), "=r"(((unsigned *)(mediate1_shared_warp + ((i_2 * 32) + (j_2 * 8))))[1])
      : "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(mediate0_shared_warp + (j_2 * 8)))[0]), "r"(((unsigned *)(mediate0_shared_warp + (j_2 * 8)))[1]), "r"(((unsigned *)(mediate1_shared_warp + ((i_2 * 32) + (j_2 * 8))))[0]), "r"(((unsigned *)(mediate1_shared_warp + ((i_2 * 32) + (j_2 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(mediate1_shared_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[0]), "=r"(((unsigned *)(mediate1_shared_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[1])
      : "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[0]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[1]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[2]), "r"(((unsigned *)(input0_shared_warp + (i_2 * 8)))[3]), "r"(((unsigned *)(mediate0_shared_warp + ((j_2 * 8) + 4)))[0]), "r"(((unsigned *)(mediate0_shared_warp + ((j_2 * 8) + 4)))[1]), "r"(((unsigned *)(mediate1_shared_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[0]), "r"(((unsigned *)(mediate1_shared_warp + (((i_2 * 32) + (j_2 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_ax1_ax2_ax3_0_fused_0_3 = 0; ax0_ax1_ax2_ax3_0_fused_0_3 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_3) {
    *(int*)(input1_shared_local_1 + 0) = *(int*)(input1_shared + (((((ax0_ax1_ax2_ax3_0_fused_0_3 * 512) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
    for (int ax0_3 = 0; ax0_3 < 8; ++ax0_3) {
      mediate0_local_1[ax0_3] = ((half)((input1_shared_local_1[(ax0_3 >> 1)] >> ((signed char)((ax0_3 & 1) * 4))) & (signed char)15));
    }
    *(uint4*)(mediate0_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_3 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(mediate0_local_1 + 0);
  }
  __syncthreads();
  for (int k_1_1 = 0; k_1_1 < 2; ++k_1_1) {
    for (int ax0_4 = 0; ax0_4 < 8; ++ax0_4) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(input0_shared[((((((int)threadIdx.y) * 4096) + (ax0_4 * 512)) + (k_1_1 * 256)) + 8192)])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(input0_shared[((((((int)threadIdx.y) * 4096) + (ax0_4 * 512)) + (k_1_1 * 256)) + 8192)])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(input0_shared_warp_1 + (ax0_4 * 8)))[0]), "=r"(((unsigned *)(input0_shared_warp_1 + (ax0_4 * 8)))[1]), "=r"(((unsigned *)(input0_shared_warp_1 + (ax0_4 * 8)))[2]), "=r"(((unsigned *)(input0_shared_warp_1 + (ax0_4 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_5 = 0; ax0_5 < 4; ++ax0_5) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(mediate0_shared[(((((int)threadIdx.z) * 2048) + (ax0_5 * 512)) + (k_1_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(mediate0_shared[(((((int)threadIdx.z) * 2048) + (ax0_5 * 512)) + (k_1_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(mediate0_shared_warp_1 + (ax0_5 * 8)))[0]), "=r"(((unsigned *)(mediate0_shared_warp_1 + (ax0_5 * 8)))[1]), "=r"(((unsigned *)(mediate0_shared_warp_1 + (ax0_5 * 8)))[2]), "=r"(((unsigned *)(mediate0_shared_warp_1 + (ax0_5 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int i_2_1 = 0; i_2_1 < 8; ++i_2_1) {
      for (int j_2_1 = 0; j_2_1 < 4; ++j_2_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(mediate1_shared_warp + ((i_2_1 * 32) + (j_2_1 * 8))))[0]), "=r"(((unsigned *)(mediate1_shared_warp + ((i_2_1 * 32) + (j_2_1 * 8))))[1])
      : "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[0]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[1]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[2]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[3]), "r"(((unsigned *)(mediate0_shared_warp_1 + (j_2_1 * 8)))[0]), "r"(((unsigned *)(mediate0_shared_warp_1 + (j_2_1 * 8)))[1]), "r"(((unsigned *)(mediate1_shared_warp + ((i_2_1 * 32) + (j_2_1 * 8))))[0]), "r"(((unsigned *)(mediate1_shared_warp + ((i_2_1 * 32) + (j_2_1 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(mediate1_shared_warp + (((i_2_1 * 32) + (j_2_1 * 8)) + 4)))[0]), "=r"(((unsigned *)(mediate1_shared_warp + (((i_2_1 * 32) + (j_2_1 * 8)) + 4)))[1])
      : "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[0]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[1]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[2]), "r"(((unsigned *)(input0_shared_warp_1 + (i_2_1 * 8)))[3]), "r"(((unsigned *)(mediate0_shared_warp_1 + ((j_2_1 * 8) + 4)))[0]), "r"(((unsigned *)(mediate0_shared_warp_1 + ((j_2_1 * 8) + 4)))[1]), "r"(((unsigned *)(mediate1_shared_warp + (((i_2_1 * 32) + (j_2_1 * 8)) + 4)))[0]), "r"(((unsigned *)(mediate1_shared_warp + (((i_2_1 * 32) + (j_2_1 * 8)) + 4)))[1]));
  }
      }
    }
  }
  for (int ax0_6 = 0; ax0_6 < 8; ++ax0_6) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      __syncthreads();
      for (int local_id = 0; local_id < 8; local_id+=2) {
*((uint *)&(&(input0_shared[((((int)threadIdx.y) * 10240) + (((int)threadIdx.z) * 1024))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))]) = *((uint *)&mediate1_shared_warp[((ax0_6 * 32) + (ax1 * 8)) + local_id]);
}
;
      __syncthreads();
      #pragma unroll
      for (int ax0_ax1_ax2_ax3_fused_0 = 0; ax0_ax1_ax2_ax3_fused_0 < 1; ++ax0_ax1_ax2_ax3_fused_0) {
        *(uint4*)(output0 + ((((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (ax0_6 * 262144)) + ((((int)threadIdx.x) >> 1) * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 64)) + (ax1 * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(input0_shared + (((((int)threadIdx.y) * 10240) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.x) * 8)));
      }
    }
  }
}

int main(){
    const int M = 16384;
    const int N = 16384;
    const int K = 16384;

    half* A = (half *)malloc(M * K * sizeof(half));
    int8_t* B = (int8_t *)malloc(K * N * sizeof(int8_t));
    half* C = (half *)malloc(M * N * sizeof(half));

    for(int i = 0; i < M * K; i++){
        A[i] = __float2half((i % 100)  / 1000.0);
    }

    for(int i = 0; i < K * N; i++){
        B[i] = (i % 100);
    }

    half* d_A, *d_C;
    int8_t* d_B;
    cudaMalloc((void**)&d_A, M * K * sizeof(half));
    cudaMalloc((void**)&d_B, K * N * sizeof(int8_t));
    cudaMalloc((void**)&d_C, M * N * sizeof(half));
    
    // copy data
    cudaMemcpy(d_A, A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice);
    const int iterations = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < iterations; i++){
        Fused<<<dim3(128, 64, 1), dim3(32, 2, 2)>>>(d_A, d_B, d_C);
    }
    // last error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds / iterations);

    cudaMemcpy(C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++){
        printf("%f ", __half2float(C[i]));
    }
}