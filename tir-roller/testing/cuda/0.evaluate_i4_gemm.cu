#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// nvcc -gencode arch=compute_80,code=sm_80 -lineinfo -O3 0.evaluate_i4_gemm.cu -o evaluate ; ./evaluate
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
extern "C" __global__ void __launch_bounds__(128) default_function_kernel(half* __restrict__ A, signed char* __restrict__ B, half* __restrict__ C) {

  half C_reindex_shared_warp[256];
  __shared__ half A_reindex_shared[16384];
  __shared__ signed char B_shared[4096];
  __shared__ half B_reindex_reindex_shared[4096];
  signed char B_local[4];
  half B_reindex_reindex_local[8];
  half A_reindex_shared_warp[64];
  half B_reindex_reindex_shared_warp[32];
  signed char B_local_1[4];
  half B_reindex_reindex_local_1[8];
  half A_reindex_shared_warp_1[64];
  half B_reindex_reindex_shared_warp_1[32];
  
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

  for (int ax1_0_3_init = 0; ax1_0_3_init < 8; ++ax1_0_3_init) {
    for (int ax2_0_3_init = 0; ax2_0_3_init < 4; ++ax2_0_3_init) {
      for (int i = 0; i < 8; ++i) {
C_reindex_shared_warp[((ax1_0_3_init * 32) + (ax2_0_3_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int ax0_ax1_ax2_fused_2 = 0; ax0_ax1_ax2_fused_2 < 8; ++ax0_ax1_ax2_fused_2) {

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_reindex_shared + (((((((int)threadIdx.y) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_ax1_ax2_fused_2 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 8)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(A_reindex_shared + (((((((int)threadIdx.y) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_ax1_ax2_fused_2 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 8))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(A + ((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (((int)threadIdx.z) * 1048576)) + (ax0_ax1_ax2_fused_2 * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + ((((int)threadIdx.x) & 3) * 8)))), "n"(16)
    );
  }
  }

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + (((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + (((((int)threadIdx.z) * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.z) * 524288)) + (((int)threadIdx.y) * 262144)) + ((((int)threadIdx.x) >> 4) * 131072)) + ((((int)threadIdx.x) & 15) * 16)))), "n"(16)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

  for (int ax3_0_0 = 0; ax3_0_0 < 511; ++ax3_0_0) {
    __syncthreads();
//     for (int ax0_ax1_ax2_fused_2_1 = 0; ax0_ax1_ax2_fused_2_1 < 8; ++ax0_ax1_ax2_fused_2_1) {

//   {
//         unsigned int addr;
// #if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
//     addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(A_reindex_shared + ((((((((ax3_0_0 + 1) & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (((int)threadIdx.z) * 2048)) + (ax0_ax1_ax2_fused_2_1 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 8)))));
// #else
//     __asm__ __volatile__(
//       "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
//       : "=r"(addr)
//       : "l"((void *)(A_reindex_shared + ((((((((ax3_0_0 + 1) & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (((int)threadIdx.z) * 2048)) + (ax0_ax1_ax2_fused_2_1 * 256)) + ((((int)threadIdx.x) >> 2) * 32)) + (((((int)threadIdx.x) & 3) ^ (((int)threadIdx.x) >> 3)) * 8))))
//     );
// #endif
//     __asm__ __volatile__(
//       #if TVM_ENABLE_L2_PREFETCH
//         "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
//       #else
//         "cp.async.cg.shared.global [%0], [%1], %2;"
//       #endif
//         :: "r"(addr), "l"((void*)(A + ((((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (((int)threadIdx.z) * 1048576)) + (ax0_ax1_ax2_fused_2_1 * 131072)) + ((((int)threadIdx.x) >> 2) * 16384)) + (ax3_0_0 * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32))), "n"(16)
//     );
//   }
//     }

  {
        unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)(B_shared + ((((((ax3_0_0 + 1) & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16)))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)(B_shared + ((((((ax3_0_0 + 1) & 1) * 2048) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.x) * 16))))
    );
#endif
    __asm__ __volatile__(
      #if TVM_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
      #else
        "cp.async.cg.shared.global [%0], [%1], %2;"
      #endif
        :: "r"(addr), "l"((void*)(B + (((((((((int)blockIdx.x) * 1048576) + (((int)threadIdx.z) * 524288)) + (((int)threadIdx.y) * 262144)) + ((((int)threadIdx.x) >> 4) * 131072)) + (ax3_0_0 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + 256))), "n"(16)
    );
  }
__asm__ __volatile__("cp.async.commit_group;");

__asm__ __volatile__("cp.async.wait_group 1;");

    __syncthreads();
    for (int ax0_ax1_ax2_ax3_0_fused_0 = 0; ax0_ax1_ax2_ax3_0_fused_0 < 4; ++ax0_ax1_ax2_ax3_0_fused_0) {
      *(int*)(B_local + 0) = *(int*)(B_shared + ((((((ax3_0_0 & 1) * 2048) + (ax0_ax1_ax2_ax3_0_fused_0 * 512)) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)));
      for (int ax0 = 0; ax0 < 8; ++ax0) {
        B_reindex_reindex_local[ax0] = ((half)((B_local[(ax0 >> 1)] >> ((signed char)((ax0 & 1) * 4))) & (signed char)15));
      }
      *(uint4*)(B_reindex_reindex_shared + ((((ax0_ax1_ax2_ax3_0_fused_0 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_reindex_reindex_local + 0);
    }
    __syncthreads();
    for (int ax3_0_1 = 0; ax3_0_1 < 2; ++ax3_0_1) {
      for (int ax0_0 = 0; ax0_0 < 8; ++ax0_0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_reindex_shared[((((((ax3_0_0 & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_0 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_reindex_shared[((((((ax3_0_0 & 1) * 8192) + (((int)threadIdx.y) * 4096)) + (ax0_0 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8))])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_warp + (ax0_0 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax0_0 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax0_0 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_warp + (ax0_0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (ax3_0_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax0_1 * 512)) + (ax3_0_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax1_0_3 = 0; ax1_0_3 < 8; ++ax1_0_3) {
        for (int ax2_0_3 = 0; ax2_0_3 < 4; ++ax2_0_3) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax2_0_3 * 8)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + (ax2_0_3 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3 * 32) + (ax2_0_3 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp + (ax1_0_3 * 8)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + ((ax2_0_3 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp + ((ax2_0_3 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3 * 32) + (ax2_0_3 * 8)) + 4)))[1]));
  }
        }
      }
    }
  }
__asm__ __volatile__("cp.async.wait_group 0;");

  __syncthreads();
  for (int ax0_ax1_ax2_ax3_0_fused_0_1 = 0; ax0_ax1_ax2_ax3_0_fused_0_1 < 4; ++ax0_ax1_ax2_ax3_0_fused_0_1) {
    *(int*)(B_local_1 + 0) = *(int*)(B_shared + (((((ax0_ax1_ax2_ax3_0_fused_0_1 * 512) + (((int)threadIdx.y) * 256)) + (((int)threadIdx.z) * 128)) + (((int)threadIdx.x) * 4)) + 2048));
    for (int ax0_2 = 0; ax0_2 < 8; ++ax0_2) {
      B_reindex_reindex_local_1[ax0_2] = ((half)((B_local_1[(ax0_2 >> 1)] >> ((signed char)((ax0_2 & 1) * 4))) & (signed char)15));
    }
    *(uint4*)(B_reindex_reindex_shared + ((((ax0_ax1_ax2_ax3_0_fused_0_1 * 1024) + (((int)threadIdx.y) * 512)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B_reindex_reindex_local_1 + 0);
  }
  __syncthreads();
  for (int ax3_0_1_1 = 0; ax3_0_1_1 < 2; ++ax3_0_1_1) {
    for (int ax0_0_1 = 0; ax0_0_1 < 8; ++ax0_0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_reindex_shared[(((((((int)threadIdx.y) * 4096) + (ax0_0_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 8192)])) + 0)));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_reindex_shared[(((((((int)threadIdx.y) * 4096) + (ax0_0_1 * 512)) + ((((int)threadIdx.x) & 15) * 32)) + ((((ax3_0_1_1 * 2) + (((int)threadIdx.x) >> 4)) ^ ((((int)threadIdx.x) & 7) >> 1)) * 8)) + 8192)])) + 0))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_reindex_shared_warp_1 + (ax0_0_1 * 8)))[0]), "=r"(((unsigned *)(A_reindex_shared_warp_1 + (ax0_0_1 * 8)))[1]), "=r"(((unsigned *)(A_reindex_shared_warp_1 + (ax0_0_1 * 8)))[2]), "=r"(((unsigned *)(A_reindex_shared_warp_1 + (ax0_0_1 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_3 = 0; ax0_3 < 4; ++ax0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax0_3 * 512)) + (ax3_0_1_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_reindex_reindex_shared[(((((int)threadIdx.z) * 2048) + (ax0_3 * 512)) + (ax3_0_1_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax0_3 * 8)))[0]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax0_3 * 8)))[1]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax0_3 * 8)))[2]), "=r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax0_3 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax1_0_3_1 = 0; ax1_0_3_1 < 8; ++ax1_0_3_1) {
      for (int ax2_0_3_1 = 0; ax2_0_3_1 < 4; ++ax2_0_3_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax2_0_3_1 * 8)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + (ax2_0_3_1 * 8)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[0]), "r"(((unsigned *)(C_reindex_shared_warp + ((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8))))[1]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
      "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
      :  "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "=r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1])
      : "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[0]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[1]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[2]), "r"(((unsigned *)(A_reindex_shared_warp_1 + (ax1_0_3_1 * 8)))[3]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + ((ax2_0_3_1 * 8) + 4)))[0]), "r"(((unsigned *)(B_reindex_reindex_shared_warp_1 + ((ax2_0_3_1 * 8) + 4)))[1]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[0]), "r"(((unsigned *)(C_reindex_shared_warp + (((ax1_0_3_1 * 32) + (ax2_0_3_1 * 8)) + 4)))[1]));
  }
      }
    }
  }
  for (int ax0_0_2 = 0; ax0_0_2 < 8; ++ax0_0_2) {
    for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
      __syncthreads();
      for (int local_id = 0; local_id < 8; ++local_id) {
        A_reindex_shared[(((((((((int)threadIdx.y) * 10240) + (((local_id & 3) >> 1) * 640)) + ((((int)threadIdx.x) >> 2) * 80)) + (((int)threadIdx.z) * 64)) + ((local_id >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (local_id & 1))] = C_reindex_shared_warp[(((ax0_0_2 * 32) + (ax1_0 * 8)) + local_id)];
      }
      __syncthreads();
      *(uint4*)(C + ((((((((((int)blockIdx.y) * 4194304) + (((int)threadIdx.y) * 2097152)) + (ax0_0_2 * 262144)) + ((((int)threadIdx.x) >> 1) * 16384)) + (((int)blockIdx.x) * 128)) + (((int)threadIdx.z) * 64)) + (ax1_0 * 16)) + ((((int)threadIdx.x) & 1) * 8))) = *(uint4*)(A_reindex_shared + ((((((int)threadIdx.y) * 10240) + ((((int)threadIdx.x) >> 1) * 80)) + (((int)threadIdx.z) * 64)) + ((((int)threadIdx.x) & 1) * 8)));
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
        default_function_kernel<<<dim3(128, 64, 1), dim3(32, 2, 2)>>>(d_A, d_B, d_C);
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