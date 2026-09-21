#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

// Per-row statistics for MQA logits: for every Q row, the kernel accumulates the sum and
// the sum of squares of the stored logits inside that row's valid KV range into a caller
// provided, zero-initialized FP32 `[num_rows, >= 2]` tensor (`row_stats[row * stride + 0/1]`).
//
// Pipeline:
//  - every math thread owns one KV column and keeps FP32 partials in registers across all KV
//    splits of a Q block (two predicated FP32 instructions per token per split);
//  - at the end of the Q block each math warp reduces its 32 columns with shuffles and lane 0
//    stores one `(sum, sqr_sum)` per row into a shared-memory slot indexed by the Q stage,
//    before the warp's usual `empty_q` release-arrive;
//  - the Q producer already acquire-waits on `empty_q` before it recycles a stage; with
//    statistics on it first sums the 8 math-warp partials of the block that used the stage and
//    issues one coalesced `red.global.add` per row, then refills the stage. The last stages
//    are drained after the producer loop. No barrier or wait is added to any warp.
// Different KV chunks and SMs (split-K schedules) combine through the relaxed reduction, so
// the caller must zero the tensor. Padding rows to 32 bytes (`stride = 8`) puts every row in
// its own L2 sector, which spreads same-row reductions over more L2 slices.

namespace deep_gemm::epilogue {

// Shared-memory region appended after the main MQA logits storage when statistics are enabled
template <uint32_t kNumQStages, uint32_t kNumMathWarps, uint32_t BLOCK_Q>
struct RowStatsSharedStorage {
    // `[Q stage][math warp][token] = (sum, sqr_sum)`
    float2 partials[kNumQStages][kNumMathWarps][BLOCK_Q];
};

// Bytes of the region, mirrored by the host-side shared memory size computation
template <uint32_t kNumQStages, uint32_t kNumMathWarps, uint32_t BLOCK_Q>
CUTLASS_HOST_DEVICE constexpr uint32_t get_row_stats_smem_size() {
    return static_cast<uint32_t>(sizeof(RowStatsSharedStorage<kNumQStages, kNumMathWarps, BLOCK_Q>));
}

// Math warp: reduce one row's partials across the warp and publish them for the stats warp
CUTLASS_DEVICE void warp_publish_row_stats(float2* partials, const uint32_t& lane_idx,
                                           float sum, float sqr_sum) {
    sum = math::warp_reduce_sum(sum);
    sqr_sum = math::warp_reduce_sum(sqr_sum);
    if (lane_idx == 0)
        *partials = make_float2(sum, sqr_sum);
}

} // namespace deep_gemm::epilogue
