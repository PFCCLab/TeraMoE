#pragma once

#include "configs.cuh"
#include "exception.cuh"
#include "utils.cuh"
#include <utility>

namespace deep_ep {
namespace internode {

struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

__host__ __device__ __forceinline__ int get_num_bytes_per_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                         num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                     sizeof(int4)));
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(int hidden_int4,
                                                                            int num_scales,
                                                                            int num_topk_idx,
                                                                            int num_topk_weights,
                                                                            int num_rdma_ranks,
                                                                            int num_rdma_recv_buffer_tokens,
                                                                            int num_channels) {
    return {(get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens *
             num_rdma_ranks * 2 * num_channels) /
                sizeof(int),
            (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_channels};
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(int hidden_int4,
                                                                           int num_scales,
                                                                           int num_topk_idx,
                                                                           int num_topk_weights,
                                                                           int num_rdma_ranks,
                                                                           int num_nvl_ranks,
                                                                           int num_nvl_recv_buffer_tokens,
                                                                           int num_channels,
                                                                           bool is_dispatch) {
    EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

    return {
        (num_nvl_recv_buffer_tokens * get_num_bytes_per_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_nvl_ranks *
         num_channels) /
            sizeof(int),
        num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_channels,
    };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int nvl_rank) {
    return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) : dst_rdma_rank;
}

constexpr __host__ __device__ int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <int kNumRanks,
          bool kMaybeWithBias,
          typename dtype_t,
          int kMaxNumRanks,
          bool kUseTMA,
          int kNumStages,
          int kNumTMALoadBytes = 0,
          typename GetAddrFn,
          typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank,
                             int head_idx,
                             int lane_id,
                             int hidden_int4,
                             int num_topk,
                             int4* combined_row,
                             float* combined_topk_weights,
                             const int4* bias_0_int4,
                             const int4* bias_1_int4,
                             int num_max_recv_tokens,
                             const GetAddrFn& get_addr_fn,
                             const ReceiveTWFn& recv_tw_fn,
                             uint8_t* smem_ptr,
                             uint32_t (&tma_phase)[kNumStages]) {
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // Broadcast current heads
    // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    #pragma unroll
    for (int i = 0; i < kNumRanks; ++i)
        if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
            topk_ranks[num_topk_ranks++] = i;
        }
    EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);
    EP_STATIC_ASSERT(not(kUseTMA and kMaybeWithBias), "TMA cannot be used by receiver warps");
    EP_STATIC_ASSERT(kNumStages == 2, "Only support 2 stages now");

    // Reduce data
    if constexpr (kUseTMA) {
        constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
        EP_DEVICE_ASSERT(hidden_int4 % 32 == 0);

        auto tma_load_buffer = [=](const int& i, const int& j) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + j * kNumTMALoadBytes);
        };
        auto tma_store_buffer = [=](const int& i) -> int4* {
            return reinterpret_cast<int4*>(smem_ptr + i * kNumTMABufferBytesPerStage + NUM_MAX_NVL_PEERS * kNumTMALoadBytes);
        };
        auto tma_mbarrier = [=](const int& i) -> uint64_t* {
            return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + (NUM_MAX_NVL_PEERS + 1) * kNumTMALoadBytes);
        };

        // Prefetch
        if (lane_id < num_topk_ranks)
            tma_load_1d(
                tma_load_buffer(0, lane_id), get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], 0), tma_mbarrier(0), kNumTMALoadBytes);
        mbarrier_arrive_and_expect_tx(tma_mbarrier(0), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
        __syncwarp();

        for (int shifted = 0, iter = 0; shifted < hidden_int4; shifted += 32, iter += 1) {
            const int stage_idx = iter % kNumStages;
            const int next_stage_idx = (iter + 1) % kNumStages;

            // Prefetch next stage
            if (shifted + 32 < hidden_int4) {
                if (lane_id < num_topk_ranks)
                    tma_load_1d(tma_load_buffer(next_stage_idx, lane_id),
                                get_addr_fn(topk_ranks[lane_id], slot_indices[lane_id], shifted + 32),
                                tma_mbarrier(next_stage_idx),
                                kNumTMALoadBytes);
                mbarrier_arrive_and_expect_tx(tma_mbarrier(next_stage_idx), lane_id < num_topk_ranks ? kNumTMALoadBytes : 0);
                __syncwarp();
            }

            mbarrier_wait(tma_mbarrier(stage_idx), tma_phase[stage_idx]);
            float values[kDtypePerInt4] = {0};
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(tma_load_buffer(stage_idx, j) + lane_id);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Wait shared memory to be released
            tma_store_wait<kNumStages - 1>();

            // Copy into shared and issue TMA
            auto out_dtypes = reinterpret_cast<dtype_t*>(tma_store_buffer(stage_idx) + lane_id);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            tma_store_fence();
            __syncwarp();

            if (elect_one_sync())
                tma_store_1d(tma_store_buffer(stage_idx), combined_row + shifted, kNumTMALoadBytes);
            __syncwarp();
        }

        // Flush all writes
        tma_store_wait<0>();
    } else {
        #pragma unroll
        for (int i = lane_id; i < hidden_int4; i += 32) {
            // Read bias
            int4 bias_0_value_int4, bias_1_value_int4;
            if constexpr (kMaybeWithBias) {
                bias_0_value_int4 = bias_0_int4 != nullptr ? ld_nc_global(bias_0_int4 + i) : make_int4(0, 0, 0, 0);
                bias_1_value_int4 = bias_1_int4 != nullptr ? ld_nc_global(bias_1_int4 + i) : make_int4(0, 0, 0, 0);
            }

            // Read buffers
            int4 recv_value_int4[kMaxNumRanks];
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j)
                recv_value_int4[j] = ld_nc_global(get_addr_fn(topk_ranks[j], slot_indices[j], i));

            // Reduce bias
            float values[kDtypePerInt4] = {0};
            if constexpr (kMaybeWithBias) {
                auto bias_0_values = reinterpret_cast<const dtype_t*>(&bias_0_value_int4);
                auto bias_1_values = reinterpret_cast<const dtype_t*>(&bias_1_value_int4);
                #pragma unroll
                for (int j = 0; j < kDtypePerInt4; ++j)
                    values[j] = static_cast<float>(bias_0_values[j]) + static_cast<float>(bias_1_values[j]);
            }

            // Reduce all-to-all results
            #pragma unroll
            for (int j = 0; j < num_topk_ranks; ++j) {
                auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
                #pragma unroll
                for (int k = 0; k < kDtypePerInt4; ++k)
                    values[k] += static_cast<float>(recv_value_dtypes[k]);
            }

            // Cast back to `dtype_t` and write
            int4 out_int4;
            auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
            #pragma unroll
            for (int j = 0; j < kDtypePerInt4; ++j)
                out_dtypes[j] = static_cast<dtype_t>(values[j]);
            st_na_global(combined_row + i, out_int4);
        }
    }

    // Reduce `topk_weights`
    if (lane_id < num_topk) {
        float value = 0;
        #pragma unroll
        for (int i = 0; i < num_topk_ranks; ++i)
            value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
        st_na_global(combined_topk_weights + lane_id, value);
    }

    // Return the minimum top-k rank
    return topk_ranks[0];
}

}  // namespace internode
}  // namespace deep_ep
