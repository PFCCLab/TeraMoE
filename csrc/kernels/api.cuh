#pragma once

#include <cstdint>
#include <vector>

#include "configs.cuh"

namespace deep_ep {

// Intranode runtime
namespace intranode {

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream);

}  // namespace intranode

// Internode runtime
namespace internode {

std::vector<uint8_t> get_unique_id();

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode);

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

void barrier();

void finalize();

}  // namespace internode

// Layout kernels
namespace layout {

void get_dispatch_layout(const topk_idx_t* topk_idx,
                         int* num_tokens_per_rank,
                         int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert,
                         bool* is_token_in_rank,
                         int num_tokens,
                         int num_topk,
                         int num_ranks,
                         int num_experts,
                         cudaStream_t stream);

}  // namespace layout

// Intranode kernels
namespace intranode {

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     int num_tokens,
                     const bool* is_token_in_rank,
                     int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy,
                     int num_memset_int,
                     int expert_alignment,
                     void** buffer_ptrs,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int num_sms);

void cached_notify_dispatch(const int* rank_prefix_matrix,
                            int num_memset_int,
                            void** buffer_ptrs,
                            int** barrier_signal_ptrs,
                            int rank,
                            int num_ranks,
                            cudaStream_t stream);

void dispatch(void* recv_x,
              float* recv_x_scales,
              int* recv_src_idx,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              int* recv_channel_offset,
              int* send_head,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              const bool* is_token_in_rank,
              const int* channel_prefix_matrix,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_topk,
              int num_experts,
              int num_scales,
              int scale_token_stride,
              int scale_hidden_stride,
              void** buffer_ptrs,
              int rank,
              int num_ranks,
              cudaStream_t stream,
              int num_sms,
              int num_max_send_tokens,
              int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs,
                           int* send_head,
                           int num_channels,
                           int num_recv_tokens,
                           int num_memset_int,
                           int** barrier_signal_ptrs,
                           int rank,
                           int num_ranks,
                           cudaStream_t stream);

void combine(cudaDataType_t type,
             void* recv_x,
             float* recv_topk_weights,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* src_idx,
             const int* rank_prefix_matrix,
             const int* channel_prefix_matrix,
             int* send_head,
             int num_tokens,
             int num_recv_tokens,
             int hidden,
             int num_topk,
             void** buffer_ptrs,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_sms,
             int num_max_send_tokens,
             int num_recv_buffer_tokens);

}  // namespace intranode

// Internode kernels
namespace internode {

int get_source_meta_bytes();

void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode);

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode);

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode);


void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode);

}  // namespace internode

// Internode low-latency kernels
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer,
                              int* sync_buffer,
                              cudaStream_t stream);

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases);

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy);

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, cudaStream_t stream);

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, cudaStream_t stream);

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream);

}  // namespace internode_ll

namespace megakernel {

enum class ComputeDType {
    kBF16 = 0,
    kFP8E4M3 = 1,
};

}  // namespace megakernel

}  // namespace deep_ep

namespace teramoe {

using ComputeDType = ::deep_ep::megakernel::ComputeDType;
struct TeraMoEState;
struct MegaKernelBackwardState;
struct MegaKernelBackwardHostContext;

TeraMoEState* allocate_teramoe_fused_state(
    const int4* x,
    const uint32_t* x_scales,
    const ::deep_ep::topk_idx_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    void** combine_buffer_ptrs,
    int num_tokens,
    int hidden_dim,
    int hidden_int4,
    int intermediate_dim,
    int num_scales,
    int num_topk,
    int num_experts,
    int num_local_experts,
    int num_ranks,
    int rank,
    int scale_token_stride,
    int scale_hidden_stride,
    int dispatch_num_max_rdma_chunked_send_tokens,
    int dispatch_num_max_rdma_chunked_recv_tokens,
    int dispatch_num_max_nvl_chunked_send_tokens,
    int dispatch_num_max_nvl_chunked_recv_tokens,
    int combine_num_max_rdma_chunked_send_tokens,
    int combine_num_max_rdma_chunked_recv_tokens,
    int combine_num_max_nvl_chunked_send_tokens,
    int combine_num_max_nvl_chunked_recv_tokens,
    const __nv_bfloat16* W_gateup,
    const __nv_bfloat16* W_down,
    ComputeDType compute_dtype,
    const void* W_gateup_fp8,
    const void* W_down_fp8,
    const uint32_t* W_gateup_fp8_sf,
    const uint32_t* W_down_fp8_sf,
    int num_dispatch_sms,
    int num_forwarder_sms,
    int num_compute_sms,
    int num_combine_sms,
    int num_logical_channels,
    int max_tokens_per_expert,
    int max_total_recv_tokens,
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes,
    // Per-local-expert received-token counts (host array, length num_local_experts).
    // When provided, the per-expert slot buffers are packed compactly via an exclusive
    // prefix sum (total = Σ count). When nullptr, falls back to the fixed
    // num_local_experts * max_tokens_per_expert layout.
    const int* host_expert_count = nullptr,
    // Optional mapped device pointer to the same per-expert counts. When provided,
    // megakernel can build the compact expert layout on device and skip the H2D copy
    // for expert_count / expert_slot_base.
    const int* device_expert_count_mapped = nullptr,
    // Optional caller-owned buffers. Non-null pointers are borrowed by the state.
    __nv_bfloat16* external_bwd_fc1_input = nullptr,
    __nv_bfloat16* external_bwd_preact = nullptr,
    int* external_fwd_slot_map = nullptr,
    int4* external_combined_x = nullptr,
    float* external_combined_topk_weights = nullptr,
    TeraMoEState* host_state_out = nullptr,
    // RDMA-buffer-reuse mailboxes (symmetric, indexed by rdma_rank). Borrowed, not owned.
    int* rdma_reuse_dispatch_quiet_done = nullptr,
    int* rdma_reuse_combine_clear_done = nullptr,
    // Enable the in-kernel combine RDMA-reuse prelude (forward only for now).
    int rdma_reuse_prelude_enable = 0,
    // Runtime-tunable compute parameters (Python-selectable).
    int compute_batch_size = 4096,
    int combine_start_head_percent = 70);

void free_teramoe_fused_state(TeraMoEState* device_state, const TeraMoEState* cached_host_state = nullptr);
void free_megakernel_forward_transient(TeraMoEState* device_state);
void free_teramoe_forward_transients_from_host(TeraMoEState* host_state);
void get_megakernel_backward_dimensions(
    TeraMoEState* device_state,
    int* num_tokens,
    int* hidden,
    int* intermediate,
    int* num_topk,
    int* num_local_experts);
void get_megakernel_expert_counts(
    TeraMoEState* device_state,
    int* expert_counts,
    int num_local_experts);
int get_teramoe_compute_batch_size_default();
float* get_output_accum_ptr(TeraMoEState* device_state);
void* get_combined_x_ptr(TeraMoEState* device_state);
void launch_teramoe_fused_forward(
    TeraMoEState* device_state,
    const TeraMoEState* host_state,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream);
MegaKernelBackwardState* allocate_teramoe_fused_backward_state(
    TeraMoEState* fwd_device_state,
    const void* grad_output,
    void* grad_input,
    void* grad_w_gateup,
    void* grad_w_down,
    void* grad_topk_weights,
    void* wgrad_x_slot,
    void* wgrad_act_slot,
    void* wgrad_dz_slot,
    void* wgrad_dgu_slot,
    const int* host_expert_count,
    int total_sms,
    MegaKernelBackwardHostContext** host_context,
    cudaStream_t stream,
    const TeraMoEState* cached_fwd_host_state = nullptr);
void free_teramoe_fused_backward_state(
    MegaKernelBackwardState* backward_state,
    const MegaKernelBackwardHostContext* host_context = nullptr);
void free_teramoe_backward_host_context(MegaKernelBackwardHostContext* host_context);
void prepare_megakernel_communication_replay(
    TeraMoEState* device_state,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream);
void prepare_teramoe_backward_communication_replay(
    const MegaKernelBackwardHostContext* host_context,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream);
void launch_teramoe_fused_backward(
    MegaKernelBackwardState* backward_state,
    const MegaKernelBackwardHostContext* host_context,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream);

namespace detail {
namespace state_cache {
TeraMoEState* alloc_host();
void copy_host(TeraMoEState* dst, const TeraMoEState* src);
void free_host(TeraMoEState* ptr);
}  // namespace state_cache
}  // namespace detail


void teramoe_cached_notify(int hidden_int4,
                     int num_scales,
                     int num_topk_idx,
                     int num_topk_weights,
                     int num_ranks,
                     int num_channels,
                     int num_combined_tokens,
                     int* combined_rdma_head,
                     const int* rdma_channel_prefix_matrix,
                     const int* rdma_rank_prefix_sum,
                     int* combined_nvl_head,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool is_cached_dispatch,
                     bool skip_rdma_clean);

}  // namespace teramoe
