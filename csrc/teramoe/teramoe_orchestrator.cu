#include "config.hpp"
#include "kernels/api.cuh"
#include "kernels/buffer.cuh"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"
#include "kernels/ibgda_device.cuh"
#include "kernels/internode_common.cuh"
#include "kernels/launch.cuh"
#include "kernels/utils.cuh"
#include "teramoe_wrapper.cuh"

#include <cute/arch/simd_sm100.hpp>

#include <ATen/cuda/CUDABlas.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <limits>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

namespace teramoe {

using namespace ::deep_ep;
using ComputeDType = ::deep_ep::megakernel::ComputeDType;
namespace umma = ::deep_ep::megakernel::umma;

#define COMPUTE_BATCH_SIZE (state->compute_batch_size)
constexpr int COMPUTE_GROUP_SIZE = teramoe_config::kComputeGroupSize;
constexpr int COMPUTE_SCHEDULER_SMS = teramoe_config::kComputeSchedulerSms;
constexpr int GATHER_SMS = teramoe_config::kGatherSms;
constexpr int GATHER_SCHED_TID_BEGIN = teramoe_config::kGatherSchedTidBegin;
constexpr int NORMAL_SCHED_THREADS = teramoe_config::kNormalSchedThreads;
constexpr int GATHER_SCHED_MAX_WARPS = teramoe_config::kGatherSchedMaxWarps;
constexpr int MK_COMPUTE_CLUSTER_DIM = teramoe_config::kComputeClusterDim;
// COMBINE_START_HEAD_PERCENT is now a runtime field in TeraMoEState.
constexpr int MK_TIMEOUT_LOG_BUDGET = teramoe_config::kTimeoutLogBudget;
constexpr int MK_DISPATCH_ROLE_COUNT = teramoe_config::kDispatchRoleCount;
constexpr int PUB_RING_DEPTH = teramoe_config::kPubRingDepth;
constexpr int PUB_CONSUME_BATCH = teramoe_config::kPubConsumeBatch;
constexpr int PUB_PRODUCE_BATCH = teramoe_config::kPubProduceBatch;

enum TimeoutLogSite {
    kTimeoutLogComputeRoundFlush = 0,
    kTimeoutLogComputeReady = 1,
    kTimeoutLogCombineRdmaReceiver = 2,
    kTimeoutLogCombineForwarderNvl = 3,
    kTimeoutLogDispatchRound = 4,
    kTimeoutLogDispatchChannel = 5,
    kTimeoutLogCombineRdmaCheck = 6,
    kTimeoutLogCombineNvlCheck = 7,
    kTimeoutLogCombineNvlSender = 8,
    kTimeoutLogCombineForwarderRdma = 9,
    kTimeoutLogCombineBarrier = 10,
    kTimeoutLogCount = 11,
};

// Dispatch/combine constants. Original DeepEP computes kNumRDMARanks as
// num_ranks / NUM_MAX_NVL_PEERS, then uses SWITCH_RDMA_RANKS to select the
// compile-time kNumRDMARanks specialization.
constexpr int kNumCombineForwarderWarps = 24;
constexpr int kNumCombineTMABytesPerSenderWarp = 16384;
constexpr int kNumCombineTMABytesPerForwarderWarp = 9248;

template <int kNumRDMARanks>
struct MegaKernelRdmaConfig {
    static constexpr int kNumCombineWarpsPerForwarder =
        (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1;
    static constexpr int kNumCombineForwarders = kNumRDMARanks * kNumCombineWarpsPerForwarder;
    static constexpr int kNumCombineRDMAReceivers = kNumCombineForwarders - NUM_MAX_NVL_PEERS;
    static constexpr int kNumTopkCombineRDMARanks = internode::get_num_topk_rdma_ranks(kNumRDMARanks);
    static constexpr int kMegaKernelNumThreads = (kNumCombineForwarders + 1) * 32;
};

// SM Role assignment (configured at launch time)
enum class SmRole {
    kDispatch,      // Runs full DeepEP dispatch (even SM = forwarder, odd SM = sender)
    kCombine,       // DeepEP combine (even SM = NVLSender+RDMAReceiver, odd SM = Forwarder)
    kScheduler,     // Enqueues expert compute batches for dynamic compute groups
    kCompute,       // Pops compute tasks, does GEMM+SwiGLU
    kGather         // Dedicated gather SM worker for nhits>1 local reduce
};

struct ComputeTask {
    int expert_id;
    int start_slot;
    int num_tokens;
    int is_flush;
};

struct MegaKernelBackwardState;

// Headroom for the smaller arena chunk granularity below: with an 8 MiB default
// chunk, large activation buffers each take their own exact-sized chunk and the
// remaining small buffers pack into a few chunks, so the chunk count grows but
// stays well under this cap for realistic cases.
constexpr int kMegakernelArenaChunkCap = 128;

struct TeraMoEState {
    int* timeout_log_counters;        // [kTimeoutLogCount] per-site bounded logging budget
    // --- DeepEP NVSHMEM infrastructure (from Buffer object) ---
    void* rdma_buffer_ptr;            // Symmetric RDMA buffer base (for SymBuffer construction)
    void** buffer_ptrs;               // NVL buffer pointer array [NUM_MAX_NVL_PEERS]
    void** allocator_combine_buffer_ptrs; // External Buffer infrastructure used to rebuild state

    // --- Dispatch input data ---
    const int4* x;                    // [num_tokens, hidden_int4] input token data
    const uint32_t* x_scales;         // [num_tokens, num_scales] packed input scales
    const topk_idx_t* topk_idx;       // [num_tokens, num_topk] expert indices (global)
    const float* topk_weights;        // [num_tokens, num_topk] routing weights
    const bool* is_token_in_rank;     // [num_tokens, num_ranks] routing bitmap

    // --- Dispatch metadata for RDMA ---
    const int* rdma_channel_prefix_matrix;   // Per-rank cumulative token counts per logical channel
    const int* recv_rdma_rank_prefix_sum;    // Prefix sums for forwarder
    const int* gbl_channel_prefix_matrix;    // Global NVL-level prefix matrix per logical channel
    const int* recv_gbl_rank_prefix_sum;     // Global rank prefix sums
    int* send_rdma_head;             // [num_logical_channels, num_tokens, kNumRDMARanks] for combine
    int* send_nvl_head;              // [num_logical_channels, num_rdma_recv_tokens_ub, NUM_MAX_NVL_PEERS] for combine NVL tracking
    int* recv_rdma_channel_prefix_matrix;   // Written by forwarder
    int* recv_gbl_channel_prefix_matrix;    // Written by NVL receiver
    int* recv_rdma_channel_token_count;     // [kNumRDMARanks * num_logical_channels] non-cumulative logical-channel count
    int* recv_gbl_channel_token_count;      // [num_ranks * num_logical_channels] non-cumulative logical-channel count

    // --- Dispatch dimensions ---
    int num_tokens;                   // Total tokens to dispatch
    int hidden_int4;                  // hidden_dim / 4 (int4 units)
    int num_scales;                   // Number of scales per token (0 for BF16)
    int num_topk;                     // Top-k experts per token
    int num_experts;                  // Total experts globally
    int scale_token_stride;           // Stride for x_scales
    int scale_hidden_stride;          // Stride for x_scales hidden dim
    int allocator_num_dispatch_sms;
    int allocator_num_forwarder_sms;
    int allocator_num_compute_sms;
    int allocator_num_combine_sms;
    int allocator_num_logical_channels;
    int allocator_max_tokens_per_expert;
    int allocator_max_total_recv_tokens;
    int64_t allocator_num_rdma_bytes;
    int64_t allocator_num_nvl_bytes;

    // --- RDMA buffer sizing ---
    int num_max_rdma_chunked_send_tokens;   // Max tokens per RDMA put batch
    int num_max_rdma_chunked_recv_tokens;   // Recv buffer capacity per rank
    int num_max_nvl_chunked_send_tokens;    // Max tokens per NVL send batch
    int num_max_nvl_chunked_recv_tokens;    // NVL recv buffer capacity

    // --- Topology ---
    int rank;                         // Global rank (rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank)
    int num_ranks;                    // Total ranks

    // --- Receive-side signaling (written by Forwarder, read by Compute) ---
    int* expert_recv_count;           // [num_local_experts] continuous ready count (advanced by scheduler)
    int* expert_slot_ready;           // [num_local_experts * max_tokens_per_expert] per-slot ready flag

    // --- Per-expert receive storage (filled by NVL receiver) ---
    __nv_bfloat16* recv_tokens;       // [num_local_experts * max_tokens_per_expert, hidden]
    int* expert_token_offsets;        // [num_local_experts] — atomic write offset
    // --- Compact per-expert slot layout ---
    int* expert_slot_base;            // [num_local_experts] base offset into per-expert-slot buffers
    int* expert_count;                // [num_local_experts] received token count per local expert
    int* recv_token_source_info;      // [max_total_recv_tokens, 2] — (recv_token_idx, topk_slot)
    float* g_meta_route_w;            // [num_compute_groups * COMPUTE_BATCH_SIZE]
    int* g_meta_recv_idx;             // [num_compute_groups * COMPUTE_BATCH_SIZE] recv_token per row
    int* g_meta_topk_slot;            // [num_compute_groups * COMPUTE_BATCH_SIZE] compact fwd slot per row
    unsigned char* g_meta_is_single;  // [num_compute_groups * COMPUTE_BATCH_SIZE]

    // --- Compute signaling / per-slot output path (MEGAKERNEL_COMPUTE_DESIGN section III) ---
    int* token_compute_expected;        // [max_total_recv_tokens] how many local experts must compute this token
    __nv_bfloat16* compute_output_slot; // [num_local_experts * max_tokens_per_expert, hidden] per-slot output
    __nv_bfloat16* bwd_fc1_input;       // [max_total_recv_tokens, hidden] fc1 input (permuted X) by recv_token
    __nv_bfloat16* bwd_preact;          // [max_total_recv_tokens, num_topk, 2 * intermediate] saved gate/up by recv_token/topk_slot
    bool owns_bwd_fc1_input;
    bool owns_bwd_preact;
    int* fwd_slot_map;                  // [max_total_recv_tokens * num_topk] (recv_token,topk_slot) -> forward slot, -1 if none
    bool owns_fwd_slot_map;
    int* token_nhits;                   // [max_total_recv_tokens] #local-expert hits for this recv token
    int* token_slot_list;               // [max_total_recv_tokens * num_topk] absolute slot ids per hit
    int* expert_batch_enqueued;         // [num_local_experts * max_batches_per_expert] enqueue source: 0=none, 1=normal(combine-key), 3=tail
    int max_batches_per_expert;
    int* compute_group_barrier;         // [num_compute_groups] reusable global barrier counters
    int* compute_group_phase;           // [num_compute_groups] reusable global barrier phase flags
    ComputeTask* compute_tasks;         // [max_compute_tasks] dynamic compute task queue
    int max_compute_tasks;
    int* compute_task_head;             // CAS pop cursor
    int* compute_task_tail;             // visible publish cursor consumed by workers
    int* compute_task_reserve_tail;     // atomic reservation cursor used by scheduler lanes
    int* compute_enqueue_done;          // set after all scheduler lanes publish tail tasks
    int* scheduler_done_count;          // final-flush arrival barrier, then scheduler completion count
    int* priority_scheduler_done;       // 0=running/barrier closed, 1=priority stopped, 2=final-flush barrier released
    int* expert_enqueue_cursor;         // [num_local_experts] how many slots have been enqueued
    int* ready_batch_queue;             // [num_local_experts * max_batches_per_expert] appended batch id; -1 = not yet published
    int* ready_batch_reserve_tail;      // atomic append cursor (receiver)
    int* expert_batch_ready_count;      // [num_local_experts * max_batches_per_expert] per-batch slot-ready counter (publisher atomicAdd)
    int* compute_group_task_idx;        // [num_compute_groups] broadcast popped task idx to group SMs

    // --- Dedicated Gather SM state ---
    int* token_done_count;              // [max_total_recv_tokens] atomicAdd by compute worker per slot completion
    int* gather_claimed;                // [max_total_recv_tokens] CAS flag: scheduler has batched this token for gather
    int* combine_token_ready;           // [max_total_recv_tokens] set when token is ready for combine
    int* gather_ready_queue;            // [max_total_recv_tokens] token storage for scheduler-built gather tasks
    int* gather_ready_head;             // task queue consumer cursor; gather SMs CAS-pop one task at a time
    int* gather_ready_tail;             // ordered visible task tail published by scheduler lanes
    int* gather_ready_reserve_tail;     // token-storage reservation cursor into gather_ready_queue
    int* gather_scan_cursor;            // [COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS] next token for each gather scheduler warp
    int* gather_task_count;             // task metadata reservation cursor
    int* gather_task_tokens;            // [max_total_recv_tokens] task_idx -> token_base in gather_ready_queue
    int* gather_task_nhits;             // [max_total_recv_tokens] task_idx -> number of tokens in the task
    int* combine_done_count;            // atomic: how many combine SMs have fully finished
    int* combine_all_done;              // flag: 1 once all combine SMs finished; gather SMs poll this to exit

    // --- Expert weights ---
    const __nv_bfloat16* W_gateup;    // [num_local_experts, 2 * intermediate, hidden], rows [g0,u0,...]
    const __nv_bfloat16* W_down;      // [num_local_experts, hidden, intermediate]

    ComputeDType compute_dtype;

    umma::ComputeTmaAtoms* compute_tma;      // device ptr; wgateup[e]
    umma::ComputeDownTmaAtoms* compute_down_tma;  // device ptr; wdown[e]
    umma::InputTmaAtom_t* group_input_tma;   // device array [num_compute_groups]
    int num_compute_groups;                  // for indexing group_input_tma / barriers
    CUtensorMap recv_tokens_a_tma;

    // --- Compute output buffer ---
    __nv_bfloat16* combine_input;     // [max_total_recv_tokens, hidden] DeepEP compact recv-token namespace
    float* combine_input_topk_weights; // [max_total_recv_tokens, num_topk] DeepEP compact recv-token namespace
    internode::SourceMeta* combine_input_src_meta; // [max_total_recv_tokens] DeepEP compact recv-token namespace
    __nv_bfloat16* gemm_workspace;    // Scratch for gate/up intermediate results

    // --- Combine output ---
    float* output_accum;              // [num_tokens, hidden] float accumulator

    // --- Combine infrastructure (DeepEP combine kernel inputs) ---
    void* combine_rdma_buffer_ptr;            // Symmetric RDMA buffer for combine
    void** combine_buffer_ptrs;               // NVL buffer ptrs for combine [NUM_MAX_NVL_PEERS]
    int64_t num_rdma_bytes;                   // Capacity of each dispatch/combine RDMA region
    int64_t num_nvl_bytes;                    // Capacity of each dispatch/combine NVL region
    int4* combined_x;                         // [num_combined_tokens, hidden_int4] final output
    float* combined_topk_weights;             // [num_combined_tokens, num_topk] final topk weights
    bool owns_combined_x;
    bool owns_combined_topk_weights;
    const bool* is_combined_token_in_rank;    // [num_combined_tokens, num_ranks]
    const float* combine_topk_weights;        // topk_weights for combine
    const int4* combine_bias_0;               // bias (nullptr for MoE)
    const int4* combine_bias_1;               // bias (nullptr for MoE)
    const int* combined_rdma_head;            // [num_logical_channels, num_combined_tokens, kNumRDMARanks]
    int* combined_nvl_head;                   // [num_logical_channels, num_rdma_recv_tokens_ub, NUM_MAX_NVL_PEERS]
    const void* combine_src_meta;             // SourceMeta array
    const int* combine_rdma_channel_prefix_matrix;
    const int* combine_rdma_rank_prefix_sum;
    const int* combine_gbl_channel_prefix_matrix;
    const int* combine_gbl_channel_token_count;      // [num_ranks * num_logical_channels] non-cumulative logical-channel count
    const int* combine_rdma_channel_token_count;     // [kNumRDMARanks * num_logical_channels] non-cumulative logical-channel count
    int combine_num_tokens;                   // num tokens for combine (= tokens received by this rank)
    int combine_num_combined_tokens;          // num combined tokens (= original dispatch num_tokens)
    int combine_rdma_head_stride;             // num_combined_tokens * kNumRDMARanks per logical channel
    int combine_nvl_head_stride;              // num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS per logical channel
    int combine_hidden;                       // hidden dim in dtype units
    int num_max_combine_rdma_chunked_send_tokens;
    int num_max_combine_rdma_chunked_recv_tokens;
    int num_max_combine_nvl_chunked_send_tokens;
    int num_max_combine_nvl_chunked_recv_tokens;
    int num_combine_sms;                      // Must be even (even/odd SM pairing)
    int num_combine_channels;                 // = num_combine_sms / 2
    int num_logical_channels;                 // Logical channel count for dispatch/compute/combine overlap

    // --- Compute dimensions ---
    int hidden_dim;
    int intermediate_dim;
    int num_local_experts;
    int max_tokens_per_expert;
    int total_expert_slots;           // Σ expert_count[le] = size of the per-expert-slot buffers
    int max_total_recv_tokens;
    int compute_batch_size;           // Runtime-selected: 1024, 2048, or 4096
    int combine_start_head_percent;   // Runtime-tunable: combine SM waits until head/tail >= this %

    // --- SM allocation ---
    int num_dispatch_sms;
    int num_forwarder_sms;
    int num_combine_sms_total;        // Total combine SMs (alias for num_combine_sms above)
    int num_compute_sms;

    // --- Per-logical-channel overlap signaling ---
    int* channel_dispatch_done;       // [num_logical_channels] atomicAdd counter, reaches NUM_MAX_NVL_PEERS when logical channel done
    int* channel_normalized;          // [num_logical_channels] set to 1 after head normalization completes for that logical channel
    int* dispatch_channel_barrier;    // [num_logical_channels] counts dispatch SMs done with a logical channel
    int* dispatch_round_barrier;      // [num_dispatch_rounds] counts physical channels done with a dispatch round
    int* combine_channel_barrier;     // [num_logical_channels] counts combine SMs done with a logical channel

    // --- Dispatch SM config ---
    int num_dispatch_channels;        // = num_dispatch_sms / 2 (physical even/odd SM pairing)

    // --- RDMA buffer reuse (dispatch <-> combine) ---
    int* rdma_reuse_dispatch_quiet_done;  // borrowed (Buffer-owned symmetric memory)
    int* rdma_reuse_combine_clear_done;   // borrowed (Buffer-owned symmetric memory)
    int* rdma_reuse_prelude_done;         // state-owned [1] gate: leader sets, others wait
    int rdma_reuse_prelude_enable;        // 1 = run combine RDMA-reuse prelude (forward only)

    // --- Publish offload (dispatch->compute bridge) ---
    int* pending_topk_idx;            // [max_total_recv_tokens * num_topk] receiver-stashed expert ids
    float* pending_topk_weights;      // [max_total_recv_tokens * num_topk] receiver-stashed routing weights
    internode::SourceMeta* pending_meta; // [max_total_recv_tokens] receiver-stashed SourceMeta
    // [max_total_recv_tokens * num_topk] receiver-allocated expert abs-slot per topk hit
    // -1 for non-hit topk slots, else expert_slot_base+slot.
    int* pending_slot;
    int* pub_ring;                    // [num_pub_warps_total * PUB_RING_DEPTH] recv_token_idx queue
    int* pub_ring_head;               // [num_pub_warps_total] consumer cursor (publisher)
    int* pub_ring_tail;               // [num_pub_warps_total] producer cursor (receiver)
    int* recv_warp_done;              // [num_pub_warps_total] receiver warp finished producing
    int* publish_warp_done;           // [num_pub_warps_total] publisher warp drained and released metadata
    int* publish_done_count;          // atomic: how many publisher warps have drained
    int* publish_all_done;            // flag: 1 once all publishers drained (scheduler/gather use)
    int num_pub_warps_total;          // = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS

    // Backing store for the fused buffer-init descriptors (see fused_fill_kernel). Freed with the state.
    void* fused_fill_desc_buf;

    // --- Owned arena bookkeeping ---
    void* persistent_arena_chunks[kMegakernelArenaChunkCap];
    int persistent_arena_chunk_count;
    void* transient_arena_chunks[kMegakernelArenaChunkCap];
    int transient_arena_chunk_count;

};

// Instantiated template constants
constexpr int kNumDispatchRDMASenderWarps = 7;
constexpr int kNumTMABytesPerWarp = 16384;
constexpr bool kLowLatencyMode = false;
constexpr bool kCachedMode = false;

__device__ __forceinline__ int get_publish_warp_index(int dispatch_sm_idx, int src_nvl_rank) {
    return (dispatch_sm_idx / 2) * NUM_MAX_NVL_PEERS + src_nvl_rank;
}

__device__ __forceinline__ int publish_recv_token_from_pending(
    TeraMoEState* state,
    int recv_token_idx,
    int local_expert_begin,
    int num_topk,
    int lane_id
) {
    const int local_expert_end = local_expert_begin + state->num_local_experts;
    int expert_id = -1;
    float route_w = 0.0f;
    bool is_local_hit = false;
    if (lane_id < num_topk) {
        expert_id = ld_nc_global(&state->pending_topk_idx[recv_token_idx * num_topk + lane_id]);
        is_local_hit = (expert_id >= local_expert_begin && expert_id < local_expert_end);
        if (is_local_hit) {
            route_w = ld_nc_global(&state->pending_topk_weights[recv_token_idx * num_topk + lane_id]);
            state->combine_input_topk_weights[recv_token_idx * num_topk + lane_id] = route_w;
        }
    }

    const unsigned hit_mask = __ballot_sync(0xffffffff, is_local_hit);
    const int num_hits = __popc(hit_mask);
    int hit_base = 0;
    int hit_abs_slot = -1;
    int hit_slot = -1;
    int hit_rank = is_local_hit ? __popc(hit_mask & ((1u << lane_id) - 1)) : -1;

    if (num_hits > 0) {
        if (lane_id == 0) {
            state->combine_input_src_meta[recv_token_idx] = state->pending_meta[recv_token_idx];
            if (num_hits > num_topk) {
                printf("MK publish token hit overflow, rank=%d recv_token=%d num_hits=%d num_topk=%d\n",
                       state->rank, recv_token_idx, num_hits, num_topk);
                __threadfence_system(); trap();
            }
            st_na_global(&state->token_nhits[recv_token_idx], num_hits);
            st_na_global(&state->token_compute_expected[recv_token_idx], num_hits);
        }

        if (is_local_hit) {
            int local_expert_id = expert_id - local_expert_begin;
            hit_abs_slot = ld_nc_global(&state->pending_slot[recv_token_idx * num_topk + lane_id]);
            hit_slot = hit_abs_slot - state->expert_slot_base[local_expert_id];
        }


        if (is_local_hit) {
            int* dst_ptr = &state->recv_token_source_info[hit_abs_slot * 2];
            st_na_global(dst_ptr, recv_token_idx);
            st_na_global(dst_ptr + 1, lane_id);
            state->token_slot_list[recv_token_idx * num_topk + hit_base + hit_rank] = hit_abs_slot;
        }

        if (is_local_hit) {
            int local_expert_id = expert_id - local_expert_begin;
            st_na_release(&state->expert_slot_ready[state->expert_slot_base[local_expert_id] + hit_slot], 1);
            int batch_id = hit_slot / COMPUTE_BATCH_SIZE;
            int batch_start = batch_id * COMPUTE_BATCH_SIZE;
            int ecnt = state->expert_count[local_expert_id];
            int batch_end = batch_start + COMPUTE_BATCH_SIZE;
            if (batch_end > ecnt) batch_end = ecnt;
            int batch_size = batch_end - batch_start;
            int mbe = state->max_batches_per_expert;
            int done = atomicAdd(&state->expert_batch_ready_count[local_expert_id * mbe + batch_id], 1) + 1;
            if (done == batch_size) {
                int enc = local_expert_id * mbe + batch_id;
                int qpos = atomicAdd(state->ready_batch_reserve_tail, 1);
                st_na_release(&state->ready_batch_queue[qpos], enc);
            }
        }
    }
    __syncwarp();
    return num_hits;
}

__device__ void publish_worker(int dispatch_sm_idx, int src_nvl_rank, TeraMoEState* state) {
    const int lane_id = get_lane_id();
    const int pw = get_publish_warp_index(dispatch_sm_idx, src_nvl_rank);
    const int local_expert_begin = state->rank * state->num_local_experts;
    const int num_topk = state->num_topk;
    int head = 0;

    while (true) {
        int recv_done_snapshot = ld_acquire_global(&state->recv_warp_done[pw]);
        int tail = ld_acquire_global(&state->pub_ring_tail[pw]);
        if (head == tail) {
            if (recv_done_snapshot != 0) {
                tail = ld_acquire_global(&state->pub_ring_tail[pw]);
                if (head == tail)
                    break;
            }
            __nanosleep(32);
            continue;
        }

        int batch_count = tail - head;
        if (batch_count > PUB_CONSUME_BATCH)
            batch_count = PUB_CONSUME_BATCH;


        for (int i = 0; i < batch_count; ++i) {
            int recv_token_idx = ld_acquire_global(&state->pub_ring[pw * PUB_RING_DEPTH + ((head + i) % PUB_RING_DEPTH)]);
            int num_hits = publish_recv_token_from_pending(state, recv_token_idx, local_expert_begin, num_topk, lane_id
            );
            if (lane_id == 0) {
            }
        }

        head += batch_count;
        if (lane_id == 0) {
            st_na_release(&state->pub_ring_head[pw], head);
        }
        __syncwarp();
    }

    if (lane_id == 0) {
        __threadfence();
        st_na_release(&state->publish_warp_done[pw], 1);
        int done = atomicAdd(state->publish_done_count, 1) + 1;
        if (done == state->num_pub_warps_total) {
            __threadfence();
            st_na_release(state->publish_all_done, 1);
        }
    }
}

__device__ __forceinline__ void compute_group_sync(TeraMoEState* state, int group_id, int group_size) {
    EP_DEVICE_ASSERT(group_size > 0 && group_size <= COMPUTE_GROUP_SIZE);
    EP_DEVICE_ASSERT(group_id >= 0 && group_id < state->num_compute_groups);
    __syncthreads();
    memory_fence_gpu();
    if (threadIdx.x == 0) {
        int phase = ld_acquire_global(&state->compute_group_phase[group_id]);
        int arrived = atomicAdd(&state->compute_group_barrier[group_id], 1) + 1;
        if (arrived == group_size) {
            st_release_gpu_global(&state->compute_group_barrier[group_id], 0);
            st_release_gpu_global(&state->compute_group_phase[group_id], phase + 1);
        } else {
            while (ld_acquire_global(&state->compute_group_phase[group_id]) == phase)
                __nanosleep(64);
        }
    }
    __syncthreads();
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_worker(
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    TeraMoEState* state,
    uint8_t* smem_buffer
);

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    uint8_t* smem_buffer);

template <ComputeDType kComputeDType>
__device__ __forceinline__ void combine_precompute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    uint8_t* smem_buffer);

#define MK_FORWARD_COMPUTE_WORKER(COMPUTE_DTYPE, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, STATE, SMEM_BUFFER) \
    compute_worker<COMPUTE_DTYPE>((SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (STATE), (SMEM_BUFFER))

#define MK_BACKWARD_COMPUTE_WORKER(COMPUTE_DTYPE, BS, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, SMEM_BUFFER) \
    compute_backward_worker<COMPUTE_DTYPE>((BS), (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (SMEM_BUFFER))

#define MK_DISPATCH_REUSED_COMPUTE(IS_BACKWARD, COMPUTE_DTYPE, BS, SM_ID, COMPUTE_SM_IDX, NUM_COMPUTE_SMS, STATE, SMEM_BUFFER) \
    if constexpr (IS_BACKWARD) { \
        EP_DEVICE_ASSERT((BS) != nullptr); \
        MK_BACKWARD_COMPUTE_WORKER(COMPUTE_DTYPE, (BS), (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (SMEM_BUFFER)); \
    } else { \
        MK_FORWARD_COMPUTE_WORKER(COMPUTE_DTYPE, (SM_ID), (COMPUTE_SM_IDX), (NUM_COMPUTE_SMS), (STATE), (SMEM_BUFFER)); \
    }

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType, bool kDispatchBackwardCompute = false>
__device__ void dispatch_worker(
    int sm_id,
    int dispatch_sm_idx,  // 0-based index among all dispatch SMs
    TeraMoEState* state,
    MegaKernelBackwardState* backward_state = nullptr
) {
    using namespace internode;
    constexpr int kNumTopkRDMARanks = internode::get_num_topk_rdma_ranks(kNumRDMARanks);
    const auto num_sms = state->num_dispatch_sms;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    const auto num_channels = state->num_dispatch_channels, channel_id = sm_id / 2;
    constexpr int num_logical_channels_per_physical = kStage;
    const int num_logical_channels = num_channels * num_logical_channels_per_physical;
    const bool is_forwarder = dispatch_sm_idx % 2 == 0;
    const auto rdma_rank = state->rank / NUM_MAX_NVL_PEERS, nvl_rank = state->rank % NUM_MAX_NVL_PEERS;
    const auto num_ranks = state->num_ranks;

    constexpr int kDispatchWorkerWarps = kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS;
    EP_DEVICE_ASSERT(num_warps >= kDispatchWorkerWarps);
    EP_DEVICE_ASSERT(num_warps >= kDispatchWorkerWarps + NUM_MAX_NVL_PEERS);

    if (!is_forwarder && warp_id >= kDispatchWorkerWarps &&
        warp_id < kDispatchWorkerWarps + NUM_MAX_NVL_PEERS) {
        const int publisher_slot = warp_id - kDispatchWorkerWarps;
        const int paired_receiver_warp = kNumDispatchRDMASenderWarps + 1 + publisher_slot;
        const int src_nvl_rank = (paired_receiver_warp + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS;
        publish_worker(dispatch_sm_idx, src_nvl_rank, state);
    }
    const bool dispatch_thread_active = warp_id < kDispatchWorkerWarps;
    if (dispatch_thread_active) {

    enum class WarpRole { kRDMASender, kRDMASenderCoordinator, kRDMAAndNVLForwarder, kForwarderCoordinator, kNVLReceivers };
    const auto role_meta = [=]() -> std::pair<WarpRole, int> {
        if (is_forwarder) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
            }
        } else if (warp_id < kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASender, -1};
        } else if (warp_id == kNumDispatchRDMASenderWarps) {
            return {WarpRole::kRDMASenderCoordinator, -1};
        } else {
            return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
        }
    }();
    auto warp_role = role_meta.first;
    auto target_rank = role_meta.second;
    int dispatch_role_id = 0;
    if (warp_role == WarpRole::kRDMASender)
        dispatch_role_id = 0;
    else if (warp_role == WarpRole::kRDMASenderCoordinator)
        dispatch_role_id = 1;
    else if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        dispatch_role_id = 2;
    else if (warp_role == WarpRole::kForwarderCoordinator)
        dispatch_role_id = 3;
    else
        dispatch_role_id = 4;
    int dispatch_role_slot = target_rank >= 0 ? target_rank : 0;
    if (warp_role == WarpRole::kRDMASender)
        dispatch_role_slot = warp_id;
    if (dispatch_role_slot >= NUM_MAX_NVL_PEERS)
        dispatch_role_slot = NUM_MAX_NVL_PEERS - 1;
    // Data dimensions
    const int hidden_int4 = state->hidden_int4;
    const int num_scales = state->num_scales;
    const int num_topk = state->num_topk;
    const int num_tokens = state->num_tokens;
    const int num_experts = state->num_experts;
    const int scale_token_stride = state->scale_token_stride;
    const int scale_hidden_stride = state->scale_hidden_stride;
    EP_DEVICE_ASSERT(num_topk <= 32);

    auto num_bytes_per_token = get_num_bytes_per_token(hidden_int4, num_scales, num_topk, num_topk);
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto scale_bytes = num_scales * sizeof(float);
    const int num_max_rdma_chunked_send_tokens = state->num_max_rdma_chunked_send_tokens;
    const int num_max_rdma_chunked_recv_tokens = state->num_max_rdma_chunked_recv_tokens;
    const int num_max_nvl_chunked_send_tokens = state->num_max_nvl_chunked_send_tokens;
    const int num_max_nvl_chunked_recv_tokens = state->num_max_nvl_chunked_recv_tokens;

    // Input pointers
    const int4* x = state->x;
    const uint32_t* x_scales = state->x_scales;
    const topk_idx_t* topk_idx = state->topk_idx;
    const float* topk_weights = state->topk_weights;
    const bool* is_token_in_rank = state->is_token_in_rank;
    const int* rdma_channel_prefix_matrix = state->rdma_channel_prefix_matrix;
    const int* recv_rdma_rank_prefix_sum = state->recv_rdma_rank_prefix_sum;
    const int* gbl_channel_prefix_matrix = state->gbl_channel_prefix_matrix;
    const int* recv_gbl_rank_prefix_sum = state->recv_gbl_rank_prefix_sum;
    int* send_rdma_head = state->send_rdma_head;
    int* send_nvl_head_base = state->send_nvl_head;
    int* recv_rdma_channel_prefix_matrix = state->recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix = state->recv_gbl_channel_prefix_matrix;

    // RDMA symmetric layout
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t), "Invalid number of NVL peers");
    void* rdma_buffer_ptr = state->rdma_buffer_ptr;

    // NVL buffer layouts
    void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
    int rs_wr_rank = 0, ws_rr_rank = 0;
    if (warp_role == WarpRole::kRDMAAndNVLForwarder)
        rs_wr_buffer_ptr = state->buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = state->buffer_ptrs[target_rank],
        rs_wr_rank = nvl_rank, ws_rr_rank = target_rank;
    if (warp_role == WarpRole::kNVLReceivers)
        rs_wr_buffer_ptr = state->buffer_ptrs[target_rank], ws_rr_buffer_ptr = state->buffer_ptrs[nvl_rank],
        rs_wr_rank = target_rank, ws_rr_rank = nvl_rank;

    // RDMA sender warp synchronization
    __shared__ int rdma_send_channel_lock[kNumRDMARanks];
    __shared__ int rdma_send_channel_tail[kNumRDMARanks];
    __shared__ uint32_t rdma_send_channel_window[kNumRDMARanks];
    auto sync_rdma_sender_smem = []() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

    // TMA stuffs
    auto tma_buffer = smem_buffer + target_rank * kNumTMABytesPerWarp;
    auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
    uint32_t tma_phase = 0;
    if ((warp_role == WarpRole::kRDMAAndNVLForwarder or warp_role == WarpRole::kNVLReceivers) and elect_one_sync()) {
        mbarrier_init(tma_mbarrier, 1);
        fence_barrier_init();
        EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumTMABytesPerWarp);
    }
    __syncwarp();

    // Forward warp synchronization
    __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
    __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
    auto sync_forwarder_smem = []() { asm volatile("barrier.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

    int sender_cached_rdma_channel_head = 0, sender_global_rdma_tail_idx = 0;
    int coordinator_last_issued_tail = 0;
    int forwarder_cached_rdma_channel_head = 0, forwarder_cached_rdma_channel_tail = 0;
    int forwarder_cached_nvl_channel_head = 0, forwarder_cached_nvl_channel_tail = 0;
    int forwarder_rdma_nvl_token_idx = 0;
    int receiver_cached_channel_head_idx = 0, receiver_cached_channel_tail_idx = 0;

    for (int logical_stage = 0; logical_stage < num_logical_channels_per_physical; ++logical_stage) {
        const int logical_channel_id = channel_id * num_logical_channels_per_physical + logical_stage;
        auto rdma_channel_data = SymBuffer<uint8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token,
            kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_meta = SymBuffer<int>(
            rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2,
            kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(
            rdma_buffer_ptr, 1, kNumRDMARanks, logical_channel_id, num_logical_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(
            rdma_buffer_ptr, 1, kNumRDMARanks, logical_channel_id, num_logical_channels);

        auto nvl_channel_x = AsymBuffer<uint8_t>(
            ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
            NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_prefix_start = AsymBuffer<int>(
            ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_prefix_end = AsymBuffer<int>(
            ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(
            rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, ws_rr_rank)
            .advance_also(ws_rr_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(
            ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS,
            logical_channel_id, num_logical_channels, rs_wr_rank)
            .advance_also(rs_wr_buffer_ptr);

        sender_cached_rdma_channel_head = 0;
        sender_global_rdma_tail_idx = 0;
        coordinator_last_issued_tail = 0;
        forwarder_cached_rdma_channel_head = 0;
        forwarder_cached_rdma_channel_tail = 0;
        forwarder_cached_nvl_channel_head = 0;
        forwarder_cached_nvl_channel_tail = 0;
        forwarder_rdma_nvl_token_idx = 0;
        receiver_cached_channel_head_idx = 0;
        receiver_cached_channel_tail_idx = 0;
    // ========== kRDMASender ==========
    if (warp_role == WarpRole::kRDMASender) {
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);

        // Send channel prefix metadata
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
        for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
            auto dst_ptr =
                dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank) : rdma_channel_meta.send_buffer(dst_rdma_rank);
            if (lane_id < NUM_MAX_NVL_PEERS) {
                int prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_logical_channels + logical_channel_id;
                int prefix_start = logical_channel_id == 0 ? 0 : gbl_channel_prefix_matrix[prefix_idx - 1];
                dst_ptr[lane_id] = -prefix_start - 1;
            } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
                int src_nvl_rank = lane_id - NUM_MAX_NVL_PEERS;
                int prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
                dst_ptr[lane_id] = -gbl_channel_prefix_matrix[prefix_idx] - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
                int prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                int prefix_start = logical_channel_id == 0 ? 0 : rdma_channel_prefix_matrix[prefix_idx - 1];
                dst_ptr[lane_id] = -prefix_start - 1;
            } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
                int prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                dst_ptr[lane_id] = -rdma_channel_prefix_matrix[prefix_idx] - 1;
            }
            __syncwarp();

            if (dst_rdma_rank != rdma_rank) {
                nvshmemi_ibgda_put_nbi_warp<true>(reinterpret_cast<uint64_t>(rdma_channel_meta.recv_buffer(rdma_rank)),
                                                  reinterpret_cast<uint64_t>(rdma_channel_meta.send_buffer(dst_rdma_rank)),
                                                  sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2),
                                                  translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                  channel_id, lane_id, 0);
            }
        }
        sync_rdma_sender_smem();

        // Iterate tokens
        int64_t token_idx;
        auto& cached_rdma_channel_head = sender_cached_rdma_channel_head;
        auto& global_rdma_tail_idx = sender_global_rdma_tail_idx;
        auto send_buffer = lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
        for (token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            uint64_t is_token_in_rank_uint64 = 0;
            if (lane_id < kNumRDMARanks) {
                is_token_in_rank_uint64 =
                    __ldg(reinterpret_cast<const uint64_t*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS));
                global_rdma_tail_idx += (is_token_in_rank_uint64 != 0);
            }
            __syncwarp();

            if ((token_idx - token_start_idx) % kNumDispatchRDMASenderWarps != warp_id)
                continue;
            auto rdma_tail_idx = is_token_in_rank_uint64 == 0 ? -1 : global_rdma_tail_idx - 1;

            // Wait buffer release
            auto start_time = clock64();
            while (is_token_in_rank_uint64 != 0 and rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
                cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
                if (clock64() - start_time >= NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch RDMA sender timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA lane: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, lane_id, cached_rdma_channel_head, rdma_tail_idx);
                    __threadfence_system(); trap();
                }
            }
            __syncwarp();

            // Store RDMA head for combine in this logical channel's independent namespace.
            int* logical_send_rdma_head = send_rdma_head + logical_channel_id * num_tokens * kNumRDMARanks;
            if (lane_id < kNumRDMARanks)
                logical_send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

            // Broadcast tails
            SourceMeta src_meta;
            int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
            void* dst_send_buffers[kNumTopkRDMARanks];
            #pragma unroll
            for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
                if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
                    slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
                    topk_ranks[num_topk_ranks] = i;
                    auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
                    auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
                    if (lane_id == num_topk_ranks)
                        src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
                    dst_send_buffers[num_topk_ranks++] =
                        reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_token;
                }
            EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);


            // Copy x
            auto st_broadcast = [=](const int key, const int4& value) {
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
            };
            UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

            // Copy packed UE8M0 x scales. Layout stays 4 bytes per scale pack.
            #pragma unroll
            for (int i = lane_id; i < num_scales; i += 32) {
                auto offset = token_idx * scale_token_stride + i * scale_hidden_stride;
                auto value = ld_nc_global(x_scales + offset);
                #pragma unroll
                for (int j = 0; j < num_topk_ranks; ++j)
                    st_na_global(reinterpret_cast<uint32_t*>(dst_send_buffers[j]) + i, value);
            }
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<uint32_t*>(dst_send_buffers[i]) + num_scales;

            // Copy source metadata
            if (lane_id < num_topk_ranks)
                st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
            #pragma unroll
            for (int i = 0; i < num_topk_ranks; ++i)
                dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

            // Copy topk_idx and topk_weights
            #pragma unroll
            for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
                auto rank_idx = i / num_topk, copy_idx = i % num_topk;
                auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
                auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
                st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
                st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
            }
            __syncwarp();

            // Release transaction window
            if (is_token_in_rank_uint64 != 0) {
                acquire_lock(rdma_send_channel_lock + lane_id);
                auto latest_tail = rdma_send_channel_tail[lane_id];
                auto offset = rdma_tail_idx - latest_tail;
                while (offset >= 32) {
                    release_lock(rdma_send_channel_lock + lane_id);
                    acquire_lock(rdma_send_channel_lock + lane_id);
                    latest_tail = rdma_send_channel_tail[lane_id];
                    offset = rdma_tail_idx - latest_tail;
                }
                auto window = rdma_send_channel_window[lane_id] | (1u << offset);
                if (offset == 0) {
                    auto num_empty_slots = (~window) == 0 ? 32 : __ffs(~window) - 1;
                    st_release_cta(rdma_send_channel_tail + lane_id, latest_tail + num_empty_slots);
                    window >>= num_empty_slots;
                }
                rdma_send_channel_window[lane_id] = window;
                release_lock(rdma_send_channel_lock + lane_id);
            }
            __syncwarp();
        }

    // ========== kRDMASenderCoordinator ==========
    } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
        EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_lock[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
        (lane_id < kNumRDMARanks) ? (rdma_send_channel_window[lane_id] = 0) : 0;
        sync_rdma_sender_smem();

        int num_tokens_to_send = 0;
        if (lane_id < kNumRDMARanks) {
            num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id];
            if (logical_channel_id > 0)
                num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id - 1];
        }


        auto& last_issued_tail = coordinator_last_issued_tail;
        auto start_time = clock64();
        while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                printf("MK RDMA coordinator timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, tail: %d, remaining: %d\n",
                       channel_id, rdma_rank, nvl_rank, lane_id, last_issued_tail, num_tokens_to_send);
                __threadfence_system(); trap();
            }

            for (int i = 0, synced_num_tokens_to_send; i < kNumRDMARanks; ++i) {
                int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;
                synced_num_tokens_to_send = __shfl_sync(0xffffffff, num_tokens_to_send, dst_rdma_rank);
                if (synced_num_tokens_to_send == 0)
                    continue;

                auto processed_tail =
                    __shfl_sync(0xffffffff, ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank)), 0);
                auto synced_last_issued_tail = __shfl_sync(0xffffffff, last_issued_tail, dst_rdma_rank);
                auto num_tokens_processed = processed_tail - synced_last_issued_tail;
                if (num_tokens_processed != synced_num_tokens_to_send and num_tokens_processed < num_max_rdma_chunked_send_tokens)
                    continue;

                auto num_tokens_to_issue = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
                EP_DEVICE_ASSERT(num_tokens_to_issue >= 0 and num_tokens_to_issue <= synced_num_tokens_to_send);
                if (dst_rdma_rank != rdma_rank) {
                    auto dst_slot_idx = synced_last_issued_tail % num_max_rdma_chunked_recv_tokens;
                    EP_DEVICE_ASSERT(dst_slot_idx + num_tokens_to_issue <= num_max_rdma_chunked_recv_tokens);
                    const size_t num_bytes_per_msg = num_bytes_per_token * num_tokens_to_issue;
                    const auto dst_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    const auto src_ptr =
                        reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + dst_slot_idx * num_bytes_per_token);
                    nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr, src_ptr, num_bytes_per_msg,
                                                      translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                      channel_id, lane_id, 0);
                } else {
                    memory_fence();
                }
                __syncwarp();

                if (lane_id == dst_rdma_rank) {
                    last_issued_tail += num_tokens_to_issue;
                    num_tokens_to_send -= num_tokens_to_issue;
                    nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_tail.buffer(rdma_rank),
                                                    num_tokens_to_issue,
                                                    translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                    channel_id,
                                                    dst_rdma_rank == rdma_rank);
                }
                __syncwarp();
            }
        }

    // ========== kRDMAAndNVLForwarder ==========
    } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
        const auto dst_nvl_rank = target_rank;

        // Wait counters to arrive
        int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
        EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
        auto start_time = clock64();
        if (lane_id < kNumRDMARanks) {
            while (true) {
                auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
                auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
                auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
                auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
                if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
                    int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
                    EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
                    st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
                    st_release_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

                    src_rdma_channel_prefix = -meta_2 - 1;
                    auto src_rdma_channel_prefix_1 = -meta_3 - 1;
                    num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
                    recv_rdma_channel_prefix_matrix[lane_id * num_logical_channels + logical_channel_id] = src_rdma_channel_prefix_1;
                    state->recv_rdma_channel_token_count[lane_id * num_logical_channels + logical_channel_id] = num_tokens_to_recv_from_rdma;
                    src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
                    EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
                    break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (RDMA meta), rank=%d block=%d thread=%d channel=%d logical_ch=%d RDMA=%d nvl=%d src RDMA lane=%d dst NVL=%d meta=(%d,%d,%d,%d)\n",
                           state->rank, static_cast<int>(blockIdx.x), thread_id, channel_id, logical_channel_id,
                           rdma_rank, nvl_rank, lane_id, dst_nvl_rank,
                           meta_0, meta_1, meta_2, meta_3);
                    __threadfence_system(); trap();
                }
            }
        }
        __syncwarp();

        // Shift cached head inside this logical channel's independent NVL head namespace.
        int* send_nvl_head = send_nvl_head_base + logical_channel_id * state->combine_nvl_head_stride +
            src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

        // Wait shared memory to be cleaned
        sync_forwarder_smem();


        // Forward tokens from RDMA buffer
        int src_rdma_rank = dispatch_sm_idx % kNumRDMARanks;
        auto& cached_rdma_channel_head = forwarder_cached_rdma_channel_head;
        auto& cached_rdma_channel_tail = forwarder_cached_rdma_channel_tail;
        auto& cached_nvl_channel_head = forwarder_cached_nvl_channel_head;
        auto& cached_nvl_channel_tail = forwarder_cached_nvl_channel_tail;
        auto& rdma_nvl_token_idx = forwarder_rdma_nvl_token_idx;
        while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
            // Check NVL destination queue
            start_time = clock64();
            while (true) {
                const int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
                if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens)
                    break;
                cached_nvl_channel_head = __shfl_sync(0xffffffffu, ld_volatile_global(nvl_channel_head.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank,
                           ld_volatile_global(nvl_channel_head.buffer()), cached_nvl_channel_tail);
                    __threadfence_system(); trap();
                }
            }

            // Find next source RDMA rank
            start_time = clock64();
            while (true) {
                src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
                if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
                    if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail)
                        cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
                    if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank))
                        break;
                }

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
                    printf("MK dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA: %d\n",
                           channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id);
                    __threadfence_system(); trap();
                }
            }
            auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
            auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

            // Iterate tokens
            for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
                auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
                auto shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_token;
                auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
                bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);

                if (lane_id == src_rdma_rank) {
                    auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
                    int rdma_nvl_token_idx_before = rdma_nvl_token_idx;
                    rdma_nvl_token_idx += is_in_dst_nvl_rank;
                    int* nvl_head_ptr = send_nvl_head + i * NUM_MAX_NVL_PEERS;
                    *nvl_head_ptr = cached_head;

                }
                if (not is_in_dst_nvl_rank)
                    continue;

                // Get empty slot
                int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;
                auto dst_shifted = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;

                // TMA copy
                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, num_bytes_per_token, false);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, num_bytes_per_token);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);
                if (elect_one_sync())
                    tma_store_1d(tma_buffer, dst_shifted, num_bytes_per_token);
                __syncwarp();

                if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens)
                    src_rdma_tail = i + 1;

                tma_store_wait<0>();
                __syncwarp();
            }

            // Sync head index
            if (lane_id == src_rdma_rank)
                forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

            // Move tail index
            __syncwarp();
            if (elect_one_sync())
                st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
        }

        // Retired
        __syncwarp();
        if (elect_one_sync()) {
            forward_channel_retired[dst_nvl_rank] = true;
        }

    // ========== kForwarderCoordinator ==========
    } else if (warp_role == WarpRole::kForwarderCoordinator) {

        if (target_rank == 0) {
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
        #pragma unroll
        for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
            forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
        if (lane_id < NUM_MAX_NVL_PEERS)
            forward_channel_retired[lane_id] = false;
        sync_forwarder_smem();

        int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
        while (true) {
            int min_head = std::numeric_limits<int>::max();
            #pragma unroll
            for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
                if (not forward_channel_retired[i])
                    min_head = min(min_head, forward_channel_head[i][target_rdma]);
            if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max()))
                break;

            if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + num_max_rdma_chunked_send_tokens and
                lane_id < kNumRDMARanks) {
                nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                min_head - last_head,
                                                translate_dst_rdma_rank<kLowLatencyMode>(lane_id, nvl_rank),
                                                channel_id + num_channels,
                                                lane_id == rdma_rank);
                last_head = min_head;
            }

            __nanosleep(NUM_WAIT_NANOSECONDS);
        }
        }

    // ========== kNVLReceivers ==========
    } else {
        int src_nvl_rank = target_rank, total_offset = 0;
        const int local_expert_begin = state->rank * (num_experts / num_ranks);

        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
            total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

        // Receive channel offsets
        int start_offset = 0, end_offset = 0, num_tokens_to_recv;
        auto start_time = clock64();
        while (lane_id < kNumRDMARanks) {
            end_offset = ld_acquire_sys_global(nvl_channel_prefix_end.buffer() + lane_id);
            start_offset = end_offset < 0
                ? ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id)
                : 0;
            if (start_offset < 0 and end_offset < 0) {
                start_offset = -start_offset - 1, end_offset = -end_offset - 1;
                total_offset += start_offset;
                break;
            }

            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                int raw_start_now = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
                int raw_end_now = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
                printf("MK dispatch NVL receiver timeout, rank=%d block=%d thread=%d channel=%d logical_ch=%d RDMA=%d nvl=%d src RDMA=%d src nvl=%d prefix=(%d,%d) cached_head=%d cached_tail=%d prefix_ptr=(%p,%p) raw_now=(%d,%d) target=%d warp=%d rs_wr=%d ws_rr=%d\n",
                       state->rank, static_cast<int>(blockIdx.x), thread_id, channel_id, logical_channel_id,
                       rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset,
                       receiver_cached_channel_head_idx, receiver_cached_channel_tail_idx,
                       nvl_channel_prefix_start.buffer() + lane_id,
                       nvl_channel_prefix_end.buffer() + lane_id,
                       raw_start_now, raw_end_now,
                       target_rank, warp_id, rs_wr_rank, ws_rr_rank);
                __threadfence_system(); trap();
            }
        }
        num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

        // Save for combine usage
        if (lane_id < kNumRDMARanks) {
            int idx = (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
            recv_gbl_channel_prefix_matrix[idx] = total_offset;
        }
        // Save per-channel token count (non-cumulative) for combine NVL Sender
        if (lane_id < kNumRDMARanks) {
            int idx = (lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_logical_channels + logical_channel_id;
            int count = end_offset - start_offset;
            state->recv_gbl_channel_token_count[idx] = count;
        }
        __syncwarp();


        auto& cached_channel_head_idx = receiver_cached_channel_head_idx;
        auto& cached_channel_tail_idx = receiver_cached_channel_tail_idx;
        const int pub_warp_idx = get_publish_warp_index(dispatch_sm_idx, src_nvl_rank);
        int producer_tail = 0;
        int producer_batch_count = 0;
        if (lane_id == 0)
            producer_tail = ld_acquire_global(&state->pub_ring_tail[pub_warp_idx]);
        producer_tail = __shfl_sync(0xffffffff, producer_tail, 0);
        while (num_tokens_to_recv > 0) {
            // Wait for data
            start_time = clock64();
            while (true) {
                if (cached_channel_head_idx != cached_channel_tail_idx)
                    break;
                cached_channel_tail_idx = __shfl_sync(0xffffffff, ld_acquire_sys_global(nvl_channel_tail.buffer()), 0);

                if (elect_one_sync() and clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK dispatch NVL receiver timeout (data), channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
                           channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
                    __threadfence_system(); trap();
                }
            }

            // Copy data
            int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
            for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
                int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
                auto shifted = nvl_channel_x.buffer() + token_idx_in_buffer * num_bytes_per_token;
                auto meta = ld_nc_global(reinterpret_cast<SourceMeta*>(shifted + hidden_bytes + scale_bytes));
                int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
                (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

                // TMA copy to recv_x (DeepEP standard path)
                bool scale_aligned = (scale_bytes % 16 == 0);
                auto tma_load_bytes = hidden_bytes + (scale_aligned ? scale_bytes : 0);

                if (elect_one_sync()) {
                    tma_load_1d(tma_buffer, shifted, tma_mbarrier, tma_load_bytes);
                    mbarrier_arrive_and_expect_tx(tma_mbarrier, tma_load_bytes);
                }
                __syncwarp();
                mbarrier_wait(tma_mbarrier, tma_phase);

                auto topk_data_ptr = reinterpret_cast<int*>(shifted + hidden_bytes + scale_bytes + sizeof(SourceMeta));
                auto weight_data_ptr = reinterpret_cast<float*>(topk_data_ptr + num_topk);
                auto* src_data = reinterpret_cast<const __nv_bfloat16*>(tma_buffer);

                const int hidden_int4 = state->hidden_dim * sizeof(__nv_bfloat16) / sizeof(int4);
                const int4* src_i4 = reinterpret_cast<const int4*>(src_data);

                {
                    const int rds_local_expert_end = local_expert_begin + state->num_local_experts;
                    int rds_expert_id = -1;
                    bool rds_is_hit = false;
                    if (lane_id < num_topk) {
                        rds_expert_id = ld_nc_global(topk_data_ptr + lane_id);
                        rds_is_hit = (rds_expert_id >= local_expert_begin && rds_expert_id < rds_local_expert_end);
                    }
                    const unsigned rds_hit_mask = __ballot_sync(0xffffffff, rds_is_hit);
                    int rds_abs_slot = -1;
                    if (rds_is_hit) {
                        int rds_local_expert = rds_expert_id - local_expert_begin;
                        unsigned rds_same = __match_any_sync(rds_hit_mask, rds_local_expert) & rds_hit_mask;
                        int rds_leader = __ffs(rds_same) - 1;
                        int rds_gcount = __popc(rds_same);
                        int rds_rank_in_e = __popc(rds_same & ((1u << lane_id) - 1));
                        int rds_gbase = 0;
                        if (lane_id == rds_leader) {
                            rds_gbase = atomicAdd(&state->expert_token_offsets[rds_local_expert], rds_gcount);
                            if (rds_gbase + rds_gcount > state->expert_count[rds_local_expert]) {
                                printf("MK recv-stage expert slot overflow, rank=%d recv_token=%lld expert=%d slot=%d count=%d\n",
                                       state->rank, (long long)recv_token_idx, rds_expert_id, rds_gbase, state->expert_count[rds_local_expert]);
                                __threadfence_system(); trap();
                            }
                        }
                        rds_gbase = __shfl_sync(rds_same, rds_gbase, rds_leader);
                        int rds_slot = rds_gbase + rds_rank_in_e;
                        rds_abs_slot = state->expert_slot_base[rds_local_expert] + rds_slot;
                        state->pending_slot[recv_token_idx * num_topk + lane_id] = rds_abs_slot;
                    }
                    int4* recv_tokens_i4 = reinterpret_cast<int4*>(state->recv_tokens);
                    unsigned rds_m = rds_hit_mask;
                    while (rds_m) {
                        int rds_hl = __ffs(rds_m) - 1;
                        rds_m &= rds_m - 1;
                        int rds_slot_bcast = __shfl_sync(0xffffffff, rds_abs_slot, rds_hl);
                        int4* rds_dst = recv_tokens_i4 + (int64_t)rds_slot_bcast * hidden_int4;
                        for (int v = lane_id; v < hidden_int4; v += 32)
                            rds_dst[v] = src_i4[v];
                    }
                }

                for (int topk_slot = lane_id; topk_slot < num_topk; topk_slot += 32) {
                    state->pending_topk_idx[recv_token_idx * num_topk + topk_slot] = ld_nc_global(topk_data_ptr + topk_slot);
                    state->pending_topk_weights[recv_token_idx * num_topk + topk_slot] = ld_nc_global(weight_data_ptr + topk_slot);
                }
                if (lane_id == 0)
                    state->pending_meta[recv_token_idx] = meta;
                __syncwarp();
                if (lane_id == 0) {
                    while (producer_tail - ld_acquire_global(&state->pub_ring_head[pub_warp_idx]) >= PUB_RING_DEPTH)
                        __nanosleep(32);
                    st_na_global(&state->pub_ring[pub_warp_idx * PUB_RING_DEPTH + (producer_tail % PUB_RING_DEPTH)], static_cast<int>(recv_token_idx));
                    ++producer_tail;
                    ++producer_batch_count;
                    if (producer_batch_count >= PUB_PRODUCE_BATCH) {
                        st_na_release(&state->pub_ring_tail[pub_warp_idx], producer_tail);
                        producer_batch_count = 0;
                    }
                }
                __syncwarp();

                // Wait TMA to be finished
                tma_store_wait<0>();
                __syncwarp();

            }

            // Move queue
            if (elect_one_sync())
                st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
        }

        __syncwarp();
        if (lane_id == 0) {
            if (producer_batch_count > 0) {
                st_na_release(&state->pub_ring_tail[pub_warp_idx], producer_tail);
                producer_batch_count = 0;
            }
            __threadfence();
            if (logical_stage == num_logical_channels_per_physical - 1)
                st_na_release(&state->recv_warp_done[pub_warp_idx], 1);
            __threadfence_system();  // Ensure prefix/count writes visible before signaling
            atomicAdd(&state->channel_dispatch_done[logical_channel_id], 1);
        }
    }

    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));
    if (thread_id == 0)
        atomicAdd(&state->dispatch_channel_barrier[logical_channel_id], 1);
    if (thread_id == 0) {
        auto start_time = clock64();
        uint64_t wait_polls = 0;
        while (ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]) < 2) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch logical-channel barrier timeout, physical_ch=%d logical_ch=%d count=%d\n",
                       channel_id, logical_channel_id, ld_acquire_sys_global(&state->dispatch_channel_barrier[logical_channel_id]));
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));

    if (thread_id == 0 && dispatch_sm_idx % 2 == 0) {
        const int round_idx = logical_stage;
        __threadfence_system();
        atomicAdd(&state->dispatch_round_barrier[round_idx], 1);
    }
    if (thread_id == 0) {
        const int round_idx = logical_stage;
        auto start_time = clock64();
        uint64_t wait_polls = 0;
        while (ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]) < num_channels) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK dispatch round barrier timeout, physical_ch=%d logical_ch=%d round=%d count=%d need=%d\n",
                       channel_id, logical_channel_id, round_idx,
                       ld_acquire_sys_global(&state->dispatch_round_barrier[round_idx]), num_channels);
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
    }
    asm volatile("barrier.sync 2, %0;" :: "r"((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32));

    }

    }

    asm volatile("barrier.sync 15, %0;" :: "r"(num_threads));


    const int reused_compute_sm_idx = state->num_compute_sms + dispatch_sm_idx;
    const int total_compute_sms_after_dispatch = state->num_compute_sms + state->num_dispatch_sms;
    MK_DISPATCH_REUSED_COMPUTE(
        kDispatchBackwardCompute, kComputeDType, backward_state, sm_id,
        reused_compute_sm_idx, total_compute_sms_after_dispatch, state, smem_buffer);
    return;
}

__device__ __forceinline__ bool timeout_log_once(TeraMoEState* state, int site_id) {
    if (site_id < 0 || site_id >= kTimeoutLogCount)
        return false;
    if (state->timeout_log_counters == nullptr)
        return false;
    int ticket = atomicAdd(&state->timeout_log_counters[site_id], 1);
    return ticket < MK_TIMEOUT_LOG_BUDGET;
}

__device__ __forceinline__ void scheduler_compute_sync(int num_threads) {
    asm volatile("barrier.sync 3, %0;" :: "r"(num_threads));
}


__device__ __forceinline__ int scheduler_compute_all(int predicate, int* shared_result, int num_threads) {
    if (threadIdx.x == 0)
        *shared_result = 1;
    scheduler_compute_sync(num_threads);
    if (!predicate)
        atomicExch(shared_result, 0);
    scheduler_compute_sync(num_threads);
    int result = *shared_result;
    scheduler_compute_sync(num_threads);
    return result;
}

__device__ __forceinline__ int scheduler_compute_all_until_publish_done(
    int predicate, int* shared_result, int num_threads, TeraMoEState* state, int* dispatch_done) {
    if (threadIdx.x == 0) {
        if (*dispatch_done == 0 && ld_acquire_global(state->publish_all_done) != 0) {
            *dispatch_done = 1;
        }
        *shared_result = 1;
    }
    // Reuse scheduler_compute_all's first barrier to broadcast dispatch_done.
    scheduler_compute_sync(num_threads);
    if (*dispatch_done != 0) {
        if (threadIdx.x == 0)
            *shared_result = 0;
    } else if (!predicate) {
        atomicExch(shared_result, 0);
    }
    scheduler_compute_sync(num_threads);
    int result = *shared_result;
    scheduler_compute_sync(num_threads);
    return result;
}

__device__ __forceinline__ void scheduler_publish_task(TeraMoEState* state, int expert_id, int start_slot, int num_tokens, int is_flush, int source) {
    int tail = atomicAdd(state->compute_task_reserve_tail, 1);
    if (tail >= state->max_compute_tasks) {
        printf("MK compute task queue overflow, rank=%d tail=%d max=%d\n", state->rank, tail, state->max_compute_tasks);
        __threadfence_system(); trap();
    }
    state->compute_tasks[tail] = ComputeTask{expert_id, start_slot, num_tokens, is_flush};
    __threadfence();
    while (ld_acquire_global(state->compute_task_tail) != tail)
        __nanosleep(32);
    st_na_release(state->compute_task_tail, tail + 1);
}

__device__ __forceinline__ bool scheduler_try_enqueue_batch(
    TeraMoEState* state, int expert_id, int batch_id, int start_slot, int num_tokens, int is_flush, int source) {
    if (batch_id < 0 || batch_id >= state->max_batches_per_expert) {
        printf("MK scheduler batch id overflow, rank=%d expert=%d batch=%d max=%d\n",
               state->rank, expert_id, batch_id, state->max_batches_per_expert);
        __threadfence_system(); trap();
    }
    int idx = expert_id * state->max_batches_per_expert + batch_id;
    int old_source = atomicCAS(&state->expert_batch_enqueued[idx], 0, source);
    if (old_source != 0)
        return false;
    scheduler_publish_task(state, expert_id, start_slot, num_tokens, is_flush, source);
    return true;
}

__device__ __forceinline__ void scheduler_scan_gather_tokens(TeraMoEState* state) {
    const int tid = threadIdx.x;
    if (tid < GATHER_SCHED_TID_BEGIN)
        return;
    const int gather_tid = tid - GATHER_SCHED_TID_BEGIN;
    const int gather_threads = blockDim.x - GATHER_SCHED_TID_BEGIN;
    if (gather_threads <= 0)
        return;

    const int total_tokens = state->combine_num_tokens;

    for (int token = gather_tid; token < total_tokens; token += gather_threads) {
        if (ld_acquire_global(&state->gather_claimed[token]) != 0)
            continue;
        const int nhits = ld_acquire_global(&state->token_nhits[token]);
        if (nhits <= 1)
            continue;
        const int expected = ld_acquire_global(&state->token_compute_expected[token]);
        const int done = ld_acquire_global(&state->token_done_count[token]);
        if (expected != nhits || done < nhits)
            continue;
        if (atomicCAS(&state->gather_claimed[token], 0, 1) != 0)
            continue;

        const int slot = atomicAdd(state->gather_ready_reserve_tail, 1);
        if (slot >= state->max_total_recv_tokens) {
            printf("MK gather ready queue overflow, rank=%d slot=%d max=%d\n",
                   state->rank, slot, state->max_total_recv_tokens);
            __threadfence_system(); trap();
        }
        __threadfence();
        st_na_global(&state->gather_ready_queue[slot], token);
    }
}

__device__ void compute_scheduler_worker(TeraMoEState* state, int scheduler_id, int num_schedulers) {
    const int tid = threadIdx.x;
    const int num_threads = min(NORMAL_SCHED_THREADS, static_cast<int>(blockDim.x));
    const int num_local_experts = state->num_local_experts;

    // --- Gather scheduler lanes (unchanged) ---
    if (tid >= GATHER_SCHED_TID_BEGIN) {
        while (ld_acquire_global(state->combine_all_done) == 0) {
            scheduler_scan_gather_tokens(state);
            __nanosleep(64);
        }
        return;
    }
    if (tid >= NORMAL_SCHED_THREADS)
        return;

    // --- Shared state for the parallel normal scheduler ---
    __shared__ int s_dispatch_done;
    __shared__ int s_head;           // current drain cursor into ready_batch_queue
    __shared__ int s_rtail;          // snapshot of ready_batch_reserve_tail
    __shared__ int s_done;           // termination flag

    if (tid == 0) {
        s_head = 0;
        s_done = 0;
    }
    scheduler_compute_sync(num_threads);


    while (true) {
        // --- Broadcast dispatch_done ---
        if (tid == 0) {
            s_dispatch_done = (ld_acquire_global(state->publish_all_done) != 0) ? 1 : 0;
        }
        scheduler_compute_sync(num_threads);
        const bool dispatch_done = (s_dispatch_done != 0);

        if (s_done)
            break;

        if (tid == 0) {
            int rtail = ld_acquire_global(state->ready_batch_reserve_tail);
            int head = s_head;
            constexpr int kEnqPublished = -2;
            while (head < rtail && ld_acquire_global(&state->ready_batch_queue[head]) == kEnqPublished)
                head++;
            s_head = head;
            s_rtail = rtail;
        }
        scheduler_compute_sync(num_threads);

        if (tid == 0) {
            constexpr int kEnqPublished = -2;
            const int mbe = state->max_batches_per_expert;
            int head = s_head;
            int rtail = s_rtail;
            while (head < rtail) {
                int enc = ld_acquire_global(&state->ready_batch_queue[head]);
                if (enc == kEnqPublished) { head++; continue; }
                if (enc < 0) break;  // -1 = reserved but not yet written; wait
                int e = enc / mbe;
                int b = enc - e * mbe;
                int start = b * COMPUTE_BATCH_SIZE;
                int ecount = state->expert_count[e];
                int size = min(COMPUTE_BATCH_SIZE, ecount - start);
                scheduler_try_enqueue_batch(state, e, b, start, size,
                                            size < COMPUTE_BATCH_SIZE ? 1 : 0, 1);
                if (ld_acquire_global(&state->expert_enqueue_cursor[e]) < start + size)
                    st_na_release(&state->expert_enqueue_cursor[e], start + size);
                st_na_release(&state->ready_batch_queue[head], kEnqPublished);
                head++;
            }
            s_head = head;
        }
        scheduler_compute_sync(num_threads);

        // --- Final drain after dispatch_done (parallel) ---
        if (dispatch_done) {
            // Wait for all publisher warps to finish
            if (tid == 0) {
                for (int pw = 0; pw < state->num_pub_warps_total; ++pw)
                    while (ld_acquire_global(&state->publish_warp_done[pw]) == 0)
                        __nanosleep(32);
                // Snapshot final queue tail after all publishers are done
                s_rtail = ld_acquire_global(state->ready_batch_reserve_tail);
            }
            scheduler_compute_sync(num_threads);

            // --- Phase 1: All threads drain FIFO entries in parallel ---
            {
                constexpr int kEnqPublished = -2;
                const int mbe = state->max_batches_per_expert;
                const int head = s_head;
                const int rtail = s_rtail;
                const int total = rtail - head;
                for (int idx = tid; idx < total; idx += num_threads) {
                    int qi = head + idx;
                    int enc = ld_acquire_global(&state->ready_batch_queue[qi]);
                    if (enc < 0) continue;  // skip sentinels
                    int e = enc / mbe;
                    int b = enc - e * mbe;
                    int start = b * COMPUTE_BATCH_SIZE;
                    int ecount = state->expert_count[e];
                    int size = min(COMPUTE_BATCH_SIZE, ecount - start);
                    scheduler_try_enqueue_batch(state, e, b, start, size,
                                                size < COMPUTE_BATCH_SIZE ? 1 : 0, 3);
                    atomicMax(&state->expert_enqueue_cursor[e], start + size);
                }
            }
            scheduler_compute_sync(num_threads);

            // --- Phase 2: Safety net per-expert sweep (parallel by expert) ---
            {
                for (int e = tid; e < num_local_experts; e += num_threads) {
                    int count = ld_acquire_global(&state->expert_token_offsets[e]);
                    int cursor = ld_acquire_global(&state->expert_enqueue_cursor[e]);
                    if (count < cursor) count = cursor;
                    while (cursor < count) {
                        int size = min(COMPUTE_BATCH_SIZE, count - cursor);
                        scheduler_try_enqueue_batch(state, e, cursor / COMPUTE_BATCH_SIZE,
                                                    cursor, size, size < COMPUTE_BATCH_SIZE ? 1 : 0, 3);
                        cursor += size;
                    }
                    atomicMax(&state->expert_enqueue_cursor[e], count);
                }
            }
            scheduler_compute_sync(num_threads);

            // --- Phase 3: tid 0 signals completion ---
            if (tid == 0) {
                __threadfence();
                st_na_release(state->compute_enqueue_done, 1);
                s_done = 1;
            }
            scheduler_compute_sync(num_threads);
            break;
        }

        // --- Idle sleep (only if nothing consumed this iteration) ---
        __nanosleep(32);
    }

}

template <ComputeDType kComputeDType, bool kStopAtDispatchDone>
__device__ __forceinline__ void compute_worker_core(

    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    TeraMoEState* state,
    int group_id_base,
    uint8_t* smem_buffer
) {

    const int thread_id = threadIdx.x;
    const int local_warp_id = thread_id / 32;
    if (num_compute_sms <= 0 || compute_sm_idx < 0 || compute_sm_idx >= num_compute_sms)
        return;
    const int local_group_id = compute_sm_idx / COMPUTE_GROUP_SIZE;
    const int group_first_sm_idx = local_group_id * COMPUTE_GROUP_SIZE;
    const int group_size = min(COMPUTE_GROUP_SIZE, num_compute_sms - group_first_sm_idx);
    const int group_id = group_id_base + local_group_id;
    const int num_compute_groups = state->num_compute_groups;
    if (group_size <= 0 || group_id >= num_compute_groups)
        return;
    const int group_sm_idx = compute_sm_idx - group_first_sm_idx;
    const int num_warps_per_sm = blockDim.x / 32;
    const int group_warp_id = group_sm_idx * num_warps_per_sm + local_warp_id;
    const int group_num_warps = group_size * num_warps_per_sm;
    const int group_thread_id = group_sm_idx * blockDim.x + thread_id;
    const int group_num_threads = group_size * blockDim.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int num_topk = state->num_topk;

    // Per-SM global-memory workspace for batched GEMM intermediates.
    // Full batches use M=128. Tail batches use the same path with rows [batch_size,128)
    // zero-filled so WMMA M tiles never read past valid token rows.
    const int padded_m = COMPUTE_BATCH_SIZE;
    // input_buf is unused (gate/up TMA-fed from recv_tokens, bwd save reads recv_tokens),
    // so the input region is dropped from the per-group workspace to save memory.
    const int input_stride = 0;
    const int gu_stride   = padded_m * (2 * intermediate);   // reserved GU scratch [M,2I]
    const int act_stride  = padded_m * intermediate;         // interleaved SwiGLU epilogue result (down A)
    const int down_stride = padded_m * hidden;
    const int gemm_stride = input_stride + gu_stride + act_stride + down_stride;
    __nv_bfloat16* input_buf = state->gemm_workspace + group_id * gemm_stride;
    __nv_bfloat16* gu_buf   = input_buf + input_stride;   // reserved GU scratch [M,2I]
    __nv_bfloat16* up_buf   = gu_buf + gu_stride;         // act = silu(gate)*up*route_w (down-proj A operand)

    auto smem_wmma_buf = reinterpret_cast<float*>(smem_buffer);

    using ComputeUmmaSmemLayout = umma::DgSmemLayout<umma::kDgRunMulticast>;
    constexpr size_t kComputeUmmaBarrierBytes =
        (ComputeUmmaSmemLayout::kNumStages * 3 + ComputeUmmaSmemLayout::kNumEpilogueStages * 2 + 1) *
        sizeof(cutlass::arch::ClusterTransactionBarrier) + sizeof(uint32_t);
    constexpr size_t kComputeUmmaScratchBytes =
        ComputeUmmaSmemLayout::SMEM_CD_SIZE +
        ComputeUmmaSmemLayout::kNumStages *
            (ComputeUmmaSmemLayout::SMEM_A_SIZE_PER_STAGE + ComputeUmmaSmemLayout::SMEM_B_SIZE_PER_STAGE) +
        kComputeUmmaBarrierBytes;
    constexpr size_t kComputeScratchBytes = kComputeUmmaScratchBytes;
    static_assert(kComputeScratchBytes <=
                  (size_t)kNumCombineTMABytesPerForwarderWarp * kNumCombineForwarderWarps,
                  "compute GEMM scratch must fit the launch dynamic-smem budget");

    const int64_t kMetaBase = (int64_t)sm_id * COMPUTE_BATCH_SIZE;
    int* s_recv_token_idx = state->g_meta_recv_idx + kMetaBase;
    int* s_topk_slot = state->g_meta_topk_slot + kMetaBase;
    unsigned char* s_is_single = state->g_meta_is_single + kMetaBase;
    float* s_route_w = state->g_meta_route_w + kMetaBase;

    bool umma_tmem_allocated = false;

    while (true) {
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
            while (true) {
                if constexpr (kStopAtDispatchDone) {
                    int enqueue_done = ld_acquire_global(state->compute_enqueue_done);
                    if (enqueue_done) {
                        int tail = ld_acquire_global(state->compute_task_tail);
                        int head = ld_acquire_global(state->compute_task_head);
                        if (tail == 0 || head * 100 >= tail * state->combine_start_head_percent) {
                            task_idx = -3;
                            break;
                        }
                    }
                }
                int head = ld_acquire_global(state->compute_task_head);
                int tail = ld_acquire_global(state->compute_task_tail);
                if (head >= tail) {
                    if constexpr (!kStopAtDispatchDone) {
                        if (ld_acquire_global(state->compute_enqueue_done))
                            task_idx = -2;
                    }
                    break;
                }
                if (atomicCAS(state->compute_task_head, head, head + 1) == head) {
                    task_idx = head;
                    break;
                }
            }
            st_release_gpu_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, group_size);

        int task_idx = ld_acquire_global(&state->compute_group_task_idx[group_id]);
        if (task_idx == -2 || task_idx == -3) {
            // Dealloc TMEM before exiting the persistent loop (if we ever allocated).
            if (umma_tmem_allocated) {
                char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
                umma::umma_dealloc(cluster_smem);
            }
            break;
        }
        if (task_idx < 0) {
            if (group_sm_idx == 0 && thread_id == 0)
                __nanosleep(128);
            compute_group_sync(state, group_id, group_size);
            continue;
        }

        ComputeTask task = state->compute_tasks[task_idx];
        int expert_id = task.expert_id;
        int start_slot = task.start_slot;
        int batch_size = task.num_tokens;

        constexpr bool kUseUmmaCompute = (MK_COMPUTE_KERNEL != 0);
        static_assert(kUseUmmaCompute, "WMMA compute path removed; MK_COMPUTE_KERNEL must be 1 (1-CTA) or 2 (2-CTA UMMA)");
        const bool use_umma_gateup_for_group = kUseUmmaCompute && group_size == COMPUTE_GROUP_SIZE;
        const bool use_umma_down_for_group = kUseUmmaCompute && group_size == COMPUTE_GROUP_SIZE;
        EP_DEVICE_ASSERT(use_umma_gateup_for_group && use_umma_down_for_group);
        constexpr int kUmmaClusterDim = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
        constexpr int kUmmaClustersPerGroup = COMPUTE_GROUP_SIZE / kUmmaClusterDim;

        constexpr int kGemmMAlign = 128;   // UMMA M-tile granularity (kDgBlockM)
        const int gemm_m = (kUmmaClusterDim == 1)
            ? min(COMPUTE_BATCH_SIZE,
                  (batch_size + kGemmMAlign - 1) / kGemmMAlign * kGemmMAlign)
            : COMPUTE_BATCH_SIZE;

        if constexpr (kUseUmmaCompute) {
            if (batch_size <= COMPUTE_BATCH_SIZE) {
                const umma::InputTmaAtom_t& prefetch_atom = state->group_input_tma[group_id];
                if (local_warp_id == 0) {
                    if (use_umma_gateup_for_group && state->compute_tma != nullptr) {
                        cute::prefetch_tma_descriptor(&prefetch_atom.a);
                        cute::prefetch_tma_descriptor(&state->compute_tma->wgateup[expert_id]);
                        cute::prefetch_tma_descriptor(&prefetch_atom.act_cd);
                    }
                    if (use_umma_down_for_group && state->compute_down_tma != nullptr) {
                        cute::prefetch_tma_descriptor(&prefetch_atom.act_a);
                        cute::prefetch_tma_descriptor(&state->compute_down_tma->wdown[expert_id]);
                        cute::prefetch_tma_descriptor(&prefetch_atom.down_cd);
                    }
                }
            }
        }

        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            int base_offset = state->expert_slot_base[expert_id] + start_slot + i;
            int recv_token = ld_acquire_global(&state->recv_token_source_info[base_offset * 2]);
            int topk_slot = ld_acquire_global(&state->recv_token_source_info[base_offset * 2 + 1]);
            int expected = ld_acquire_global(&state->token_compute_expected[recv_token]);
            s_recv_token_idx[i] = recv_token;
            s_topk_slot[i] = topk_slot;
            s_is_single[i] = static_cast<unsigned char>(expected == 1);
            s_route_w[i] = ld_nc_global(&state->combine_input_topk_weights[recv_token * num_topk + topk_slot]);
            if (group_sm_idx == 0 && recv_token >= 0 && topk_slot >= 0)
                state->fwd_slot_map[recv_token * num_topk + topk_slot] = base_offset;
            s_topk_slot[i] = base_offset;
        }
        for (int i = batch_size + thread_id; i < gemm_m; i += blockDim.x) {
            s_route_w[i] = 0.0f;
            s_recv_token_idx[i] = -1;
            s_topk_slot[i] = -1;
        }
        __syncthreads();

        const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);

        // Expert weight slices
        const __nv_bfloat16* w_gateup = &state->W_gateup[expert_id * 2 * intermediate * hidden];
        const __nv_bfloat16* w_down = &state->W_down[expert_id * hidden * intermediate];

        uint32_t umma_accum_iter = 0;

        if (use_umma_gateup_for_group && state->compute_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma_accum_iter = 0;
            const CUtensorMap* gateup_a_desc = &state->recv_tokens_a_tma;
            uint32_t gateup_a_m_base = (uint32_t)(state->expert_slot_base[expert_id] + start_slot);
            umma::umma_gateup_interleaved_persistent(
                gateup_a_desc,
                &state->compute_tma->wgateup[expert_id],
                &in_atom.act_cd,
                s_route_w,
                gemm_m, intermediate, hidden,
                cluster_in_group, num_clusters,
                cluster_smem, umma_accum_iter,
                reinterpret_cast<cutlass::bfloat16_t*>(state->bwd_preact),
                s_recv_token_idx, s_topk_slot,
                0, 2 * intermediate, batch_size, gateup_a_m_base);
            compute_group_sync(state, group_id, group_size);
        }

        const int slot_base = state->expert_slot_base[expert_id] + start_slot;
        if (use_umma_down_for_group &&
            state->compute_down_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];

            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                // TMEM already allocated; just re-init barriers for the new pass.
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma_accum_iter = 0;  // Reset accumulator phase for fresh barriers.
            umma::DownScatterParams down_scatter{
                reinterpret_cast<int4*>(state->combine_input),
                reinterpret_cast<int4*>(state->compute_output_slot),
                s_recv_token_idx,
                s_is_single,
                slot_base,
                batch_size,
                hidden_int4,
            };
            umma::umma_down_scatter_persistent(
                &in_atom.act_a,
                &state->compute_down_tma->wdown[expert_id],
                &in_atom.down_cd,
                gemm_m, hidden, intermediate,
                cluster_in_group, num_clusters,
                cluster_smem, umma_accum_iter,
                down_scatter);
            compute_group_sync(state, group_id, group_size);
        }

        const bool save_bwd = (state->bwd_fc1_input != nullptr);
        int4* bwd_in_i4 = reinterpret_cast<int4*>(state->bwd_fc1_input);
        const int4* bwd_x_src_i4 = reinterpret_cast<const int4*>(state->recv_tokens);
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int rt = s_recv_token_idx[row];
            if (save_bwd) {
                bwd_in_i4[(int64_t)rt * hidden_int4 + v] =
                    bwd_x_src_i4[(int64_t)(slot_base + row) * hidden_int4 + v];
            }
        }
        __threadfence();
        compute_group_sync(state, group_id, group_size);

        {
            for (int row = group_thread_id; row < batch_size; row += group_num_threads) {
                const int recv_token = s_recv_token_idx[row];
                if (s_is_single[row]) {
                    atomicExch(&state->combine_token_ready[recv_token], 1);
                } else {
                    atomicAdd(&state->token_done_count[recv_token], 1);
                }
            }
            compute_group_sync(state, group_id, group_size);
        }


    }
}

// Forward compute worker wrapper. The core above is shared with combine-precompute
// so the persistent kernel keeps one implementation body and only the entry policy changes.
template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_worker(
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    TeraMoEState* state,
    uint8_t* smem_buffer
) {
    compute_worker_core<kComputeDType, false>(
        sm_id, compute_sm_idx, num_compute_sms, state, 0, smem_buffer);
}

template <int kNumRDMARanks, int kStage>
__device__ void combine_worker(
    int combine_sm_idx,       // 0-based index among combine SMs
    TeraMoEState* state
) {
    using namespace internode;
    using dtype_t = nv_bfloat16;

    const int num_tokens = state->combine_num_tokens;
    const int num_combined_tokens = state->combine_num_combined_tokens;
    const int num_channels = state->num_combine_channels;
    constexpr int num_logical_channels_per_physical = kStage;
    const int num_logical_channels = num_channels * num_logical_channels_per_physical;
    EP_DEVICE_ASSERT(num_channels == state->num_dispatch_channels);  // physical channel counts must match for queue reuse
    const int num_ranks = state->num_ranks;
    constexpr int kNumRDMARanks_C = kNumRDMARanks;
    const int rdma_rank = state->rank / NUM_MAX_NVL_PEERS;

    enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kNumForwarders_C = RdmaCfg::kNumCombineForwarders;
    constexpr int kNumWarpsPerForwarder_C = RdmaCfg::kNumCombineWarpsPerForwarder;
    constexpr int kNumRDMAReceivers_C = RdmaCfg::kNumCombineRDMAReceivers;
    constexpr int kNumTopkRDMARanks_C = RdmaCfg::kNumTopkCombineRDMARanks;

    const auto sm_id = combine_sm_idx;
    const auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;
    const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
    const auto channel_id = sm_id / 2;
    const bool is_forwarder_sm = sm_id % 2 == 1;

    const int num_topk = state->num_topk;
    const int hidden = state->combine_hidden;
    EP_DEVICE_ASSERT(num_topk <= 32);
    EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
    const int hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));
    const int hidden_bytes = hidden_int4 * sizeof(int4);
    const int num_bytes_per_token = get_num_bytes_per_token(hidden_int4, 0, 0, num_topk);

    const auto nvl_rank = state->rank % NUM_MAX_NVL_PEERS;

    auto role_meta = [=]() -> std::pair<WarpRole, int> {
        auto warp_id = thread_id / 32;
        if (not is_forwarder_sm) {
            if (warp_id < NUM_MAX_NVL_PEERS) {
                auto shuffled_warp_id = (warp_id + channel_id) % NUM_MAX_NVL_PEERS;
                return {WarpRole::kNVLSender, shuffled_warp_id};
            } else if (warp_id < kNumForwarders_C) {
                return {WarpRole::kRDMAReceiver, warp_id - NUM_MAX_NVL_PEERS};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        } else {
            if (warp_id < kNumForwarders_C) {
                auto shuffled_warp_id = (warp_id + channel_id) % kNumForwarders_C;
                return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
            } else {
                return {WarpRole::kCoordinator, 0};
            }
        }
    }();
    auto warp_role = role_meta.first;
    auto warp_id = role_meta.second;

    const int num_max_rdma_chunked_send_tokens = state->num_max_combine_rdma_chunked_send_tokens;
    const int num_max_rdma_chunked_recv_tokens = state->num_max_combine_rdma_chunked_recv_tokens;
    const int num_max_nvl_chunked_send_tokens = state->num_max_combine_nvl_chunked_send_tokens;
    const int num_max_nvl_chunked_recv_tokens = state->num_max_combine_nvl_chunked_recv_tokens;
    auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks_C;

    // Combine input/output pointers
    int4* combined_x = state->combined_x;
    float* combined_topk_weights = state->combined_topk_weights;
    const float* topk_weights = state->combine_topk_weights;
    const int* combined_rdma_head_base = state->combined_rdma_head;     // Will be normalized in-place
    int* combined_nvl_head_global_base = state->combined_nvl_head;      // Will be normalized in-place
    const SourceMeta* src_meta = reinterpret_cast<const SourceMeta*>(state->combine_src_meta);
    const int* rdma_channel_prefix_matrix = state->combine_rdma_channel_prefix_matrix;
    const int* rdma_rank_prefix_sum = state->combine_rdma_rank_prefix_sum;
    const int* gbl_channel_prefix_matrix = state->combine_gbl_channel_prefix_matrix;
    void* rdma_buffer_ptr = state->combine_rdma_buffer_ptr;
    void** buffer_ptrs = state->combine_buffer_ptrs;

    constexpr int kCombineNumTMAStages = 2;

    if (state->rdma_reuse_prelude_enable) {
        const int num_rdma_ranks_local = num_ranks / NUM_MAX_NVL_PEERS;
        if (combine_sm_idx == 0 && thread_id < 32) {
            const int ndc = state->num_dispatch_channels;

            // 1. Dispatch send finished (dispatch_channel_barrier[lc] reaches 2 only after the RDMA
            //    sender coordinator + forwarder passed the post-send CTA barrier). Lane-distributed.
            for (int lc = lane_id; lc < num_logical_channels; lc += 32) {
                while (ld_acquire_sys_global(&state->dispatch_channel_barrier[lc]) < 2)
                    __nanosleep(64);
            }
            __syncwarp();

            // 2. Quiet dispatch QPs (sender ch + forwarder ch+ndc) to same-nvl remote peers.
            //    Task t = ((dr * ndc) + ch) * 2 + is_fwd -> unique (dst_pe, qp), one lane each.
            const int qp_tasks = num_rdma_ranks_local * ndc * 2;
            for (int t = lane_id; t < qp_tasks; t += 32) {
                const int dr = t / (ndc * 2);
                const int rem = t % (ndc * 2);
                const int ch = rem >> 1;
                const int is_fwd = rem & 1;
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_quiet(dst_pe, is_fwd ? (ch + ndc) : ch);
            }
            __syncwarp();

            // 3. Publish dispatch-quiet-done into every same-nvl peer's mailbox (and locally).
            if (lane_id == 0)
                st_release_sys_global(&state->rdma_reuse_dispatch_quiet_done[rdma_rank], 1);
            for (int dr = lane_id; dr < num_rdma_ranks_local; dr += 32) {
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_rma_p(&state->rdma_reuse_dispatch_quiet_done[rdma_rank], 1, dst_pe, 0);
            }
            __syncwarp();

            // 4. Wait until all same-nvl RDMA peers finished dispatch quiet (lane-distributed).
            for (int src = lane_id; src < num_rdma_ranks_local; src += 32) {
                while (ld_acquire_sys_global(&state->rdma_reuse_dispatch_quiet_done[src]) == 0)
                    __nanosleep(64);
            }
            __syncwarp();

            // 5. Clean this GPU's combine RDMA metadata (data-after head/tail/meta int region).
            //    Mirrors get_rdma_clean_meta(combine_hidden_int4, 0, 0, num_topk, ...) used by the
            //    host combine cached_notify. num_bytes_per_token already == combine layout bytes.
            {
                const int recv_tokens = num_max_rdma_chunked_recv_tokens;  // combine recv capacity
                const long long clean_offset =
                    (long long)num_bytes_per_token * recv_tokens * num_rdma_ranks_local * 2 * num_logical_channels
                    / (long long)sizeof(int);
                const int clean_count =
                    (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks_local * 2 * num_logical_channels;
                int* clean_p = static_cast<int*>(state->combine_rdma_buffer_ptr);
                for (int i = lane_id; i < clean_count; i += 32)
                    clean_p[clean_offset + i] = 0;
            }
            __syncwarp();
            if (lane_id == 0)
                __threadfence_system();  // make the metadata clean visible before publishing clear-done
            __syncwarp();

            // 6. Publish combine-clear-done into every same-nvl peer's mailbox (and locally).
            if (lane_id == 0)
                st_release_sys_global(&state->rdma_reuse_combine_clear_done[rdma_rank], 1);
            for (int dr = lane_id; dr < num_rdma_ranks_local; dr += 32) {
                if (dr == rdma_rank) continue;
                const int dst_pe = translate_dst_rdma_rank<kLowLatencyMode>(dr, nvl_rank);
                nvshmemi_ibgda_rma_p(&state->rdma_reuse_combine_clear_done[rdma_rank], 1, dst_pe, 0);
            }
            __syncwarp();

            // 7. Wait until all same-nvl RDMA peers finished the combine metadata clean.
            for (int src = lane_id; src < num_rdma_ranks_local; src += 32) {
                while (ld_acquire_sys_global(&state->rdma_reuse_combine_clear_done[src]) == 0)
                    __nanosleep(64);
            }
            __syncwarp();

            // Release the rest of the combine SMs.
            if (lane_id == 0)
                st_release_sys_global(state->rdma_reuse_prelude_done, 1);
        }
        while (ld_acquire_sys_global(state->rdma_reuse_prelude_done) == 0)
            __nanosleep(64);
        __syncthreads();
    }

    for (int logical_stage = 0; logical_stage < num_logical_channels_per_physical; ++logical_stage) {
        const int logical_channel_id = channel_id * num_logical_channels_per_physical + logical_stage;
    const int* combined_rdma_head = combined_rdma_head_base + logical_channel_id * state->combine_rdma_head_stride;
    int* combined_nvl_head_base = combined_nvl_head_global_base + logical_channel_id * state->combine_nvl_head_stride;
    int combine_nvl_sender_cached_channel_head_idx = 0;
    int combine_nvl_sender_cached_channel_tail_idx = 0;
    int combine_forwarder_cached_nvl_channel_tail_idx = 0;
    int combine_coordinator_last_rdma_head = 0;
    int combine_coordinator_last_nvl_head[kNumRDMARanks_C] = {0};
    uint32_t combine_forwarder_tma_phase[kCombineNumTMAStages] = {0};

    if (!is_forwarder_sm && warp_role == WarpRole::kCoordinator) {
        // Step 1: Wait for this channel's dispatch to complete
        if (lane_id == 0) {
            auto start_time = clock64();
            while (ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]) < NUM_MAX_NVL_PEERS) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK combine normalizer timeout waiting for channel_dispatch_done[%d] (physical_ch=%d), got %d\n",
                           logical_channel_id, channel_id, ld_acquire_sys_global(&state->channel_dispatch_done[logical_channel_id]));
                    __threadfence_system(); trap();
                }
                __nanosleep(32);
            }
        }
        __syncwarp();

        // Step 2: Normalize combined_rdma_head for this channel's token range.
        // This mirrors original DeepEP: token_idx remains the global/original token index,
        // but the head array itself is private to logical_channel_id.
        {
            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);
            if (lane_id < kNumRDMARanks_C) {
                int rdma_prefix_idx = lane_id * num_logical_channels + logical_channel_id;
                int rdma_ch_count = state->combine_rdma_channel_token_count[rdma_prefix_idx];
                int last_head = 1 << 25;
                for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
                    auto current_head = __ldg(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                    bool is_in_src_rdma = current_head >= 0;
                    int last_before = last_head;
                    int normalized_head = current_head;
                    if (current_head < 0) {
                        normalized_head = -last_head - 1;
                        const_cast<int*>(combined_rdma_head)[token_idx * kNumRDMARanks_C + lane_id] = normalized_head;
                    } else {
                        last_head = current_head;
                    }
                }
            }
        }
        __syncwarp();

        // Step 3: Normalize combined_nvl_head for the RDMA compact token span.
        // Match original DeepEP's TMA batch normalize instead of per-token system-acquire loads.
        {
            constexpr int tma_batch_size = kNumCombineTMABytesPerSenderWarp - static_cast<int>(sizeof(uint64_t));
            constexpr int num_head_bytes_per_token = sizeof(int) * NUM_MAX_NVL_PEERS;
            constexpr int num_tokens_per_batch = tma_batch_size / num_head_bytes_per_token;
            EP_STATIC_ASSERT(num_head_bytes_per_token % 16 == 0, "num_head_bytes_per_token should be divisible by 16");

            extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
            auto tma_buffer = smem_tma_buffer;
            auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + tma_batch_size);
            uint32_t tma_phase = 0;
            if (elect_one_sync()) {
                tma_store_wait<0>();
                mbarrier_init(tma_mbarrier, 1);
                fence_barrier_init();
                EP_DEVICE_ASSERT(tma_batch_size + static_cast<int>(sizeof(uint64_t)) <= kNumCombineTMABytesPerSenderWarp);
            }
            __syncwarp();

            for (int dst_rdma_rank = 0; dst_rdma_rank < kNumRDMARanks_C; ++dst_rdma_rank) {
                int rdma_prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
                int channel_end = ld_nc_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
                int channel_count = ld_nc_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
                int channel_start = channel_end - channel_count;
                int rank_shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
                int rdma_token_start = rank_shift + channel_start;
                int rdma_token_end = rank_shift + channel_end;
                int nvl_head_capacity = state->combine_nvl_head_stride / NUM_MAX_NVL_PEERS;
                EP_DEVICE_ASSERT(channel_count >= 0 and channel_end >= channel_start);
                EP_DEVICE_ASSERT(rdma_token_start >= 0 and rdma_token_end >= rdma_token_start and rdma_token_end <= nvl_head_capacity);

                int last_head = 1 << 25;
                for (int batch_end_idx = rdma_token_end; batch_end_idx > rdma_token_start; batch_end_idx -= num_tokens_per_batch) {
                    int batch_start_idx = max(rdma_token_start, batch_end_idx - num_tokens_per_batch);
                    int batch_bytes = (batch_end_idx - batch_start_idx) * num_head_bytes_per_token;

                    if (elect_one_sync()) {
                        tma_load_1d(tma_buffer,
                                    combined_nvl_head_base + batch_start_idx * NUM_MAX_NVL_PEERS,
                                    tma_mbarrier,
                                    batch_bytes);
                        mbarrier_arrive_and_expect_tx(tma_mbarrier, batch_bytes);
                    }
                    mbarrier_wait(tma_mbarrier, tma_phase);
                    __syncwarp();

                    for (int token_idx = batch_end_idx - 1; token_idx >= batch_start_idx; --token_idx) {
                        if (lane_id < NUM_MAX_NVL_PEERS) {
                            auto current_head = reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id];
                            int normalized_head = current_head;
                            if (current_head < 0) {
                                normalized_head = -last_head - 1;
                                reinterpret_cast<int*>(tma_buffer)[(token_idx - batch_start_idx) * NUM_MAX_NVL_PEERS + lane_id] = normalized_head;
                            } else {
                                last_head = current_head;
                            }
                        }
                    }
                    tma_store_fence();
                    __syncwarp();

                    if (elect_one_sync())
                        tma_store_1d(tma_buffer,
                                     combined_nvl_head_base + batch_start_idx * NUM_MAX_NVL_PEERS,
                                     batch_bytes);
                    tma_store_wait<0>();
                    __syncwarp();
                }
            }
        }

        // Step 4: Signal normalization complete
        tma_store_wait<0>();
        __threadfence_system();
        if (lane_id == 0) {
            st_release_sys_global(&state->channel_normalized[logical_channel_id], 1);
        }
    }

    // All warps (except the sender-SM coordinator that just signaled) wait for normalization
    if (warp_role != WarpRole::kCoordinator || is_forwarder_sm) {
        if (lane_id == 0) {
            auto start_time = clock64();
            while (ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]) == 0) {
                if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                    printf("MK combine warp timeout waiting for channel_normalized[%d] (physical_ch=%d), role=%d\n",
                           logical_channel_id, channel_id, (int)warp_role);
                    __threadfence_system(); trap();
                }
                __nanosleep(32);
            }
        }
        __syncwarp();
    }


    if (warp_role == WarpRole::kNVLSender) {
        // ========== NVL Sender (direct port from internode.cu L1784-1922) ==========
        const auto dst_nvl_rank = warp_id;

        auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
        auto nvl_channel_x = AsymBuffer<uint8_t>(dst_buffer_ptr,
                                                 num_max_nvl_chunked_recv_tokens * num_bytes_per_token,
                                                 NUM_MAX_NVL_PEERS,
                                                 logical_channel_id,
                                                 num_logical_channels,
                                                 nvl_rank)
                                 .advance_also(local_buffer_ptr);
        auto nvl_channel_head = AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, dst_nvl_rank)
                                    .advance_also(dst_buffer_ptr);
        auto nvl_channel_tail = AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);

        // TMA stuffs
        extern __shared__ __align__(1024) uint8_t smem_tma_buffer[];
        auto tma_buffer = smem_tma_buffer + dst_nvl_rank * kNumCombineTMABytesPerSenderWarp;
        auto tma_mbarrier = reinterpret_cast<uint64_t*>(tma_buffer + num_bytes_per_token);
        uint32_t tma_phase = 0;
        if (elect_one_sync()) {
            mbarrier_init(tma_mbarrier, 1);
            fence_barrier_init();
            EP_DEVICE_ASSERT(num_bytes_per_token + sizeof(uint64_t) <= kNumCombineTMABytesPerSenderWarp);
        }
        __syncwarp();

        // Get tasks for each RDMA lane in the DeepEP compact recv-token namespace.
        // In the fused kernel, the next prefix entry may belong to a later logical
        // channel that has not run yet, so use the current channel's explicit count.
        int token_start_idx = 0, token_end_idx = 0;
        int nvl_sender_prefix_idx = -1;
        int nvl_sender_count = 0;
        if (lane_id < kNumRDMARanks_C) {
            nvl_sender_prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_logical_channels + logical_channel_id;
            token_start_idx = ld_nc_global(gbl_channel_prefix_matrix + nvl_sender_prefix_idx);
            nvl_sender_count = ld_nc_global(state->combine_gbl_channel_token_count + nvl_sender_prefix_idx);
            token_end_idx = token_start_idx + nvl_sender_count;
            EP_DEVICE_ASSERT(token_start_idx >= 0 and nvl_sender_count >= 0 and token_end_idx <= num_tokens);
        }
        __syncwarp();

        auto& cached_channel_head_idx = combine_nvl_sender_cached_channel_head_idx;
        auto& cached_channel_tail_idx = combine_nvl_sender_cached_channel_tail_idx;

        // Iterate over all tokens and send by chunks
        int current_rdma_idx = channel_id % kNumRDMARanks_C;
        while (true) {
            if (__all_sync(0xffffffff, token_start_idx >= token_end_idx))
                break;

            bool is_lane_ready = false;
            auto start_time = clock64();
            while (true) {
                int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                is_lane_ready = lane_id < kNumRDMARanks_C and token_start_idx < token_end_idx and
                    num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
                if (__any_sync(0xffffffff, is_lane_ready))
                    break;

                if (lane_id < kNumRDMARanks_C and token_start_idx < token_end_idx)
                    cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

                if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks_C) {
                    int live_head = ld_volatile_global(nvl_channel_head.buffer() + lane_id);
                    int live_tail = ld_acquire_sys_global(nvl_channel_tail.buffer() + lane_id);
                    int used_slots = cached_channel_tail_idx - cached_channel_head_idx;
                    int free_slots = num_max_nvl_chunked_recv_tokens_per_rdma - used_slots;
                    int blocked_head_token = cached_channel_head_idx < cached_channel_tail_idx ?
                        gbl_channel_prefix_matrix[nvl_sender_prefix_idx] + cached_channel_head_idx : -1;
                    int next_send_token = token_start_idx < token_end_idx ? token_start_idx : -1;
                    int last_sent_token = cached_channel_tail_idx > 0 ?
                        gbl_channel_prefix_matrix[nvl_sender_prefix_idx] + cached_channel_tail_idx - 1 : -1;
                    printf("MK combine NVL sender timeout, ch: %d, logical_ch: %d, RDMA: %d, nvl: %d, dst NVL: %d, lane: %d, "
                           "head=%d, live_head=%d, tail=%d, live_tail=%d, used=%d, free=%d, capacity=%d, slots_needed=%d, "
                           "prefix_idx=%d, sender_base=%d, sender_count=%d, range=[%d,%d), "
                           "waiting_next_token=%d, blocked_head_token=%d, last_sent_token=%d, "
                           "head_ptr=%p, tail_ptr=%p, channel_normalized=%d\n",
                           channel_id, logical_channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id,
                           cached_channel_head_idx, live_head, cached_channel_tail_idx, live_tail,
                           used_slots, free_slots, num_max_nvl_chunked_recv_tokens_per_rdma, num_max_nvl_chunked_send_tokens,
                           nvl_sender_prefix_idx, gbl_channel_prefix_matrix[nvl_sender_prefix_idx], nvl_sender_count,
                           gbl_channel_prefix_matrix[nvl_sender_prefix_idx], token_end_idx,
                           next_send_token, blocked_head_token, last_sent_token,
                           (void*)(nvl_channel_head.buffer() + lane_id), (void*)(nvl_channel_tail.buffer() + lane_id),
                           ld_acquire_sys_global(&state->channel_normalized[logical_channel_id]));
                    __threadfence_system(); trap();
                }
            }

            for (int i = 0; i < kNumRDMARanks_C; ++i) {
                current_rdma_idx = (current_rdma_idx + 1) % kNumRDMARanks_C;
                if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
                    continue;

                auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
                int producer_token_end_idx = __shfl_sync(0xffffffff, token_end_idx, current_rdma_idx);
                int num_tokens_in_chunk = min(num_max_nvl_chunked_send_tokens, producer_token_end_idx - static_cast<int>(token_idx));

                for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
                    __syncwarp();
                    int dst_slot_idx = 0;
                    if (lane_id == current_rdma_idx) {
                        dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
                        dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
                    }
                    dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

                    auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * num_bytes_per_token;
                    tma_store_wait<0>();
                    auto gather_wait_start = clock64();
                    int ready = 0;
                    while (true) {
                        if (lane_id == 0)
                            ready = ld_acquire_global(&state->combine_token_ready[token_idx]);
                        ready = __shfl_sync(0xffffffff, ready, 0);
                        if (ready == 1)
                            break;

                        if (lane_id == 0 && clock64() - gather_wait_start > NUM_TIMEOUT_CYCLES) {
                            if (timeout_log_once(state, kTimeoutLogComputeReady)) {
                                int nh = ld_acquire_global(&state->token_nhits[token_idx]);
                                int expected = ld_acquire_global(&state->token_compute_expected[token_idx]);
                                int done = ld_acquire_global(&state->token_done_count[token_idx]);
                                int ready_snapshot = ld_acquire_global(&state->combine_token_ready[token_idx]);
                                printf("MK combine gather-semaphore timeout, rank=%d token=%lld nh=%d expected=%d done=%d ready=%d\n",
                                       state->rank, (long long)token_idx, nh, expected, done, ready_snapshot);
                            }
                            __threadfence_system(); trap();
                        }
                        __nanosleep(32);
                    }
                    {
                        const int4* token_out_i4 = reinterpret_cast<const int4*>(state->combine_input);
                        // No per-slot wait here. combine_token_ready is the only readiness signal.
                        __syncwarp();
                        if (lane_id == 0) {
                            tma_load_1d(tma_buffer,
                                        token_out_i4 + token_idx * hidden_int4,
                                        tma_mbarrier, hidden_bytes, false);
                            mbarrier_arrive_and_expect_tx(tma_mbarrier, hidden_bytes);
                        }
                        __syncwarp();
                        mbarrier_wait(tma_mbarrier, tma_phase);
                    }
                    __syncwarp();

                    if (lane_id == 0)
                        *reinterpret_cast<SourceMeta*>(tma_buffer + hidden_bytes) = ld_nc_global(src_meta + token_idx);

                    if (lane_id < num_topk)
                        *reinterpret_cast<float*>(tma_buffer + hidden_bytes + sizeof(SourceMeta) + lane_id * sizeof(float)) =
                            ld_nc_global(topk_weights + token_idx * num_topk + lane_id);

                    tma_store_fence();
                    __syncwarp();
                    if (elect_one_sync())
                        tma_store_1d(tma_buffer, shifted_x_buffers, num_bytes_per_token, false);
                }
                lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
            }

            tma_store_wait<0>();
            __syncwarp();
            if (lane_id < kNumRDMARanks_C and is_lane_ready) {
                st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
            }
        }
    } else {
        // RDMA symmetric layout
        auto rdma_channel_data = SymBuffer<int8_t>(
            rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_token, kNumRDMARanks_C, logical_channel_id, num_logical_channels);
        auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, logical_channel_id, num_logical_channels);
        auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks_C, logical_channel_id, num_logical_channels);

        // NVL layouts
        void* local_nvl_buffer = buffer_ptrs[nvl_rank];
        void* nvl_buffers[NUM_MAX_NVL_PEERS];
        #pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
            nvl_buffers[i] = buffer_ptrs[i];
        auto nvl_channel_x =
            AsymBuffer<uint8_t>(
                local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_bytes_per_token, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels)
                .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
        auto nvl_channel_head =
            AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels, nvl_rank)
                .advance_also(local_nvl_buffer);
        auto nvl_channel_tail = AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks_C, NUM_MAX_NVL_PEERS, logical_channel_id, num_logical_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

        // Shared memory for warp synchronization
        __shared__ volatile int forwarder_nvl_head[kNumForwarders_C][NUM_MAX_NVL_PEERS];
        __shared__ volatile bool forwarder_retired[kNumForwarders_C];
        __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers_C][kNumRDMARanks_C];
        __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers_C];
        auto sync_forwarder_smem = [=]() { asm volatile("barrier.sync 0, %0;" ::"r"((kNumForwarders_C + 1) * 32)); };
        auto sync_rdma_receiver_smem = [=]() { asm volatile("barrier.sync 1, %0;" ::"r"((kNumRDMAReceivers_C + 1) * 32)); };

        if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
            // ========== NVL+RDMA Forwarder (internode.cu L1955-2144) ==========
            const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder_C;
            const auto sub_warp_id = warp_id % kNumWarpsPerForwarder_C;
            auto send_buffer =
                dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank) : rdma_channel_data.send_buffer(dst_rdma_rank);
            auto sync_large_warp = [=]() {
                if (kNumWarpsPerForwarder_C == 1) {
                    __syncwarp();
                } else {
                    asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder_C * 32));
                }
            };

            // TMA stuffs
            constexpr int kNumStages = kCombineNumTMAStages;
            constexpr int kNumTMALoadBytes = sizeof(int4) * 32;
            constexpr int kNumTMABufferBytesPerStage = kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1) + 16;
            constexpr int kNumTMABytesPerForwarderWarp = kNumStages * kNumTMABufferBytesPerStage;
            EP_STATIC_ASSERT(kNumTMABytesPerForwarderWarp <= kNumCombineTMABytesPerForwarderWarp,
                             "combine forwarder TMA buffer is not large enough");

            extern __shared__ __align__(1024) uint8_t smem_buffer[];
            auto smem_ptr = smem_buffer + warp_id * kNumCombineTMABytesPerForwarderWarp;
            auto tma_mbarrier = [=](const int& i) {
                return reinterpret_cast<uint64_t*>(smem_ptr + i * kNumTMABufferBytesPerStage + kNumTMALoadBytes * (NUM_MAX_NVL_PEERS + 1));
            };
            auto& tma_phase = combine_forwarder_tma_phase;
            // Logical channels reuse the same per-warp TMA scratch space. Make sure
            // the previous channel has no outstanding TMA store before reinitializing
            // the mbarriers for this channel.
            tma_store_wait<0>();
            if (lane_id < kNumStages) {
                mbarrier_init(tma_mbarrier(lane_id), 32);
                fence_barrier_init();
            }
            __syncwarp();

            nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_bytes_per_token);
            nvl_channel_head.advance(dst_rdma_rank);
            nvl_channel_tail.advance(dst_rdma_rank);

            lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
            sync_forwarder_smem();

            auto& cached_nvl_channel_tail_idx = combine_forwarder_cached_nvl_channel_tail_idx;
            // RDMA prefix entries for later logical channels may not be available yet.
            // Use this channel's explicit count to recover its local start.
            int rdma_prefix_idx = dst_rdma_rank * num_logical_channels + logical_channel_id;
            int channel_end = ld_nc_global(rdma_channel_prefix_matrix + rdma_prefix_idx);
            int num_tokens_to_combine = ld_nc_global(state->combine_rdma_channel_token_count + rdma_prefix_idx);
            int channel_start = channel_end - num_tokens_to_combine;
            int num_tokens_prefix = channel_start + (dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1]);
            int* logical_combined_nvl_head = combined_nvl_head_base + num_tokens_prefix * NUM_MAX_NVL_PEERS;

            for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine; token_start_idx += num_max_rdma_chunked_send_tokens) {
                auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
                auto num_chunked_tokens = token_end_idx - token_start_idx;
                auto start_time = clock64();
                while (sub_warp_id == 0 and lane_id == 0) {
                    int num_used_slots = token_start_idx - ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                    if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens)
                        break;

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        int cur_head = ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank));
                        printf("MK combine forwarder (RDMA check) timeout, ch: %d, dst_rdma: %d, "
                               "logical_token_start=%d, rdma_token_start=%d, cur_head=%d, capacity=%d, needed=%d\n",
                               channel_id, dst_rdma_rank,
                               token_start_idx, token_start_idx, cur_head, num_max_rdma_chunked_recv_tokens, num_chunked_tokens);
                        __threadfence_system(); trap();
                    }
                }
                sync_large_warp();

                for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx; token_idx += kNumWarpsPerForwarder_C) {
                    // Read normalized head (original DeepEP logic)
                    int expected_head = -1;
                    if (lane_id < NUM_MAX_NVL_PEERS) {
                        int lane_raw_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                        expected_head = lane_raw_head;
                        // Normalized semantics: negative = -(next_valid_head)-1, positive = actual head
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head);
                    }

                    start_time = clock64();
                    // Wait for NVL tail to advance past expected_head (original DeepEP logic)
                    while (cached_nvl_channel_tail_idx <= expected_head) {
                        cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

                        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
                            int head0 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS);
                            int head1 = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + 1);
                            int lane_raw_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
                            int local_head = ld_nc_global(logical_combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + nvl_rank);
                            int tail0 = ld_acquire_sys_global(nvl_channel_tail.buffer(0));
                            int tail1 = ld_acquire_sys_global(nvl_channel_tail.buffer(1));
                            int global_token_idx = num_tokens_prefix + token_idx;
                            int sender_prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_logical_channels + logical_channel_id;
                            int sender_count = state->combine_gbl_channel_token_count[sender_prefix_idx];
                            int sender_base = gbl_channel_prefix_matrix[sender_prefix_idx];
                            int sender_local_head = expected_head >= 0 ? expected_head : -1;
                            int sender_token = sender_local_head >= 0 ? sender_base + sender_local_head : -1;
                            int local_dst_prefix_idx = (dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank) * num_logical_channels + logical_channel_id;
                            int local_dst_count = state->combine_gbl_channel_token_count[local_dst_prefix_idx];
                            int local_dst_base = gbl_channel_prefix_matrix[local_dst_prefix_idx];
                            int local_dst_token = sender_local_head >= 0 ? local_dst_base + sender_local_head : -1;
                            printf("MK combine forwarder (NVL check) timeout, rank=%d rdma_rank=%d nvl_rank=%d ch=%d logical_ch=%d dst_rdma=%d token=%d global_token=%d lane=%d lane_raw_head=%d local_head=%d expected_head=%d cached_tail=%d head0=%d head1=%d tail0=%d tail1=%d sender_prefix_idx=%d sender_token=%d sender_range=[%d,%d) sender_count=%d local_dst_prefix_idx=%d local_dst_token=%d local_dst_range=[%d,%d) local_dst_count=%d nvl_tail_ptr=%p nvl_x_lane_base=%p sub_warp=%d rank_prefix=%d tokens=%d\n",
                                   state->rank, rdma_rank, nvl_rank, channel_id, logical_channel_id,
                                   dst_rdma_rank, token_idx, global_token_idx, lane_id,
                                   lane_raw_head, local_head, expected_head,
                                   cached_nvl_channel_tail_idx, head0, head1, tail0, tail1,
                                   sender_prefix_idx, sender_token, sender_base, sender_base + sender_count, sender_count,
                                   local_dst_prefix_idx, local_dst_token, local_dst_base, local_dst_base + local_dst_count, local_dst_count,
                                   (void*)nvl_channel_tail.buffer(lane_id), (void*)nvl_channel_x.buffer(lane_id), sub_warp_id,
                                   num_tokens_prefix, num_tokens_to_combine);
                            __threadfence_system(); trap();
                        }
                    }

                    // Combine current token
                    auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
                    void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_token;
                    auto get_addr_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                        return reinterpret_cast<int4*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token) +
                            hidden_int4_idx;
                    };
                    auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
                        return ld_nc_global(reinterpret_cast<float*>(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * num_bytes_per_token +
                                                                     hidden_bytes + sizeof(SourceMeta)) +
                                            topk_idx);
                    };

                    // [IMPORTANT]
                    // when hidden_size < 1024, using tma combine_token may cause hang
                    // make sure you combine_token correctly when using high parallelism
                    combine_token<NUM_MAX_NVL_PEERS, false, dtype_t, NUM_MAX_NVL_PEERS, true, kNumStages, kNumTMALoadBytes>(
                        expected_head >= 0,
                        expected_head,
                        lane_id,
                        hidden_int4,
                        num_topk,
                        static_cast<int4*>(shifted),
                        reinterpret_cast<float*>(static_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
                        nullptr,
                        nullptr,
                        num_max_nvl_chunked_recv_tokens_per_rdma,
                        get_addr_fn,
                        recv_tw_fn,
                        smem_ptr,
                        tma_phase);

                    if (lane_id < NUM_MAX_NVL_PEERS)
                        expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                                          : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
                }
                sync_large_warp();

                // Issue RDMA send
                    if (sub_warp_id == kNumWarpsPerForwarder_C - 1) {
                        auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
                        const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_token;
                        const auto dst_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.recv_buffer(rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        const auto src_ptr =
                            reinterpret_cast<uint64_t>(rdma_channel_data.send_buffer(dst_rdma_rank) + rdma_slot_idx * num_bytes_per_token);
                        auto* rdma_tail_ptr = rdma_channel_tail.buffer(rdma_rank);

                        if (dst_rdma_rank != rdma_rank) {
                            nvshmemi_ibgda_put_nbi_warp<true>(dst_ptr,
                                                              src_ptr,
                                                              num_bytes_per_msg,
                                                              translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                              logical_channel_id + num_logical_channels,
                                                              lane_id,
                                                              0);
                        } else {
                            rdma_tail_ptr = rdma_channel_tail.buffer(dst_rdma_rank);
                            memory_fence();
                        }

                        __syncwarp();
                        if (elect_one_sync()) {
                            nvshmemi_ibgda_amo_nonfetch_add(rdma_tail_ptr,
                                                            num_chunked_tokens,
                                                            translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                            logical_channel_id + num_logical_channels,
                                                            dst_rdma_rank == rdma_rank);
                        }
                    }

            }

            __syncwarp();
            // Set INT_MAX before retiring so Coordinator won't be blocked by this warp
            if (lane_id < NUM_MAX_NVL_PEERS)
                forwarder_nvl_head[warp_id][lane_id] = std::numeric_limits<int>::max();
            if (elect_one_sync())
                forwarder_retired[warp_id] = true;

        } else if (warp_role == WarpRole::kRDMAReceiver) {
            // ========== RDMA Receiver (internode.cu L2145-2222) ==========
            lane_id < kNumRDMARanks_C ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
            lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
            sync_rdma_receiver_smem();

            int token_start_idx, token_end_idx;
            get_channel_task_range(num_combined_tokens, num_logical_channels, logical_channel_id, token_start_idx, token_end_idx);

            int cached_channel_tail_idx = 0;
            for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers_C) {
                int expected_head = -1;
                if (lane_id < kNumRDMARanks_C) {
                    expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                    // Normalized semantics: negative = -(next_valid_head)-1, positive = actual wait head.
                    int normalized_wait_head = expected_head < 0 ? -expected_head - 1 : expected_head;
                    rdma_receiver_rdma_head[warp_id][lane_id] = normalized_wait_head;

                }

                auto start_time = clock64();
                // Wait for RDMA tail (normalized heads: always wait, negative heads skip via large value)
                while (cached_channel_tail_idx <= expected_head) {
                    cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

                    if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                        if (timeout_log_once(state, kTimeoutLogCombineRdmaReceiver)) {
                            int tail0 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(0)));
                            int tail1 = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(1)));
                            int head0 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(0)));
                            int head1 = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(1)));
                            int rdma_ch_count = state->combine_rdma_channel_token_count[lane_id * num_logical_channels + logical_channel_id];
                            printf("MK combine RDMA receiver timeout, rank=%d rdma_rank=%d nvl_rank=%d block=%d combine_sm=%d ch=%d logical_ch=%d warp=%d token=%lld token_range=[%d,%d) expected_head=%d cached_tail=%d tail_snapshot=%d rdma_tail_ptr=%p lane=%d tail0=%d tail1=%d head0=%d head1=%d rdma_ch_count=%d tail0_ptr=%p tail1_ptr=%p head0_ptr=%p head1_ptr=%p rdma_recv_buffer_lane=%p combined_rdma_head_addr=%p\n",
                                   state->rank, rdma_rank, nvl_rank, blockIdx.x, sm_id, channel_id, logical_channel_id,
                                   warp_id, (long long)token_idx,
                                   token_start_idx, token_end_idx, expected_head, cached_channel_tail_idx,
                                   static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id))),
                                   (void*)rdma_channel_tail.buffer(lane_id), lane_id, tail0, tail1, head0, head1,
                                   rdma_ch_count,
                                   rdma_channel_tail.buffer(0), rdma_channel_tail.buffer(1),
                                   rdma_channel_head.buffer(0), rdma_channel_head.buffer(1),
                                   rdma_channel_data.recv_buffer(lane_id),
                                   combined_rdma_head + token_idx * kNumRDMARanks_C + lane_id);
                        }
                        __threadfence_system(); trap();
                    }

                }
                __syncwarp();

                auto get_addr_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4* {
                    return reinterpret_cast<int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) + slot_idx * num_bytes_per_token) +
                        hidden_int4_idx;
                };
                auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
                    return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                                       slot_idx * num_bytes_per_token + hidden_bytes + sizeof(SourceMeta)) +
                                        topk_idx);
                };
                uint32_t dummy_tma_phases[2];
                combine_token<kNumRDMARanks_C, true, dtype_t, kNumTopkRDMARanks_C, false, 2>(
                    expected_head >= 0,
                    expected_head,
                    lane_id,
                    hidden_int4,
                    num_topk,
                    combined_x + token_idx * hidden_int4,
                    combined_topk_weights + token_idx * num_topk,
                    nullptr,
                    nullptr,
                    num_max_rdma_chunked_recv_tokens,
                    get_addr_fn,
                    recv_tw_fn,
                    nullptr,
                    dummy_tma_phases);
            }

            __syncwarp();
            // Set INT_MAX before retiring so Coordinator won't be blocked by this warp
            if (lane_id < kNumRDMARanks_C)
                rdma_receiver_rdma_head[warp_id][lane_id] = std::numeric_limits<int>::max();
            if (elect_one_sync())
                rdma_receiver_retired[warp_id] = true;

        } else {
            // ========== Coordinator (internode.cu L2223-2269) ==========
            is_forwarder_sm ? sync_forwarder_smem() : sync_rdma_receiver_smem();
            const auto num_warps_per_rdma_rank = kNumForwarders_C / kNumRDMARanks_C;

            auto& last_rdma_head = combine_coordinator_last_rdma_head;
            auto& last_nvl_head = combine_coordinator_last_nvl_head;
            int dst_rdma_rank = lane_id < kNumRDMARanks_C ? lane_id : 0;
            int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;

            while (true) {
                if (not is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers_C or rdma_receiver_retired[lane_id]))
                    break;
                if (is_forwarder_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders_C or forwarder_retired[lane_id]))
                    break;

                if (not is_forwarder_sm) {
                    int min_head = std::numeric_limits<int>::max();
                    #pragma unroll
                    for (int i = 0; i < kNumRDMAReceivers_C; ++i)
                        if (not rdma_receiver_retired[i])
                            min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
                    if (min_head != std::numeric_limits<int>::max() and min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and
                        lane_id < kNumRDMARanks_C) {
                        nvshmemi_ibgda_amo_nonfetch_add(rdma_channel_head.buffer(rdma_rank),
                                                        min_head - last_rdma_head,
                                                        translate_dst_rdma_rank<kLowLatencyMode>(dst_rdma_rank, nvl_rank),
                                                        logical_channel_id + num_logical_channels,
                                                        dst_rdma_rank == rdma_rank);
                        last_rdma_head = min_head;
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < kNumRDMARanks_C; ++i) {
                        int min_head = std::numeric_limits<int>::max();
                        #pragma unroll
                        for (int j = 0; j < num_warps_per_rdma_rank; ++j)
                            if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);

                        if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and lane_id < NUM_MAX_NVL_PEERS) {
                            st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
                        }
                    }
                }

                __nanosleep(NUM_WAIT_NANOSECONDS);
            }
        }
    }

    __syncthreads();
    if (thread_id == 0) {
        atomicAdd(&state->combine_channel_barrier[logical_channel_id], 1);
    }
    if (thread_id == 0) {
        auto start_time = clock64();
        while (ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]) < 2) {
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("MK combine logical-channel barrier timeout, rank=%d combine_sm_idx=%d physical_ch=%d logical_ch=%d is_forwarder_sm=%d count=%d\n",
                       state->rank, combine_sm_idx, channel_id, logical_channel_id,
                       static_cast<int>(is_forwarder_sm), ld_acquire_sys_global(&state->combine_channel_barrier[logical_channel_id]));
                __threadfence_system(); trap();
            }
            __nanosleep(32);
        }
    }
    __syncthreads();
    }

    __syncthreads();
    if (thread_id == 0) {
        int finished = atomicAdd(state->combine_done_count, 1) + 1;
        if (finished == state->num_combine_sms) {
            __threadfence();
            atomicExch(state->combine_all_done, 1);
        }
    }
}

template <ComputeDType kComputeDType>
__device__ void combine_precompute_worker(
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    TeraMoEState* state,
    uint8_t* smem_buffer
) {
    const int post_group_count =
        (state->num_compute_sms + state->num_dispatch_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    compute_worker_core<kComputeDType, true>(
        sm_id, combine_sm_idx, num_combine_sms, state, post_group_count, smem_buffer);
}

__device__ __forceinline__ void gather_accum_bf162(float2& acc, __nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t packed = *reinterpret_cast<uint32_t*>(&value);
    uint16_t lo = static_cast<uint16_t>(packed & 0xffffu);
    uint16_t hi = static_cast<uint16_t>(packed >> 16);
    asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.x) : "h"(lo));
    asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.y) : "h"(hi));
#else
    float2 v = __bfloat1622float2(value);
    acc.x += v.x;
    acc.y += v.y;
#endif
}

__device__ __forceinline__ void gather_accum_int4(float2* acc, int4 raw) {
    const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int p = 0; p < 4; ++p)
        gather_accum_bf162(acc[p], bv2[p]);
}

__device__ void gather_worker(TeraMoEState* state, int gather_sm_idx) {
    const int tid = threadIdx.x;
    const int total_tokens = state->combine_num_tokens;
    if (total_tokens == 0) return;

    constexpr int kGatherBatch = 64;
    __shared__ int s_head_base;
    __shared__ int s_batch_count;
    __shared__ int s_has_task;
    __shared__ int s_tokens[kGatherBatch];

    while (true) {
        if (tid == 0) {
            s_has_task = 0;
            s_head_base = 0;
            s_batch_count = 0;
            while (true) {
                const int tail = ld_acquire_global(state->gather_ready_reserve_tail);
                const int head = ld_acquire_global(state->gather_ready_head);
                if (head < tail) {
                    int batch = tail - head;
                    if (batch > kGatherBatch) batch = kGatherBatch;
                    if (atomicCAS(state->gather_ready_head, head, head + batch) == head) {
                        s_head_base = head;
                        s_batch_count = batch;
                        s_has_task = 1;
                        break;
                    }
                } else {
                    if (ld_acquire_global(state->combine_all_done) != 0)
                        break;
                    __nanosleep(64);
                }
            }
        }
        __syncthreads();
        if (!s_has_task)
            break;

        const int head_base = s_head_base;
        const int batch_count = s_batch_count;

        // Resolve the batch's token ids, spinning on the -1 sentinel until each producer's
        // store lands (reserve counter is bumped before the token write).
        for (int batch_idx = tid; batch_idx < batch_count; batch_idx += blockDim.x) {
            int token = ld_acquire_global(&state->gather_ready_queue[head_base + batch_idx]);
            while (token < 0) {
                __nanosleep(32);
                token = ld_acquire_global(&state->gather_ready_queue[head_base + batch_idx]);
            }
            s_tokens[batch_idx] = token;
        }
        __syncthreads();

        constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);
        constexpr int kBfloat162PerInt4 = kElemsPerInt4 / 2;
        const int hidden = state->combine_hidden;
        const int hidden_int4 = hidden / kElemsPerInt4;
        const int num_topk = state->num_topk;
        int4* slot_base_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        int4* token_out_i4 = reinterpret_cast<int4*>(state->combine_input);

        constexpr int kGatherChunkInt4 = 128;
        constexpr int kGatherVecsPerLane = kGatherChunkInt4 / 32;
        const int warp_id = tid >> 5;
        const int lane_id = tid & 31;
        const int num_warps = (blockDim.x + 31) >> 5;
        const int chunks_per_token = (hidden_int4 + kGatherChunkInt4 - 1) / kGatherChunkInt4;
        const int total_work = batch_count * chunks_per_token;

        for (int work = warp_id; work < total_work; work += num_warps) {
            const int batch_idx = work / chunks_per_token;
            const int chunk = work - batch_idx * chunks_per_token;
            const int token_idx = s_tokens[batch_idx];
            const int nhits = ld_acquire_global(&state->token_nhits[token_idx]);
            const int chunk_base = chunk * kGatherChunkInt4;
            const int chunk_end = min(chunk_base + kGatherChunkInt4, hidden_int4);
            float2 acc[kGatherVecsPerLane][kBfloat162PerInt4];

            #pragma unroll
            for (int j = 0; j < kGatherVecsPerLane; ++j) {
                #pragma unroll
                for (int p = 0; p < kBfloat162PerInt4; ++p)
                    acc[j][p] = make_float2(0.0f, 0.0f);
            }

            for (int k = 0; k < nhits; ++k) {
                int slot = state->token_slot_list[token_idx * num_topk + k];
                #pragma unroll
                for (int j = 0; j < kGatherVecsPerLane; ++j) {
                    const int vi = chunk_base + lane_id + j * 32;
                    if (vi < chunk_end) {
                        int4 raw;
                        asm volatile("ld.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
                            : "l"(slot_base_i4 + (int64_t)slot * hidden_int4 + vi));
                        gather_accum_int4(acc[j], raw);
                    }
                }
            }

            #pragma unroll
            for (int j = 0; j < kGatherVecsPerLane; ++j) {
                const int vi = chunk_base + lane_id + j * 32;
                if (vi < chunk_end) {
                    int4 packed;
                    __nv_bfloat162* pv2 = reinterpret_cast<__nv_bfloat162*>(&packed);
                    #pragma unroll
                    for (int p = 0; p < kBfloat162PerInt4; ++p)
                        pv2[p] = __float22bfloat162_rn(acc[j][p]);
                    token_out_i4[(int64_t)token_idx * hidden_int4 + vi] = packed;
                }
            }
        }

        __syncthreads();
        // System-scope fence: combine_input[token] is consumed by the combine sender via
        // TMA on a different SM, which can bypass L1 and observe stale L2 under a device
        // fence. Publish readiness only after the reduce is globally visible.
        __threadfence();

        for (int batch_idx = tid; batch_idx < batch_count; batch_idx += blockDim.x) {
            const int token_idx = s_tokens[batch_idx];
            atomicExch(&state->combine_token_ready[token_idx], 1);
        }
        __syncthreads();
    }
}

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
__global__ void __launch_bounds__(MegaKernelRdmaConfig<kNumRDMARanks>::kMegaKernelNumThreads, 1) teramoe_fused_forward_kernel(
    TeraMoEState* state
) {
    const int sm_id = blockIdx.x;
    const int num_dispatch_sms = state->num_dispatch_sms;
    const int num_combine_sms = state->num_combine_sms;
    const int num_compute_sms = state->num_compute_sms;

    // Role-specific workers reuse the single dynamic shared-memory allocation.
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Determine SM role based on blockIdx.x
    // Layout: [Dispatch] [Combine] [Scheduler] [Compute groups] [Gather]
    SmRole role;
    int role_idx;

    const int compute_begin = num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS;
    const int gather_begin = compute_begin + num_compute_sms;
    const int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
    if (sm_id < num_dispatch_sms) {
        role = SmRole::kDispatch;
        role_idx = sm_id;
    } else if (sm_id < num_dispatch_sms + num_combine_sms) {
        role = SmRole::kCombine;
        role_idx = sm_id - num_dispatch_sms;
    } else if (sm_id < compute_begin) {
        role = SmRole::kScheduler;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms;
    } else if (sm_id < gather_begin) {
        role = SmRole::kCompute;
        role_idx = sm_id - compute_begin;
    } else {
        role = SmRole::kGather;
        role_idx = sm_id - gather_begin;
    }

    switch (role) {
        case SmRole::kDispatch:
            dispatch_worker<kNumRDMARanks, kStage, kComputeDType, false>(sm_id, role_idx, state);
            break;

        case SmRole::kCombine:
            combine_precompute_worker<kComputeDType>(sm_id, role_idx, num_combine_sms, state, smem_buffer);
            combine_worker<kNumRDMARanks, kStage>(role_idx, state);
            break;

        case SmRole::kScheduler:
            compute_scheduler_worker(state, role_idx, COMPUTE_SCHEDULER_SMS);
            break;

        case SmRole::kCompute:
            MK_FORWARD_COMPUTE_WORKER(
                kComputeDType, sm_id, role_idx,
                total_compute_sms_after_dispatch, state, smem_buffer);
            break;

        case SmRole::kGather:
            gather_worker(state, role_idx);
            break;

        default:
            break;
    }
}

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
static void launch_teramoe_fused_forward_case(
    TeraMoEState* device_state,
    const TeraMoEState& host_state,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kThreads = RdmaCfg::kMegaKernelNumThreads;
    const int num_ranks = host_state.num_ranks;

    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(teramoe_fused_forward_kernel<kNumRDMARanks, kStage, kComputeDType>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }

    constexpr int num_gather_sms = GATHER_SMS;
    const int launch_total_sms = total_sms + num_gather_sms;
    EP_HOST_ASSERT(MK_COMPUTE_CLUSTER_DIM == 1 || (launch_total_sms % 2 == 0 && "launch_total_sms must be even for MK_COMPUTE_KERNEL=2 cluster_dim=2"));
    EP_HOST_ASSERT(host_state.num_combine_sms % 2 == 0);
    EP_HOST_ASSERT(host_state.num_combine_sms > 0);
    EP_HOST_ASSERT(kThreads >= (RdmaCfg::kNumCombineForwarders + 1) * 32);

#ifndef DISABLE_SM90_FEATURES
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = launch_total_sms;
    cfg.blockDim = kThreads;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;

    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = MK_COMPUTE_CLUSTER_DIM;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, teramoe_fused_forward_kernel<kNumRDMARanks, kStage, kComputeDType>, device_state));
#else
    teramoe_fused_forward_kernel<kNumRDMARanks, kStage, kComputeDType><<<launch_total_sms, kThreads, smem_size, stream>>>(device_state);
#endif
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_teramoe_fused_forward_impl(
    TeraMoEState* device_state,
    const TeraMoEState* host_state,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    TeraMoEState copied_host_state;
    const TeraMoEState* launcher_host_state = host_state;
    if (launcher_host_state == nullptr) {
        CUDA_CHECK(cudaMemcpy(&copied_host_state, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
        launcher_host_state = &copied_host_state;
    }
    const int num_ranks = launcher_host_state->num_ranks;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(compute_dtype == ComputeDType::kBF16 &&
                   "megakernel forward currently supports BF16 only");

#define MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, kStage, kComputeDType) \
    launch_teramoe_fused_forward_case<kNumRDMARanks, kStage, kComputeDType>(device_state, *launcher_host_state, total_sms, smem_size, stream); \
    break

#define MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE(kNumRDMARanks, kComputeDType) \
    switch (stage) { \
        case 1: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 1, kComputeDType); \
        case 2: MEGAKERNEL_LAUNCH_STAGE_CASE(kNumRDMARanks, 2, kComputeDType); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel stage"); \
    } \
    break

#define MEGAKERNEL_LAUNCH_CASE(kNumRDMARanks) \
    switch (compute_dtype) { \
        case ComputeDType::kBF16: MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE(kNumRDMARanks, ComputeDType::kBF16); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel compute dtype"); \
    } \
    break

    SWITCH_RDMA_RANKS(MEGAKERNEL_LAUNCH_CASE);

#undef MEGAKERNEL_LAUNCH_CASE
#undef MEGAKERNEL_LAUNCH_CASE_WITH_DTYPE
#undef MEGAKERNEL_LAUNCH_STAGE_CASE

}


// Use PyTorch's CUDA caching allocator for long-lived fused-kernel state buffers.
static inline cudaError_t teramoe_caching_alloc(void** pp, size_t nbytes) {
    *pp = (nbytes == 0) ? nullptr : c10::cuda::CUDACachingAllocator::raw_alloc(nbytes);
    return cudaSuccess;
}

static inline cudaError_t teramoe_caching_free(void* ptr) {
    if (ptr != nullptr)
        c10::cuda::CUDACachingAllocator::raw_delete(ptr);
    return cudaSuccess;
}

struct FusedFillDesc {
    void* ptr;
    size_t bytes;
    uint32_t word;
};

// Simple parallel fill: one block per desc, all threads stride over words.
// Host splits large buffers into <=CHUNK_SIZE descs so blocks are balanced.
// When expert_count_mapped is provided, block 0 also builds the compact expert
// layout in the same launch so we avoid a separate H2D or a separate kernel.
__global__ void fused_fill_kernel(
    const FusedFillDesc* descs,
    int ndescs,
    const int* expert_count_mapped,
    int* expert_slot_base,
    int* expert_count,
    int num_local_experts) {
    if (expert_count_mapped != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
        int slot_base = 0;
        for (int le = 0; le < num_local_experts; ++le) {
            int cnt = expert_count_mapped[le];
            if (cnt < 0)
                cnt = 0;
            expert_count[le] = cnt;
            expert_slot_base[le] = slot_base;
            slot_base += cnt;
        }
    }

    const int desc_idx = blockIdx.x;
    if (desc_idx >= ndescs)
        return;
    const FusedFillDesc desc = descs[desc_idx];
    auto* words = reinterpret_cast<uint32_t*>(desc.ptr);
    const size_t nwords = desc.bytes / sizeof(uint32_t);
    for (size_t i = threadIdx.x; i < nwords; i += blockDim.x)
        words[i] = desc.word;
}

static void initialize_megakernel_launch_state(const TeraMoEState& state, cudaStream_t stream);

namespace {
struct TmaCacheKey {
    const void* w_gateup;
    const void* w_down;
    const void* gemm_ws;
    int num_local_experts;
    int hidden_dim;
    int intermediate_dim;
    int num_compute_groups;
    int compute_batch_size;
};
struct TmaCacheEntry {
    TmaCacheKey key{};
    bool valid = false;
    umma::ComputeTmaAtoms* compute_tma = nullptr;
    umma::InputTmaAtom_t* group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* compute_down_tma = nullptr;
};
constexpr int kTmaCacheCap = 8;
static thread_local TmaCacheEntry s_tma_cache[kTmaCacheCap];
static thread_local int s_tma_cache_next = 0;

constexpr size_t kMegakernelArenaChunkBytes = 8ull * 1024ull * 1024ull;

struct MegakernelArenaAllocator {
    std::vector<void*> chunks;
    void* current_chunk = nullptr;
    size_t current_chunk_bytes = 0;
    size_t offset = 0;

    cudaError_t alloc(void** pp, size_t nbytes) {
        if (nbytes == 0) {
            *pp = nullptr;
            return cudaSuccess;
        }
        constexpr size_t kAlign = NUM_BUFFER_ALIGNMENT_BYTES;
        auto align_up = [](size_t v, size_t a) {
            return (v + a - 1) / a * a;
        };
        nbytes = align_up(nbytes, kAlign);
        size_t aligned_offset = align_up(offset, kAlign);
        if (current_chunk == nullptr || aligned_offset + nbytes > current_chunk_bytes) {
            if (chunks.size() >= kMegakernelArenaChunkCap)
                return cudaErrorMemoryAllocation;
            const size_t chunk_bytes = align_up(std::max(nbytes, kMegakernelArenaChunkBytes), kAlign);
            void* chunk = nullptr;
            cudaError_t err = teramoe_caching_alloc(&chunk, chunk_bytes);
            if (err != cudaSuccess)
                return err;
            chunks.push_back(chunk);
            current_chunk = chunk;
            current_chunk_bytes = chunk_bytes;
            offset = 0;
            aligned_offset = 0;
        } else {
            offset = aligned_offset;
        }
        *pp = static_cast<void*>(static_cast<char*>(current_chunk) + aligned_offset);
        offset = aligned_offset + nbytes;
        return cudaSuccess;
    }
};

static inline void store_arena_chunks(const MegakernelArenaAllocator& arena, void** dst, int* count) {
    EP_HOST_ASSERT(arena.chunks.size() <= kMegakernelArenaChunkCap);
    *count = static_cast<int>(arena.chunks.size());
    for (int i = 0; i < *count; ++i)
        dst[i] = arena.chunks[i];
}

static inline cudaError_t free_arena_chunks(void** chunks, int count) {
    for (int i = 0; i < count; ++i) {
        if (chunks[i] != nullptr) {
            cudaError_t err = teramoe_caching_free(chunks[i]);
            if (err != cudaSuccess)
                return err;
            chunks[i] = nullptr;
        }
    }
    return cudaSuccess;
}

}  // anonymous namespace

// Host-side functions cannot use the COMPUTE_BATCH_SIZE macro (which dereferences `state`).
// Undefine it and use a local variable / function parameter instead.
#undef COMPUTE_BATCH_SIZE

TeraMoEState* allocate_teramoe_fused_state(
    // --- Dispatch input data (from PyTorch tensors) ---
    const int4* x,
    const uint32_t* x_scales,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    // --- Dispatch prefix matrices (from notify_dispatch) ---
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    // --- Buffer infrastructure ---
    void* rdma_buffer_ptr,
    void** buffer_ptrs,
    void** combine_buffer_ptrs,
    // --- Dimensions ---
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
    // --- Buffer sizing ---
    int dispatch_num_max_rdma_chunked_send_tokens,
    int dispatch_num_max_rdma_chunked_recv_tokens,
    int dispatch_num_max_nvl_chunked_send_tokens,
    int dispatch_num_max_nvl_chunked_recv_tokens,
    int combine_num_max_rdma_chunked_send_tokens,
    int combine_num_max_rdma_chunked_recv_tokens,
    int combine_num_max_nvl_chunked_send_tokens,
    int combine_num_max_nvl_chunked_recv_tokens,
    // --- Expert weights ---
    const __nv_bfloat16* W_gateup,
    const __nv_bfloat16* W_down,
    ComputeDType compute_dtype,
    const void* W_gateup_fp8,
    const void* W_down_fp8,
    const uint32_t* W_gateup_fp8_sf,
    const uint32_t* W_down_fp8_sf,
    // --- SM config ---
    int num_dispatch_sms,
    int num_forwarder_sms,
    int num_compute_sms,
    int num_combine_sms,
    int num_logical_channels,
    // --- Max token budget ---
    int max_tokens_per_expert,
    int max_total_recv_tokens,
    // --- Buffer sizes for combine mirror ---
    int64_t num_rdma_bytes,
    int64_t num_nvl_bytes,
    const int* host_expert_count,
    const int* device_expert_count_mapped,
    // Optional caller-owned buffers. Non-null pointers are borrowed by the state.
    __nv_bfloat16* external_bwd_fc1_input,
    __nv_bfloat16* external_bwd_preact,
    int* external_fwd_slot_map,
    int4* external_combined_x,
    float* external_combined_topk_weights,
    TeraMoEState* host_state_out,
    int* rdma_reuse_dispatch_quiet_done,
    int* rdma_reuse_combine_clear_done,
    int rdma_reuse_prelude_enable,
    int compute_batch_size,
    int combine_start_head_percent
) {
    MegakernelArenaAllocator persistent_arena;
    MegakernelArenaAllocator transient_arena;
    MegakernelArenaAllocator* arena = &persistent_arena;
#define cudaMalloc(pp, n) arena->alloc(reinterpret_cast<void**>(pp), (n))
    auto cache_alloc = [](auto** pp, size_t nbytes) -> cudaError_t {
        return teramoe_caching_alloc(reinterpret_cast<void**>(pp), nbytes);
    };
    struct MkFillRec { void* ptr; int byte_value; size_t bytes; };
    std::vector<MkFillRec> mk_fill_recs;
    auto record_deferred_fill = [&](void* ptr, int byte_value, size_t bytes) -> cudaError_t {
        if (bytes != 0)
            mk_fill_recs.push_back({ptr, byte_value, bytes});
        return cudaSuccess;
    };
#define cudaMemset(p, v, n) record_deferred_fill((p), (v), (n))

    // Allocate workspace buffers on device
    int* expert_recv_count;
    int* timeout_log_counters;
    __nv_bfloat16* recv_tokens;
    int* expert_token_offsets;
    int* recv_token_source_info;
    float* g_meta_route_w;
    int* g_meta_recv_idx;
    int* g_meta_topk_slot;
    unsigned char* g_meta_is_single;
    int* compute_group_barrier;
    int* compute_group_phase;
    ComputeTask* compute_tasks;
    int* compute_task_head;
    int* compute_task_tail;
    int* compute_task_reserve_tail;
    int* compute_enqueue_done;
    int* scheduler_done_count;
    int* priority_scheduler_done;
    int* expert_enqueue_cursor;
    int* compute_group_task_idx;
    __nv_bfloat16* combine_input;
    float* combine_input_topk_weights;
    internode::SourceMeta* combine_input_src_meta;
    __nv_bfloat16* gemm_workspace;
    float* output_accum;
    int* send_rdma_head;
    int* send_nvl_head;
    int* recv_rdma_channel_prefix_matrix;
    int* recv_gbl_channel_prefix_matrix;

    // Mirror DeepEP host-side launch invariants before allocating state. These
    // protect the producer/consumer queue geometry used by dispatch and combine.
    EP_HOST_ASSERT(compute_batch_size == 1024 || compute_batch_size == 2048 || compute_batch_size == 4096);
    EP_HOST_ASSERT(combine_start_head_percent >= 0 && combine_start_head_percent <= 100);
    EP_HOST_ASSERT(static_cast<int64_t>(num_scales) * scale_hidden_stride < std::numeric_limits<int>::max());
    EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
    EP_HOST_ASSERT(hidden_int4 > 0);
    EP_HOST_ASSERT(compute_dtype == ComputeDType::kBF16 && "megakernel FP8 is not supported");
    EP_HOST_ASSERT(W_gateup_fp8 == nullptr && W_down_fp8 == nullptr &&
                   W_gateup_fp8_sf == nullptr && W_down_fp8_sf == nullptr &&
                   "megakernel FP8 is not supported");
    EP_HOST_ASSERT(num_topk <= 32);
    EP_HOST_ASSERT(num_experts > 0 and num_local_experts > 0);
    EP_HOST_ASSERT(num_ranks > 0 and num_experts % num_ranks == 0);
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_logical_channels * 2 > 3);
    int kNumRDMARanks = num_ranks / NUM_MAX_NVL_PEERS;

    EP_HOST_ASSERT(dispatch_num_max_rdma_chunked_send_tokens > 0 and dispatch_num_max_rdma_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(dispatch_num_max_nvl_chunked_send_tokens > 0 and dispatch_num_max_nvl_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(dispatch_num_max_rdma_chunked_recv_tokens % dispatch_num_max_rdma_chunked_send_tokens == 0);
    EP_HOST_ASSERT(dispatch_num_max_nvl_chunked_send_tokens < dispatch_num_max_nvl_chunked_recv_tokens);

    auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / kNumRDMARanks, 1);
    int num_forwarder_warps = kNumRDMARanks * num_warps_per_forwarder;
    EP_HOST_ASSERT(kNumRDMARanks <= kNumCombineForwarderWarps);
    EP_HOST_ASSERT(num_forwarder_warps > NUM_MAX_NVL_PEERS and num_forwarder_warps % kNumRDMARanks == 0);
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens % kNumRDMARanks == 0);
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens / kNumRDMARanks >
                   std::max(combine_num_max_rdma_chunked_send_tokens, combine_num_max_nvl_chunked_send_tokens));
    EP_HOST_ASSERT(combine_num_max_nvl_chunked_recv_tokens / kNumRDMARanks - num_warps_per_forwarder >= combine_num_max_nvl_chunked_send_tokens);
    EP_HOST_ASSERT(combine_num_max_rdma_chunked_send_tokens >= num_warps_per_forwarder);

    // Signaling
    CUDA_CHECK(cudaMalloc(&expert_recv_count, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_recv_count, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&timeout_log_counters, kTimeoutLogCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(timeout_log_counters, 0, kTimeoutLogCount * sizeof(int)));

    // Per-expert slot layout — indexed as [expert_slot_base[le] + slot].
    // Compact packing: when host_expert_count is provided (forward path), each expert's
    // region is sized by its real received-token count and regions are packed contiguously
    // via an exclusive prefix sum (total = Σ count). When null (backward / legacy path),
    // falls back to the fixed le*max_tokens_per_expert layout (total = num_local_experts*max_tpe).
    // The forward path also passes device_expert_count_mapped so fused_fill_kernel can
    // materialize expert_count / expert_slot_base in the same launch that clears buffers.
    int* expert_slot_base;
    int* expert_count;
    CUDA_CHECK(cudaMalloc(&expert_slot_base, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_count, num_local_experts * sizeof(int)));
    size_t total_expert_slots = 0;
    if (host_expert_count != nullptr) {
        for (int le = 0; le < num_local_experts; ++le) {
            const int cnt = host_expert_count[le];
            EP_HOST_ASSERT(cnt >= 0 && cnt <= max_tokens_per_expert);
            total_expert_slots += static_cast<size_t>(cnt);
        }
    } else {
        for (int le = 0; le < num_local_experts; ++le) {
            total_expert_slots += static_cast<size_t>(max_tokens_per_expert);
        }
    }
    if (total_expert_slots == 0) total_expert_slots = 1;  // avoid zero-size allocations
    if (device_expert_count_mapped == nullptr) {
        std::vector<int> h_expert_slot_base(num_local_experts);
        std::vector<int> h_expert_count(num_local_experts);
        size_t slot_base = 0;
        for (int le = 0; le < num_local_experts; ++le) {
            const int cnt = (host_expert_count != nullptr) ? host_expert_count[le] : max_tokens_per_expert;
            h_expert_slot_base[le] = static_cast<int>(slot_base);
            h_expert_count[le] = cnt;
            slot_base += static_cast<size_t>(cnt);
        }
        cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
        CUDA_CHECK(cudaMemcpyAsync(expert_slot_base, h_expert_slot_base.data(),
                              num_local_experts * sizeof(int), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(expert_count, h_expert_count.data(),
                              num_local_experts * sizeof(int), cudaMemcpyHostToDevice, s));
    }

    int* expert_slot_ready;
    CUDA_CHECK(cudaMalloc(&expert_slot_ready, total_expert_slots * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_slot_ready, 0, total_expert_slots * sizeof(int)));
    size_t recv_tokens_bytes = total_expert_slots * hidden_dim * sizeof(__nv_bfloat16);
    // recv_tokens gets COMPUTE_BATCH_SIZE extra rows of tail padding so the gate/up
    // GEMM's TMA A-load (m_base + gemm_m, gemm_m rounded up to the 128-row UMMA tile)
    // for the last expert's tail batch never reads past the buffer. Only the tail
    // padding is zeroed (real slots are written by dispatch; padding rows are masked
    // by valid_rows in the epilogue but we keep them defined to avoid NaN/Inf reads).
    size_t recv_tokens_padded_bytes =
        ((size_t)total_expert_slots + compute_batch_size) * hidden_dim * sizeof(__nv_bfloat16);
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&recv_tokens, recv_tokens_padded_bytes));
    CUDA_CHECK(cudaMemset(recv_tokens + total_expert_slots * hidden_dim, 0,
                          (size_t)compute_batch_size * hidden_dim * sizeof(__nv_bfloat16)));
    arena = &persistent_arena;

    CUDA_CHECK(cudaMalloc(&expert_token_offsets, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_token_offsets, 0, num_local_experts * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_token_source_info, total_expert_slots * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_token_source_info, 0xff, total_expert_slots * 2 * sizeof(int)));  // Init to -1

    // Compute state
    int base_compute_groups = num_compute_sms / COMPUTE_GROUP_SIZE;
    int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
    int post_group_count = (total_compute_sms_after_dispatch + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    int num_combine_compute_groups = (num_combine_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    int num_compute_groups = post_group_count + num_combine_compute_groups;
    EP_HOST_ASSERT(base_compute_groups > 0);
    EP_HOST_ASSERT(num_compute_groups > 0);
    EP_HOST_ASSERT(num_compute_sms == base_compute_groups * COMPUTE_GROUP_SIZE);
    // Per-block (per-SM) compute-task metadata in GMEM (moved out of smem to free GEMM
    // scratch). Indexed [sm_id * COMPUTE_BATCH_SIZE + row]; each SM gathers/reads its own
    // slice exactly like the old per-SM smem (block-local, __syncthreads suffices — no
    // cross-SM sync). Sized by the full launch grid so any sm_id is in bounds.
    {
        size_t launch_total_sms = (size_t)num_dispatch_sms + num_combine_sms + num_compute_sms
                                  + COMPUTE_SCHEDULER_SMS + GATHER_SMS;
        size_t meta_rows = launch_total_sms * compute_batch_size;
        CUDA_CHECK(cudaMalloc(&g_meta_route_w, meta_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&g_meta_recv_idx, meta_rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g_meta_topk_slot, meta_rows * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&g_meta_is_single, meta_rows * sizeof(unsigned char)));
    }
    CUDA_CHECK(cudaMalloc(&compute_group_barrier, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_barrier, 0, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_phase, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_phase, 0, num_compute_groups * sizeof(int)));

    int max_compute_tasks = num_local_experts * (max_tokens_per_expert / compute_batch_size + 2);
    CUDA_CHECK(cudaMalloc(&compute_tasks, (size_t)max_compute_tasks * sizeof(ComputeTask)));
    CUDA_CHECK(cudaMalloc(&compute_task_head, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_task_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_task_reserve_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_task_reserve_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_enqueue_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_enqueue_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scheduler_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(scheduler_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&priority_scheduler_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(priority_scheduler_done, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&expert_enqueue_cursor, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_enqueue_cursor, 0, num_local_experts * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&compute_group_task_idx, num_compute_groups * sizeof(int)));
    CUDA_CHECK(cudaMemset(compute_group_task_idx, 0xff, num_compute_groups * sizeof(int)));

    // Dedicated gather SM task queue state.
    int* token_done_count;
    int* gather_claimed;
    int* combine_token_ready;
    int* gather_ready_queue;
    int* gather_ready_head;
    int* gather_ready_tail;
    int* gather_ready_reserve_tail;
    int* gather_scan_cursor;
    int* gather_task_count;
    int* gather_task_tokens;
    int* gather_task_nhits;
    CUDA_CHECK(cudaMalloc(&token_done_count, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_done_count, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_claimed, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_claimed, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_token_ready, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_token_ready, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_queue, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_queue, 0xff, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_head, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_head, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_ready_reserve_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_ready_reserve_tail, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_scan_cursor, COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_scan_cursor, 0, COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_tokens, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_tokens, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gather_task_nhits, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(gather_task_nhits, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    int* combine_done_count;
    int* combine_all_done;
    CUDA_CHECK(cudaMalloc(&combine_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_all_done, 0, sizeof(int)));
    // One publisher per NVL receiver warp per receiver SM.
    const int num_pub_warps_total = (num_dispatch_sms / 2) * NUM_MAX_NVL_PEERS;
    int* pending_topk_idx;
    float* pending_topk_weights;
    internode::SourceMeta* pending_meta;
    int* pending_slot;
    int* pub_ring;
    int* pub_ring_head;
    int* pub_ring_tail;
    int* recv_warp_done;
    int* publish_warp_done;
    int* publish_done_count;
    int* publish_all_done;
    CUDA_CHECK(cudaMalloc(&pending_topk_idx, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(pending_topk_idx, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pending_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(pending_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pending_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    CUDA_CHECK(cudaMalloc(&pending_slot, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(pending_slot, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pub_ring, (size_t)num_pub_warps_total * PUB_RING_DEPTH * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring, 0, (size_t)num_pub_warps_total * PUB_RING_DEPTH * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pub_ring_head, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring_head, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&pub_ring_tail, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(pub_ring_tail, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_warp_done, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_warp_done, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_warp_done, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_warp_done, 0, (size_t)num_pub_warps_total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_done_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_done_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&publish_all_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(publish_all_done, 0, sizeof(int)));

    // Combine input namespace from dispatch receive.
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&combine_input, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(combine_input, 0, (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&combine_input_topk_weights, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMemset(combine_input_topk_weights, 0, (size_t)max_total_recv_tokens * num_topk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&combine_input_src_meta, (size_t)max_total_recv_tokens * sizeof(internode::SourceMeta)));
    arena = &persistent_arena;

    // Per-token compute signaling
    int* token_compute_expected;
    CUDA_CHECK(cudaMalloc(&token_compute_expected, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_compute_expected, 0, (size_t)max_total_recv_tokens * sizeof(int)));

    // Per-slot output path scratch + reverse map (MEGAKERNEL_COMPUTE_DESIGN III).
    __nv_bfloat16* compute_output_slot;
    int* token_nhits;
    int* token_slot_list;
    int* expert_batch_enqueued;
    int* ready_batch_queue;
    int* ready_batch_reserve_tail;
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&compute_output_slot, recv_tokens_bytes));
    CUDA_CHECK(cudaMemset(compute_output_slot, 0, recv_tokens_bytes));
    arena = &persistent_arena;
    // Backward activation save: original fc1 input X plus gate/up preact by recv_token.
    // A backward replay borrows the forward state's saved buffers directly, avoiding a
    // transient allocate/free cycle for hundreds of MiB.
    __nv_bfloat16* bwd_fc1_input = external_bwd_fc1_input;
    const bool owns_bwd_fc1_input = bwd_fc1_input == nullptr;
    const size_t bwd_fc1_input_bytes = (size_t)max_total_recv_tokens * hidden_dim * sizeof(__nv_bfloat16);
    if (owns_bwd_fc1_input) {
        CUDA_CHECK(cudaMalloc(&bwd_fc1_input, bwd_fc1_input_bytes));
        CUDA_CHECK(cudaMemset(bwd_fc1_input, 0, bwd_fc1_input_bytes));
    }
    // bwd_preact is indexed by compact forward slot and needs total_expert_slots rows.
    __nv_bfloat16* bwd_preact = external_bwd_preact;
    const bool owns_bwd_preact = bwd_preact == nullptr;
    const size_t bwd_preact_bytes = (size_t)total_expert_slots * 2 * intermediate_dim * sizeof(__nv_bfloat16);
    if (owns_bwd_preact) {
        CUDA_CHECK(cudaMalloc(&bwd_preact, bwd_preact_bytes));
        CUDA_CHECK(cudaMemset(bwd_preact, 0, bwd_preact_bytes));
    }
    // Maps (recv_token, topk_slot) to the forward slot; initialized to -1.
    int* fwd_slot_map = external_fwd_slot_map;
    const bool owns_fwd_slot_map = fwd_slot_map == nullptr;
    const size_t fwd_slot_map_bytes = (size_t)max_total_recv_tokens * num_topk * sizeof(int);
    if (owns_fwd_slot_map) {
        CUDA_CHECK(cudaMalloc(&fwd_slot_map, fwd_slot_map_bytes));
        CUDA_CHECK(cudaMemset(fwd_slot_map, 0xff, fwd_slot_map_bytes));
    }
    CUDA_CHECK(cudaMalloc(&token_nhits, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_nhits, 0, (size_t)max_total_recv_tokens * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&token_slot_list, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    CUDA_CHECK(cudaMemset(token_slot_list, 0xff, (size_t)max_total_recv_tokens * num_topk * sizeof(int)));
    const int max_batches_per_expert = (max_tokens_per_expert + compute_batch_size - 1) / compute_batch_size;
    CUDA_CHECK(cudaMalloc(&expert_batch_enqueued, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_batch_enqueued, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    // Arrival-order FIFO batch queue (dispatch receiver -> scheduler). Sized at the flat
    // (expert, batch) id space; total appended entries <= Σ ceil(count_e/BATCH) <= that.
    // Initialized to -1 (0xff) = "not yet published".
    CUDA_CHECK(cudaMalloc(&ready_batch_queue, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(ready_batch_queue, 0xff, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ready_batch_reserve_tail, sizeof(int)));
    CUDA_CHECK(cudaMemset(ready_batch_reserve_tail, 0, sizeof(int)));
    int* expert_batch_ready_count;
    CUDA_CHECK(cudaMalloc(&expert_batch_ready_count, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));
    CUDA_CHECK(cudaMemset(expert_batch_ready_count, 0, (size_t)num_local_experts * max_batches_per_expert * sizeof(int)));

    // GEMM workspace: per-compute-group batched intermediates for M=128 compute batches.
    // Layout still keeps the legacy [input(M*hidden)][GU scratch(M*2I)][act(M*I)][down(M*hidden)]
    // footprint so the forward/backward workspace shape stays stable, but the forward down path
    // now writes directly to its final destinations and no longer consumes the down slice.
    // The interleaved gate/up path writes act directly from the GEMM epilogue; the GU scratch
    // region remains reserved so existing descriptors/helpers stay valid.
    // The input region is dropped here (input_buf unused: gate/up and grad_act are
    // TMA-fed from recv_tokens), matching the device gemm_stride.
    const size_t per_group_input_elems = 0;
    size_t per_group_elems = per_group_input_elems +
        (size_t)compute_batch_size * (hidden_dim + 3 * intermediate_dim);
    size_t workspace_bytes = num_compute_groups * per_group_elems * sizeof(__nv_bfloat16);
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&gemm_workspace, workspace_bytes));
    arena = &persistent_arena;

    TmaCacheKey cur_key{W_gateup, W_down, gemm_workspace,
                        num_local_experts, hidden_dim, intermediate_dim, num_compute_groups,
                        compute_batch_size};

    umma::ComputeTmaAtoms* d_compute_tma = nullptr;
    umma::InputTmaAtom_t* d_group_input_tma = nullptr;
    umma::ComputeDownTmaAtoms* d_compute_down_tma = nullptr;

    int hit_idx = -1;
    for (int i = 0; i < kTmaCacheCap; ++i) {
        if (s_tma_cache[i].valid && std::memcmp(&s_tma_cache[i].key, &cur_key, sizeof(TmaCacheKey)) == 0) {
            hit_idx = i;
            break;
        }
    }

    if (hit_idx >= 0) {
        // Reuse cached device TMA descriptors; no H2D needed.
        const TmaCacheEntry& e = s_tma_cache[hit_idx];
        d_compute_tma = e.compute_tma;
        d_group_input_tma = e.group_input_tma;
        d_compute_down_tma = e.compute_down_tma;
    } else {
        if (W_gateup != nullptr && W_down != nullptr &&
            num_local_experts <= umma::kMaxLocalExperts) {
            umma::ComputeTmaAtoms h_atoms;
            umma::build_compute_tma_atoms(h_atoms, W_gateup, num_local_experts,
                                          intermediate_dim, hidden_dim);
            CUDA_CHECK(cache_alloc(&d_compute_tma, sizeof(umma::ComputeTmaAtoms)));
            CUDA_CHECK(cudaMemcpy(d_compute_tma, &h_atoms, sizeof(umma::ComputeTmaAtoms), cudaMemcpyHostToDevice));

            std::vector<umma::InputTmaAtom_t> h_in;
            h_in.reserve(num_compute_groups);
            for (int g = 0; g < num_compute_groups; ++g) {
                const __nv_bfloat16* in_g = gemm_workspace + (size_t)g * per_group_elems;
                const __nv_bfloat16* gu_g   = in_g + per_group_input_elems;
                const __nv_bfloat16* act_g  = gu_g + (size_t)compute_batch_size * (2 * intermediate_dim);
                const __nv_bfloat16* down_g = act_g + (size_t)compute_batch_size * intermediate_dim;
                h_in.push_back(umma::make_input_group_atoms(in_g, gu_g, act_g, down_g,
                                                            compute_batch_size, hidden_dim,
                                                            intermediate_dim, hidden_dim));
            }
            CUDA_CHECK(cache_alloc(&d_group_input_tma, num_compute_groups * sizeof(umma::InputTmaAtom_t)));
            CUDA_CHECK(cudaMemcpy(d_group_input_tma, h_in.data(),
                                  num_compute_groups * sizeof(umma::InputTmaAtom_t), cudaMemcpyHostToDevice));

            umma::ComputeDownTmaAtoms h_down;
            umma::build_compute_down_tma_atoms(h_down, W_down, num_local_experts, hidden_dim, intermediate_dim);
            CUDA_CHECK(cache_alloc(&d_compute_down_tma, sizeof(umma::ComputeDownTmaAtoms)));
            CUDA_CHECK(cudaMemcpy(d_compute_down_tma, &h_down, sizeof(umma::ComputeDownTmaAtoms), cudaMemcpyHostToDevice));
        }

        // Insert into cache (round-robin). Evicting a slot frees its buffers; safe because
        // the kernel that used them completed before this slot can be reused.
        TmaCacheEntry& slot = s_tma_cache[s_tma_cache_next];
        if (slot.valid) {
            if (slot.compute_tma) teramoe_caching_free(slot.compute_tma);
            if (slot.group_input_tma) teramoe_caching_free(slot.group_input_tma);
            if (slot.compute_down_tma) teramoe_caching_free(slot.compute_down_tma);
        }
        slot.key = cur_key;
        slot.valid = true;
        slot.compute_tma = d_compute_tma;
        slot.group_input_tma = d_group_input_tma;
        slot.compute_down_tma = d_compute_down_tma;
        s_tma_cache_next = (s_tma_cache_next + 1) % kTmaCacheCap;
    } // end cache miss

    // Output accumulator [num_tokens, hidden_dim] in float32
    arena = &transient_arena;
    CUDA_CHECK(cudaMalloc(&output_accum, (size_t)num_tokens * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMemset(output_accum, 0, (size_t)num_tokens * hidden_dim * sizeof(float)));

    // Dispatch tracking heads. Each logical channel owns an independent DeepEP-shaped head space.
    const int combine_rdma_head_stride = num_tokens * kNumRDMARanks;
    // In megakernel we don't have num_rdma_recv_tokens at alloc time, use num_tokens * num_topk as upper bound.
    int num_rdma_recv_tokens_ub = num_tokens * num_topk;
    const int combine_nvl_head_stride = num_rdma_recv_tokens_ub * NUM_MAX_NVL_PEERS;
    CUDA_CHECK(cudaMalloc(&send_rdma_head, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_rdma_head, 0xFF, (size_t)num_logical_channels * combine_rdma_head_stride * sizeof(int)));  // Init to -1
    CUDA_CHECK(cudaMalloc(&send_nvl_head, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));
    CUDA_CHECK(cudaMemset(send_nvl_head, 0xFF, (size_t)num_logical_channels * combine_nvl_head_stride * sizeof(int)));  // Init to -1
    arena = &persistent_arena;

    // Recv logical-channel prefix matrices (written by forwarder)
    int num_physical_channels = num_dispatch_sms / 2;  // even/odd pairing
    int num_combine_channels = num_combine_sms / 2;
    EP_HOST_ASSERT(num_combine_channels == num_physical_channels);
    EP_HOST_ASSERT(num_logical_channels >= num_physical_channels);

    // Per-logical-channel overlap signaling
    int* channel_dispatch_done;
    int* channel_normalized;
    int* dispatch_channel_barrier;
    int* dispatch_round_barrier;
    int* combine_channel_barrier;
    CUDA_CHECK(cudaMalloc(&channel_dispatch_done, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(channel_dispatch_done, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&channel_normalized, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(channel_normalized, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_channel_barrier, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_channel_barrier, 0, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dispatch_round_barrier, ((num_logical_channels + num_physical_channels - 1) / num_physical_channels) * sizeof(int)));
    CUDA_CHECK(cudaMemset(dispatch_round_barrier, 0, ((num_logical_channels + num_physical_channels - 1) / num_physical_channels) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&combine_channel_barrier, num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(combine_channel_barrier, 0, num_logical_channels * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_prefix_matrix, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_prefix_matrix, 0, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_prefix_matrix, num_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_prefix_matrix, 0, num_ranks * num_logical_channels * sizeof(int)));

    // Per-logical-channel token counts (non-cumulative) for overlap
    int* recv_rdma_channel_token_count;
    int* recv_gbl_channel_token_count;
    CUDA_CHECK(cudaMalloc(&recv_rdma_channel_token_count, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_rdma_channel_token_count, 0, kNumRDMARanks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&recv_gbl_channel_token_count, num_ranks * num_logical_channels * sizeof(int)));
    CUDA_CHECK(cudaMemset(recv_gbl_channel_token_count, 0, num_ranks * num_logical_channels * sizeof(int)));

    // Build host-side state and copy to device
    TeraMoEState host_state;
    memset(&host_state, 0, sizeof(host_state));

    // NVSHMEM infra
    host_state.rdma_buffer_ptr = rdma_buffer_ptr;
    host_state.buffer_ptrs = buffer_ptrs;
    host_state.allocator_combine_buffer_ptrs = combine_buffer_ptrs;

    // Dispatch input
    host_state.x = x;
    host_state.x_scales = x_scales;
    host_state.topk_idx = topk_idx;
    host_state.topk_weights = topk_weights;
    host_state.is_token_in_rank = is_token_in_rank;

    // Dispatch metadata
    host_state.rdma_channel_prefix_matrix = rdma_channel_prefix_matrix;
    host_state.recv_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum;
    host_state.gbl_channel_prefix_matrix = gbl_channel_prefix_matrix;
    host_state.recv_gbl_rank_prefix_sum = recv_gbl_rank_prefix_sum;
    host_state.send_rdma_head = send_rdma_head;
    host_state.send_nvl_head = send_nvl_head;
    host_state.recv_rdma_channel_prefix_matrix = recv_rdma_channel_prefix_matrix;
    host_state.recv_gbl_channel_prefix_matrix = recv_gbl_channel_prefix_matrix;
    host_state.recv_rdma_channel_token_count = recv_rdma_channel_token_count;
    host_state.recv_gbl_channel_token_count = recv_gbl_channel_token_count;

    // Dimensions
    host_state.num_tokens = num_tokens;
    host_state.hidden_int4 = hidden_int4;
    host_state.num_scales = num_scales;
    host_state.num_topk = num_topk;
    host_state.num_experts = num_experts;
    host_state.scale_token_stride = scale_token_stride;
    host_state.scale_hidden_stride = scale_hidden_stride;
    host_state.allocator_num_dispatch_sms = num_dispatch_sms;
    host_state.allocator_num_forwarder_sms = num_forwarder_sms;
    host_state.allocator_num_compute_sms = num_compute_sms;
    host_state.allocator_num_combine_sms = num_combine_sms;
    host_state.allocator_num_logical_channels = num_logical_channels;
    host_state.allocator_max_tokens_per_expert = max_tokens_per_expert;
    host_state.allocator_max_total_recv_tokens = max_total_recv_tokens;
    host_state.allocator_num_rdma_bytes = num_rdma_bytes;
    host_state.allocator_num_nvl_bytes = num_nvl_bytes;

    // Buffer sizing
    host_state.num_max_rdma_chunked_send_tokens = dispatch_num_max_rdma_chunked_send_tokens;
    host_state.num_max_rdma_chunked_recv_tokens = dispatch_num_max_rdma_chunked_recv_tokens;
    host_state.num_max_nvl_chunked_send_tokens = dispatch_num_max_nvl_chunked_send_tokens;
    host_state.num_max_nvl_chunked_recv_tokens = dispatch_num_max_nvl_chunked_recv_tokens;

    // Topology
    host_state.rank = rank;
    host_state.num_ranks = num_ranks;

    // Receive-side signaling
    host_state.expert_recv_count = expert_recv_count;
    host_state.expert_slot_ready = expert_slot_ready;
    host_state.timeout_log_counters = timeout_log_counters;

    // Per-expert receive storage
    host_state.recv_tokens = recv_tokens;
    host_state.expert_token_offsets = expert_token_offsets;
    host_state.expert_slot_base = expert_slot_base;
    host_state.expert_count = expert_count;
    host_state.recv_token_source_info = recv_token_source_info;
    host_state.g_meta_route_w = g_meta_route_w;
    host_state.g_meta_recv_idx = g_meta_recv_idx;
    host_state.g_meta_topk_slot = g_meta_topk_slot;
    host_state.g_meta_is_single = g_meta_is_single;

    // Compute state
    host_state.compute_group_barrier = compute_group_barrier;
    host_state.compute_group_phase = compute_group_phase;
    host_state.compute_tasks = compute_tasks;
    host_state.max_compute_tasks = max_compute_tasks;
    host_state.compute_task_head = compute_task_head;
    host_state.compute_task_tail = compute_task_tail;
    host_state.compute_task_reserve_tail = compute_task_reserve_tail;
    host_state.compute_enqueue_done = compute_enqueue_done;
    host_state.scheduler_done_count = scheduler_done_count;
    host_state.priority_scheduler_done = priority_scheduler_done;
    host_state.expert_enqueue_cursor = expert_enqueue_cursor;
    host_state.compute_group_task_idx = compute_group_task_idx;

    // Dedicated gather SM task queue state
    host_state.token_done_count = token_done_count;
    host_state.gather_claimed = gather_claimed;
    host_state.combine_token_ready = combine_token_ready;
    host_state.gather_ready_queue = gather_ready_queue;
    host_state.gather_ready_head = gather_ready_head;
    host_state.gather_ready_tail = gather_ready_tail;
    host_state.gather_ready_reserve_tail = gather_ready_reserve_tail;
    host_state.gather_scan_cursor = gather_scan_cursor;
    host_state.gather_task_count = gather_task_count;
    host_state.gather_task_tokens = gather_task_tokens;
    host_state.gather_task_nhits = gather_task_nhits;
    host_state.combine_done_count = combine_done_count;
    host_state.combine_all_done = combine_all_done;

    host_state.pending_topk_idx = pending_topk_idx;
    host_state.pending_topk_weights = pending_topk_weights;
    host_state.pending_meta = pending_meta;
    host_state.pending_slot = pending_slot;
    host_state.pub_ring = pub_ring;
    host_state.pub_ring_head = pub_ring_head;
    host_state.pub_ring_tail = pub_ring_tail;
    host_state.recv_warp_done = recv_warp_done;
    host_state.publish_warp_done = publish_warp_done;
    host_state.publish_done_count = publish_done_count;
    host_state.publish_all_done = publish_all_done;
    host_state.num_pub_warps_total = num_pub_warps_total;


    // Expert weights
    host_state.W_gateup = W_gateup;
    host_state.W_down = W_down;
    host_state.compute_dtype = compute_dtype;

    // Compute output
    host_state.combine_input = combine_input;
    host_state.combine_input_topk_weights = combine_input_topk_weights;
    host_state.combine_input_src_meta = combine_input_src_meta;
    host_state.gemm_workspace = gemm_workspace;
    host_state.output_accum = output_accum;
    // S4.4 (route B2): UMMA compute TMA atoms (nullptr if disabled -> WMMA fallback).
    host_state.compute_tma = d_compute_tma;
    host_state.compute_down_tma = d_compute_down_tma;
    host_state.group_input_tma = d_group_input_tma;
    host_state.num_compute_groups = num_compute_groups;
    // Global recv_tokens A-descriptor (padded M so the last expert's tail batch TMA
    // A-load stays in bounds). Rebuilt every setup (recv_tokens ptr is arena-fresh),
    // so it is never stale — not routed through the weight-keyed TMA cache.
    host_state.recv_tokens_a_tma = umma::dg_make_a_desc(
        recv_tokens,
        static_cast<int>(total_expert_slots) + compute_batch_size,
        hidden_dim);

    // Compute dimensions
    host_state.hidden_dim = hidden_dim;
    host_state.intermediate_dim = intermediate_dim;
    host_state.num_local_experts = num_local_experts;
    host_state.max_tokens_per_expert = max_tokens_per_expert;
    host_state.total_expert_slots = static_cast<int>(total_expert_slots);
    host_state.max_total_recv_tokens = max_total_recv_tokens;
    host_state.compute_batch_size = compute_batch_size;
    host_state.combine_start_head_percent = combine_start_head_percent;

    // SM allocation
    host_state.num_dispatch_sms = num_dispatch_sms;
    host_state.num_forwarder_sms = num_forwarder_sms;
    host_state.num_compute_sms = num_compute_sms;
    host_state.num_dispatch_channels = num_physical_channels;  // even/odd SM pairing
    host_state.channel_dispatch_done = channel_dispatch_done;
    host_state.channel_normalized = channel_normalized;
    host_state.dispatch_channel_barrier = dispatch_channel_barrier;
    host_state.dispatch_round_barrier = dispatch_round_barrier;
    host_state.combine_channel_barrier = combine_channel_barrier;

    // RDMA-buffer-reuse plumbing (borrowed symmetric mailboxes + state-owned gate flag)
    host_state.rdma_reuse_dispatch_quiet_done = rdma_reuse_dispatch_quiet_done;
    host_state.rdma_reuse_combine_clear_done = rdma_reuse_combine_clear_done;
    host_state.rdma_reuse_prelude_enable = rdma_reuse_prelude_enable;
    int* rdma_reuse_prelude_done = nullptr;
    CUDA_CHECK(cudaMalloc(&rdma_reuse_prelude_done, sizeof(int)));
    CUDA_CHECK(cudaMemset(rdma_reuse_prelude_done, 0, sizeof(int)));
    host_state.rdma_reuse_prelude_done = rdma_reuse_prelude_done;

    // Combine state
    host_state.num_combine_sms = num_combine_sms;
    host_state.num_combine_channels = num_combine_channels;
    host_state.num_logical_channels = num_logical_channels;
    host_state.token_compute_expected = token_compute_expected;
    host_state.compute_output_slot = compute_output_slot;
    host_state.bwd_fc1_input = bwd_fc1_input;
    host_state.bwd_preact = bwd_preact;
    host_state.owns_bwd_fc1_input = owns_bwd_fc1_input;
    host_state.owns_bwd_preact = owns_bwd_preact;
    host_state.fwd_slot_map = fwd_slot_map;
    host_state.owns_fwd_slot_map = owns_fwd_slot_map;
    host_state.token_nhits = token_nhits;
    host_state.token_slot_list = token_slot_list;
    host_state.expert_batch_enqueued = expert_batch_enqueued;
    host_state.ready_batch_queue = ready_batch_queue;
    host_state.ready_batch_reserve_tail = ready_batch_reserve_tail;
    host_state.expert_batch_ready_count = expert_batch_ready_count;
    host_state.max_batches_per_expert = max_batches_per_expert;

    // Combine infrastructure.
    // RDMA-buffer reuse: combine reuses the single dispatch RDMA region. Both forward and backward
    // enable the in-kernel combine prelude, which guarantees all cross-machine dispatch has fully
    // drained and the combine metadata is re-cleared (with a cross-rank barrier) before combine
    // touches the region. The old separate "combine half" has been removed (allocation halved), so
    // reuse is mandatory; the prelude MUST be enabled whenever this state drives a combine.
    EP_HOST_ASSERT(rdma_reuse_prelude_enable && "combine RDMA reuse requires the prelude enabled");
    void* combine_rdma_ptr = rdma_buffer_ptr;



    host_state.combine_rdma_buffer_ptr = combine_rdma_ptr;
    host_state.combine_buffer_ptrs = combine_buffer_ptrs;
    host_state.num_rdma_bytes = num_rdma_bytes;
    host_state.num_nvl_bytes = num_nvl_bytes;
    host_state.combine_topk_weights = combine_input_topk_weights;
    host_state.is_combined_token_in_rank = is_token_in_rank;
    host_state.combined_rdma_head = send_rdma_head;  // dispatch output, combine reads back
    host_state.combined_nvl_head = send_nvl_head;
    host_state.combine_src_meta = combine_input_src_meta;  // SourceMeta in DeepEP compact combine-input namespace
    host_state.combine_rdma_channel_prefix_matrix = recv_rdma_channel_prefix_matrix;
    host_state.combine_rdma_rank_prefix_sum = recv_rdma_rank_prefix_sum;
    host_state.combine_gbl_channel_prefix_matrix = recv_gbl_channel_prefix_matrix;
    host_state.combine_gbl_channel_token_count = recv_gbl_channel_token_count;
    host_state.combine_rdma_channel_token_count = recv_rdma_channel_token_count;
    host_state.combine_num_tokens = max_total_recv_tokens;
    host_state.combine_num_combined_tokens = num_tokens;
    host_state.combine_rdma_head_stride = combine_rdma_head_stride;
    host_state.combine_nvl_head_stride = combine_nvl_head_stride;
    host_state.combine_hidden = hidden_dim;  // in dtype units (bf16), not int4
    host_state.num_max_combine_rdma_chunked_send_tokens = combine_num_max_rdma_chunked_send_tokens;
    host_state.num_max_combine_rdma_chunked_recv_tokens = combine_num_max_rdma_chunked_recv_tokens;
    host_state.num_max_combine_nvl_chunked_send_tokens = combine_num_max_nvl_chunked_send_tokens;
    host_state.num_max_combine_nvl_chunked_recv_tokens = combine_num_max_nvl_chunked_recv_tokens;
    host_state.combine_bias_0 = nullptr;
    host_state.combine_bias_1 = nullptr;

    // Combine writes directly into caller-owned Torch outputs when provided.
    int4* combined_x = external_combined_x;
    const bool owns_combined_x = combined_x == nullptr;
    if (owns_combined_x) {
        arena = &transient_arena;
        CUDA_CHECK(cudaMalloc(&combined_x, num_tokens * hidden_int4 * sizeof(int4)));
        CUDA_CHECK(cudaMemset(combined_x, 0, num_tokens * hidden_int4 * sizeof(int4)));
        arena = &persistent_arena;
    }
    host_state.combined_x = combined_x;
    host_state.owns_combined_x = owns_combined_x;

    float* combined_topk_weights = external_combined_topk_weights;
    const bool owns_combined_topk_weights = combined_topk_weights == nullptr;
    if (owns_combined_topk_weights) {
        arena = &transient_arena;
        CUDA_CHECK(cudaMalloc(&combined_topk_weights, num_tokens * num_topk * sizeof(float)));
        CUDA_CHECK(cudaMemset(combined_topk_weights, 0, num_tokens * num_topk * sizeof(float)));
        arena = &persistent_arena;
    }
    host_state.combined_topk_weights = combined_topk_weights;
    host_state.owns_combined_topk_weights = owns_combined_topk_weights;

    const int mk_num_fills = static_cast<int>(mk_fill_recs.size());
    void* fill_desc_buf = nullptr;
    int mk_num_descs = 0;  // after chunking
    std::vector<FusedFillDesc> host_descs;
    if (mk_num_fills > 0) {
        // Split large buffers into chunks so each block has roughly equal work.
        // Target: each chunk is at most CHUNK_WORDS uint32s (~256KB).
        constexpr size_t CHUNK_WORDS = 64 * 1024;  // 256KB per chunk
        host_descs.reserve(mk_num_fills * 2);
        for (int i = 0; i < mk_num_fills; ++i) {
            const uint32_t bval = static_cast<uint32_t>(mk_fill_recs[i].byte_value & 0xff);
            const uint32_t word = bval * 0x01010101u;
            size_t total_bytes = mk_fill_recs[i].bytes;
            uint8_t* base = reinterpret_cast<uint8_t*>(mk_fill_recs[i].ptr);
            size_t offset = 0;
            while (offset < total_bytes) {
                size_t chunk_bytes = min(total_bytes - offset, CHUNK_WORDS * sizeof(uint32_t));
                // Align chunk_bytes down to uint32 boundary (all allocs are 128B aligned).
                chunk_bytes = (chunk_bytes / sizeof(uint32_t)) * sizeof(uint32_t);
                if (chunk_bytes == 0) chunk_bytes = total_bytes - offset;
                host_descs.push_back(FusedFillDesc{base + offset, chunk_bytes, word});
                offset += chunk_bytes;
            }
        }
        mk_num_descs = static_cast<int>(host_descs.size());
        CUDA_CHECK(cudaMalloc(&fill_desc_buf, static_cast<size_t>(mk_num_descs) * sizeof(FusedFillDesc)));
    }
    host_state.fused_fill_desc_buf = fill_desc_buf;

    // Copy state + fill descs to device in one async batch on the current stream.
    cudaStream_t init_stream = c10::cuda::getCurrentCUDAStream().stream();
    TeraMoEState* device_state;
    CUDA_CHECK(cudaMalloc(&device_state, sizeof(TeraMoEState)));
    store_arena_chunks(persistent_arena, host_state.persistent_arena_chunks, &host_state.persistent_arena_chunk_count);
    store_arena_chunks(transient_arena, host_state.transient_arena_chunks, &host_state.transient_arena_chunk_count);
    if (host_state_out != nullptr)
        *host_state_out = host_state;
    CUDA_CHECK(cudaMemcpyAsync(device_state, &host_state, sizeof(TeraMoEState), cudaMemcpyHostToDevice, init_stream));
    if (mk_num_descs > 0) {
        CUDA_CHECK(cudaMemcpyAsync(fill_desc_buf, host_descs.data(),
                              static_cast<size_t>(mk_num_descs) * sizeof(FusedFillDesc),
                              cudaMemcpyHostToDevice, init_stream));
    }
    if (mk_num_descs > 0 || device_expert_count_mapped != nullptr) {
        const int fill_blocks = std::max(1, mk_num_descs);
        fused_fill_kernel<<<fill_blocks, 512, 0, init_stream>>>(
            static_cast<const FusedFillDesc*>(fill_desc_buf), mk_num_descs,
            device_expert_count_mapped,
            expert_slot_base,
            expert_count,
            num_local_experts);
        CUDA_CHECK(cudaGetLastError());
    }
    // Caller is responsible for ensuring host_state (stack) and host_descs (vector)
    // remain valid until the stream drains. When host_state_out != nullptr the caller
    // receives the snapshot and can control synchronization; otherwise we sync here to
    // protect the stack-local lifetime.
    CUDA_CHECK(cudaStreamSynchronize(init_stream));
#undef cudaMemset
#undef cudaMalloc

    return device_state;
}

void free_teramoe_fused_state(TeraMoEState* device_state, const TeraMoEState* cached_host_state) {
#define cudaFree(p) teramoe_caching_free(p)
    TeraMoEState host_state_copy;
    TeraMoEState* hs = &host_state_copy;
    if (cached_host_state != nullptr) {
        *hs = *cached_host_state;
    } else {
        CUDA_CHECK(cudaMemcpy(hs, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    }
#define host_state (*hs)

    if (host_state.persistent_arena_chunk_count > 0 || host_state.transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(host_state.transient_arena_chunks, host_state.transient_arena_chunk_count));
        CUDA_CHECK(free_arena_chunks(host_state.persistent_arena_chunks, host_state.persistent_arena_chunk_count));
        return;
    }

    CUDA_CHECK(cudaFree(host_state.expert_recv_count));
    CUDA_CHECK(cudaFree(host_state.expert_slot_ready));
    CUDA_CHECK(cudaFree(host_state.timeout_log_counters));
    CUDA_CHECK(cudaFree(host_state.recv_tokens));
    CUDA_CHECK(cudaFree(host_state.expert_token_offsets));
    CUDA_CHECK(cudaFree(host_state.expert_slot_base));
    CUDA_CHECK(cudaFree(host_state.expert_count));
    CUDA_CHECK(cudaFree(host_state.recv_token_source_info));
    CUDA_CHECK(cudaFree(host_state.g_meta_route_w));
    CUDA_CHECK(cudaFree(host_state.g_meta_recv_idx));
    CUDA_CHECK(cudaFree(host_state.g_meta_topk_slot));
    CUDA_CHECK(cudaFree(host_state.g_meta_is_single));
    CUDA_CHECK(cudaFree(host_state.token_compute_expected));
    CUDA_CHECK(cudaFree(host_state.compute_output_slot));
    if (host_state.owns_bwd_fc1_input) CUDA_CHECK(cudaFree(host_state.bwd_fc1_input));
    if (host_state.owns_bwd_preact) CUDA_CHECK(cudaFree(host_state.bwd_preact));
    if (host_state.owns_fwd_slot_map) CUDA_CHECK(cudaFree(host_state.fwd_slot_map));
    CUDA_CHECK(cudaFree(host_state.token_nhits));
    CUDA_CHECK(cudaFree(host_state.token_slot_list));
    CUDA_CHECK(cudaFree(host_state.expert_batch_enqueued));
    CUDA_CHECK(cudaFree(host_state.ready_batch_queue));
    CUDA_CHECK(cudaFree(host_state.ready_batch_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.expert_batch_ready_count));
    CUDA_CHECK(cudaFree(host_state.compute_group_barrier));
    CUDA_CHECK(cudaFree(host_state.compute_group_phase));
    CUDA_CHECK(cudaFree(host_state.compute_tasks));
    CUDA_CHECK(cudaFree(host_state.compute_task_head));
    CUDA_CHECK(cudaFree(host_state.compute_task_tail));
    CUDA_CHECK(cudaFree(host_state.compute_task_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.compute_enqueue_done));
    CUDA_CHECK(cudaFree(host_state.scheduler_done_count));
    CUDA_CHECK(cudaFree(host_state.priority_scheduler_done));
    CUDA_CHECK(cudaFree(host_state.expert_enqueue_cursor));
    CUDA_CHECK(cudaFree(host_state.compute_group_task_idx));
    CUDA_CHECK(cudaFree(host_state.token_done_count));
    CUDA_CHECK(cudaFree(host_state.gather_claimed));
    CUDA_CHECK(cudaFree(host_state.combine_token_ready));
    CUDA_CHECK(cudaFree(host_state.gather_ready_queue));
    CUDA_CHECK(cudaFree(host_state.gather_ready_head));
    CUDA_CHECK(cudaFree(host_state.gather_ready_tail));
    CUDA_CHECK(cudaFree(host_state.gather_ready_reserve_tail));
    CUDA_CHECK(cudaFree(host_state.gather_scan_cursor));
    CUDA_CHECK(cudaFree(host_state.gather_task_count));
    CUDA_CHECK(cudaFree(host_state.gather_task_tokens));
    CUDA_CHECK(cudaFree(host_state.gather_task_nhits));
    CUDA_CHECK(cudaFree(host_state.combine_done_count));
    CUDA_CHECK(cudaFree(host_state.combine_all_done));
    CUDA_CHECK(cudaFree(host_state.pending_topk_idx));
    CUDA_CHECK(cudaFree(host_state.pending_topk_weights));
    CUDA_CHECK(cudaFree(host_state.pending_meta));
    CUDA_CHECK(cudaFree(host_state.pending_slot));
    CUDA_CHECK(cudaFree(host_state.pub_ring));
    CUDA_CHECK(cudaFree(host_state.pub_ring_head));
    CUDA_CHECK(cudaFree(host_state.pub_ring_tail));
    CUDA_CHECK(cudaFree(host_state.recv_warp_done));
    CUDA_CHECK(cudaFree(host_state.publish_warp_done));
    CUDA_CHECK(cudaFree(host_state.publish_done_count));
    CUDA_CHECK(cudaFree(host_state.publish_all_done));
    if (host_state.owns_combined_x) CUDA_CHECK(cudaFree(host_state.combined_x));
    if (host_state.owns_combined_topk_weights) CUDA_CHECK(cudaFree(host_state.combined_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_input));
    CUDA_CHECK(cudaFree(host_state.combine_input_topk_weights));
    CUDA_CHECK(cudaFree(host_state.combine_input_src_meta));
    CUDA_CHECK(cudaFree(host_state.gemm_workspace));
    // TMA descriptors are owned by the persistent thread-local cache (see
    // allocate_teramoe_fused_state). The state only holds borrowed pointers, so it
    // must NOT free them here; the cache frees them on eviction.
    CUDA_CHECK(cudaFree(host_state.output_accum));
    CUDA_CHECK(cudaFree(host_state.send_rdma_head));
    CUDA_CHECK(cudaFree(host_state.send_nvl_head));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_prefix_matrix));
    CUDA_CHECK(cudaFree(host_state.recv_rdma_channel_token_count));
    CUDA_CHECK(cudaFree(host_state.recv_gbl_channel_token_count));
    CUDA_CHECK(cudaFree(host_state.channel_dispatch_done));
    CUDA_CHECK(cudaFree(host_state.channel_normalized));
    CUDA_CHECK(cudaFree(host_state.dispatch_channel_barrier));
    CUDA_CHECK(cudaFree(host_state.rdma_reuse_prelude_done));
    CUDA_CHECK(cudaFree(host_state.dispatch_round_barrier));
    CUDA_CHECK(cudaFree(host_state.combine_channel_barrier));
    CUDA_CHECK(cudaFree(host_state.fused_fill_desc_buf));
    CUDA_CHECK(cudaFree(device_state));
#undef host_state
#undef cudaFree
}

// Release the forward-only working buffers after the kernel has written the caller-owned
// output. The backward pass allocates a fresh v7 state and only reuses the saved-activation
// buffers (bwd_fc1_input / bwd_preact / fwd_slot_map) plus expert_count from this forward
// state (see allocate_teramoe_fused_backward_state). None of the buffers freed here are read
// by the backward, so releasing them now removes them from the forward->backward resident
// set. Each freed pointer is nulled so the eventual free_teramoe_fused_state skips it
// (teramoe_caching_free is null-safe), avoiding a double free.
void free_megakernel_forward_transient(TeraMoEState* device_state) {
    if (device_state == nullptr)
        return;
    TeraMoEState hs;
    CUDA_CHECK(cudaMemcpy(&hs, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    if (hs.transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(hs.transient_arena_chunks, hs.transient_arena_chunk_count));
        hs.transient_arena_chunk_count = 0;
        hs.recv_tokens = nullptr;
        hs.compute_output_slot = nullptr;
        hs.combine_input = nullptr;
        hs.combine_input_topk_weights = nullptr;
        hs.combine_input_src_meta = nullptr;
        hs.gemm_workspace = nullptr;
        hs.output_accum = nullptr;
        hs.send_rdma_head = nullptr;
        hs.send_nvl_head = nullptr;
        if (hs.owns_combined_x)
            hs.combined_x = nullptr;
        if (hs.owns_combined_topk_weights)
            hs.combined_topk_weights = nullptr;
        CUDA_CHECK(cudaMemcpy(device_state, &hs, sizeof(TeraMoEState), cudaMemcpyHostToDevice));
        return;
    }
    auto free_and_null = [](auto*& ptr) {
        CUDA_CHECK(teramoe_caching_free(static_cast<void*>(ptr)));
        ptr = nullptr;
    };
    free_and_null(hs.recv_tokens);
    free_and_null(hs.compute_output_slot);
    free_and_null(hs.combine_input);
    free_and_null(hs.combine_input_topk_weights);
    free_and_null(hs.combine_input_src_meta);
    free_and_null(hs.gemm_workspace);
    free_and_null(hs.output_accum);
    free_and_null(hs.send_rdma_head);
    free_and_null(hs.send_nvl_head);
    if (hs.owns_combined_x)
        free_and_null(hs.combined_x);
    if (hs.owns_combined_topk_weights)
        free_and_null(hs.combined_topk_weights);
    CUDA_CHECK(cudaMemcpy(device_state, &hs, sizeof(TeraMoEState), cudaMemcpyHostToDevice));
}

// Host-only version: uses the cached host_state snapshot directly, avoiding
// D2H + H2D cudaMemcpy entirely. The device_state is NOT updated (the backward
// re-creates its own v7 state from the host cache, and eventual free_teramoe_fused_state
// handles the remaining persistent arena chunks via the device copy). This path removes
// three memcpy operations from the forward critical path.
void free_teramoe_forward_transients_from_host(TeraMoEState* host_state) {
    if (host_state == nullptr)
        return;
    if (host_state->transient_arena_chunk_count > 0) {
        CUDA_CHECK(free_arena_chunks(host_state->transient_arena_chunks, host_state->transient_arena_chunk_count));
        host_state->transient_arena_chunk_count = 0;
        return;
    }
    // Fallback: individually free each transient pointer from host snapshot.
    auto free_ptr = [](auto*& ptr) {
        if (ptr != nullptr) {
            CUDA_CHECK(teramoe_caching_free(static_cast<void*>(ptr)));
            ptr = nullptr;
        }
    };
    free_ptr(host_state->recv_tokens);
    free_ptr(host_state->compute_output_slot);
    free_ptr(host_state->combine_input);
    free_ptr(host_state->combine_input_topk_weights);
    free_ptr(host_state->combine_input_src_meta);
    free_ptr(host_state->gemm_workspace);
    free_ptr(host_state->output_accum);
    free_ptr(host_state->send_rdma_head);
    free_ptr(host_state->send_nvl_head);
    if (host_state->owns_combined_x)
        free_ptr(host_state->combined_x);
    if (host_state->owns_combined_topk_weights)
        free_ptr(host_state->combined_topk_weights);
}

int get_teramoe_compute_batch_size_default() {
    return teramoe_config::kComputeBatchSizeDefault;
}

void get_megakernel_expert_counts(
    TeraMoEState* device_state,
    int* expert_counts,
    int num_local_experts
) {
    TeraMoEState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(expert_counts, host_state.expert_count,
                          (size_t)num_local_experts * sizeof(int), cudaMemcpyDeviceToHost));
}

void get_megakernel_backward_dimensions(
    TeraMoEState* device_state,
    int* num_tokens,
    int* hidden,
    int* intermediate,
    int* num_topk,
    int* num_local_experts
) {
    TeraMoEState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    *num_tokens = host_state.num_tokens;
    *hidden = host_state.hidden_dim;
    *intermediate = host_state.intermediate_dim;
    *num_topk = host_state.num_topk;
    *num_local_experts = host_state.num_local_experts;
}

float* get_output_accum_ptr(TeraMoEState* device_state) {
    TeraMoEState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    return host_state.output_accum;
}
void* get_combined_x_ptr(TeraMoEState* device_state) {
    TeraMoEState host_state;
    CUDA_CHECK(cudaMemcpy(&host_state, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    return host_state.combined_x;
}

// Keep the public fused-forward wrapper adjacent to its implementation so both
// dispatch paths use the same kernel specialization.
void launch_teramoe_fused_forward(
    TeraMoEState* device_state,
    const TeraMoEState* host_state,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    launch_teramoe_fused_forward_impl(
        device_state, host_state, total_sms, smem_size, stage, compute_dtype, stream);
}

// ============================================================================
// Fused TeraMOE backward path reusing forward communication state.
//
// The backward re-runs the SAME fused megakernel (dispatch + scheduler + gather + combine)
// on a patched copy of the forward TeraMoEState, swapping ONLY the compute role for a
// backward compute worker:
//   x            := grad_output   (so dispatch scatters grad_output into combine_input =: grad_down)
//   combined_x   := grad_input    (so combine reduces grad_xperm over a token's hits =: dX)
//   compute      := compute_backward_worker (reads grad_down + saved X, writes grad_xperm)
//
// Correctness: dispatch/combine are driven by routing that is deterministic per RECV_TOKEN
// (prefix-sum indices), while the expert SLOT index is assigned by a non-deterministic
// atomicAdd and therefore differs run-to-run. Hence the forward saved fc1 input by recv_token
// (fwd.bwd_fc1_input[recv_token]); the backward gathers grad_down and X by the same recv_token
// index, so everything stays aligned regardless of the re-run's slot permutation.
//
// Phase 2 math (per compute task; only grad_input / dX is produced — no dW / dprobs):
//   grad_down = combine_input row for this recv_token (filled by the re-run dispatch)
//   X_perm    = fwd.bwd_fc1_input[recv_token]
//   GU        = X_perm @ W_gateup^T           [M, 2I]   gate=GU[:,0::2], up=GU[:,1::2]
//   grad_act  = grad_down @ W_down            [M, I]
//   SwiGLU bwd (route folded in fwd act = silu(gate)*up*route):
//     g_pre=grad_act*route ; g_up=g_pre*silu(gate) ; g_gate=g_pre*up*silu'(gate)
//   grad_gu   = interleave(g_gate, g_up)      [M, 2I]
//   grad_xperm= grad_gu @ W_gateup            [M, hidden]
//   -> written to compute_output_slot/combine_input exactly like the forward down output,
//      then the reused combine reduces it into combined_x = grad_input.
//
// UMMA dgrad consumes original weights as MN-major B operands, matching DeepGEMM's
// layout handling, so no physical W_gateup_T/W_down_T materialization is needed.
// ============================================================================

struct MegaKernelBackwardState {
    TeraMoEState* fwd;                 // original forward state (buffers reused, freed by caller)
    TeraMoEState* bwd_device_state;    // patched device copy: x=grad_output, combined_x=grad_input

    const __nv_bfloat16* bwd_fc1_input;   // fwd.bwd_fc1_input [max_total_recv_tokens, hidden] (X by recv_token)
    const __nv_bfloat16* bwd_preact;      // fwd.bwd_preact [max_total_recv_tokens, num_topk, 2I]
    umma::ComputeBackwardTmaAtoms* compute_bwd_tma;
    CUtensorMap* wgrad_dgu_a_tma;         // [num_local_experts * max_batches_per_expert] batch-start A descs for wgrad_dgu_slot

    const __nv_bfloat16* grad_output;     // [num_combined_tokens, hidden] dY (== bwd_device_state->x)
    __nv_bfloat16* grad_input;            // [num_combined_tokens, hidden] dX (== bwd_device_state->combined_x)
    __nv_bfloat16* grad_w_gateup;         // [num_local_experts, 2 * intermediate, hidden]
    __nv_bfloat16* grad_w_down;           // [num_local_experts, hidden, intermediate]
    float* grad_topk_weights;             // [num_tokens, num_topk]
    __nv_bfloat16* wgrad_x_slot;           // [expert_slots, hidden]
    __nv_bfloat16* wgrad_act_slot;         // [expert_slots, intermediate], route-weighted
    __nv_bfloat16* wgrad_dz_slot;          // [expert_slots, hidden]
    __nv_bfloat16* wgrad_dgu_slot;         // [expert_slots, 2 * intermediate]
};

struct MegaKernelBackwardHostContext {
    MegaKernelBackwardState backward_state;
    TeraMoEState bwd_state;
    umma::ComputeBackwardTmaAtoms compute_bwd_tma_atoms;
    std::vector<CUtensorMap> wgrad_dgu_a_tma;
};

__device__ __forceinline__ float fused_moe_bwd_bf16_from_u32(uint32_t v, int lane) {
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v)[lane]);
}

__device__ __forceinline__ uint32_t fused_moe_bwd_pack_bf16x2(float lo, float hi) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;\n"
                 : "=r"(out) : "f"(hi), "f"(lo));
    return out;
}

__device__ __forceinline__ void fused_moe_bwd_swiglu_pair2_f32x2(
    uint32_t gu0, uint32_t gu1, uint32_t grad01, float route,
    uint32_t& out0, uint32_t& out1, uint32_t& act01, float& route_grad
) {
    float2 gate = {fused_moe_bwd_bf16_from_u32(gu0, 0), fused_moe_bwd_bf16_from_u32(gu1, 0)};
    float2 up = {fused_moe_bwd_bf16_from_u32(gu0, 1), fused_moe_bwd_bf16_from_u32(gu1, 1)};
    float2 grad_raw = {fused_moe_bwd_bf16_from_u32(grad01, 0), fused_moe_bwd_bf16_from_u32(grad01, 1)};
    float2 grad = {grad_raw.x * route, grad_raw.y * route};
    float2 sig = {
        1.0f / (1.0f + __expf(-gate.x)),
        1.0f / (1.0f + __expf(-gate.y))};
    float2 silu, activation, silu_grad, sig_minus_silu_sig, d_silu_grad, dgate;
    cute::mul(silu, gate, sig);
    cute::mul(activation, silu, up);
    cute::mul(silu_grad, silu, grad);
    cute::fma(sig_minus_silu_sig, silu, {-sig.x, -sig.y}, sig);
    cute::fma(d_silu_grad, sig_minus_silu_sig, grad, silu_grad);
    cute::mul(dgate, d_silu_grad, up);
    out0 = fused_moe_bwd_pack_bf16x2(dgate.x, silu_grad.x);
    out1 = fused_moe_bwd_pack_bf16x2(dgate.y, silu_grad.y);
    act01 = fused_moe_bwd_pack_bf16x2(route * activation.x, route * activation.y);
    route_grad += grad_raw.x * activation.x + grad_raw.y * activation.y;
}

// Expert-compute backward for one compute SM. Mirrors the forward compute_worker's
// task-queue / gather / output-scatter / signaling protocol EXACTLY (so the reused

// Re-enable COMPUTE_BATCH_SIZE macro for backward device code (was #undef'd for host functions above).
#define COMPUTE_BATCH_SIZE (state->compute_batch_size)
// combine handshake works), swapping only the GEMM math for the backward pass.
// state == bs->bwd_device_state: combine_input holds grad_down (filled by the re-run
// dispatch of grad_output), compute_output_slot/combine_input receive grad_xperm.
template <ComputeDType kComputeDType, bool kStopAtDispatchDone>
__device__ __forceinline__ void compute_backward_worker_core(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    int group_id_base,
    uint8_t* smem_buffer
) {
    TeraMoEState* state = bs->bwd_device_state;
    const int thread_id = threadIdx.x;
    const int local_warp_id = thread_id / 32;
    if (num_compute_sms <= 0 || compute_sm_idx < 0 || compute_sm_idx >= num_compute_sms)
        return;
    const int local_group_id = compute_sm_idx / COMPUTE_GROUP_SIZE;
    const int group_first_sm_idx = local_group_id * COMPUTE_GROUP_SIZE;
    const int group_size = min(COMPUTE_GROUP_SIZE, num_compute_sms - group_first_sm_idx);
    const int group_id = group_id_base + local_group_id;
    const int num_compute_groups = state->num_compute_groups;
    if (group_size <= 0 || group_id >= num_compute_groups)
        return;
    const int group_sm_idx = compute_sm_idx - group_first_sm_idx;
    const int num_warps_per_sm = blockDim.x / 32;
    const int group_warp_id = group_sm_idx * num_warps_per_sm + local_warp_id;
    const int group_num_warps = group_size * num_warps_per_sm;
    const int group_thread_id = group_sm_idx * blockDim.x + thread_id;
    const int group_num_threads = group_size * blockDim.x;
    const int num_local_experts = state->num_local_experts;
    const int max_tpe = state->max_tokens_per_expert;
    const int hidden = state->hidden_dim;
    const int intermediate = state->intermediate_dim;
    const int twoI = 2 * intermediate;
    const int num_topk = state->num_topk;
    const int hidden_int4 = hidden * sizeof(__nv_bfloat16) / sizeof(int4);

    const int input_stride = 0;
    const int gu_stride    = COMPUTE_BATCH_SIZE * twoI;
    const int act_stride   = COMPUTE_BATCH_SIZE * intermediate;
    const int down_stride  = COMPUTE_BATCH_SIZE * hidden;
    const int gemm_stride  = input_stride + gu_stride + act_stride + down_stride;
    __nv_bfloat16* input_buf = state->gemm_workspace + (size_t)group_id * gemm_stride;
    __nv_bfloat16* gu_buf    = input_buf + input_stride;
    __nv_bfloat16* up_buf    = gu_buf + gu_stride;
    __nv_bfloat16* down_buf  = up_buf + act_stride;

    float* smem_wmma_buf = reinterpret_cast<float*>(smem_buffer);

    using ComputeUmmaSmemLayout = umma::DgSmemLayout<umma::kDgRunMulticast>;
    constexpr size_t kComputeUmmaBarrierBytes =
        (ComputeUmmaSmemLayout::kNumStages * 3 + ComputeUmmaSmemLayout::kNumEpilogueStages * 2 + 1) *
        sizeof(cutlass::arch::ClusterTransactionBarrier) + sizeof(uint32_t);
    constexpr size_t kComputeUmmaScratchBytes =
        ComputeUmmaSmemLayout::SMEM_CD_SIZE +
        ComputeUmmaSmemLayout::kNumStages *
            (ComputeUmmaSmemLayout::SMEM_A_SIZE_PER_STAGE + ComputeUmmaSmemLayout::SMEM_B_SIZE_PER_STAGE) +
        kComputeUmmaBarrierBytes;
    constexpr size_t kComputeScratchBytes = kComputeUmmaScratchBytes;
    static_assert(kComputeScratchBytes <=
                  (size_t)kNumCombineTMABytesPerForwarderWarp * kNumCombineForwarderWarps,
                  "backward compute GEMM scratch must fit the launch dynamic-smem budget");

    // Compute-task metadata in GMEM (moved out of shared memory, per-SM slice), same as
    // forward. s_* names retained; each points at [sm_id*COMPUTE_BATCH_SIZE + row].
    const int64_t kMetaBase = (int64_t)sm_id * COMPUTE_BATCH_SIZE;
    int* s_recv_token_idx = state->g_meta_recv_idx + kMetaBase;
    int* s_topk_slot = state->g_meta_topk_slot + kMetaBase;
    unsigned char* s_is_single = state->g_meta_is_single + kMetaBase;
    float* s_route_w = state->g_meta_route_w + kMetaBase;

    // TMEM persistent across tasks (same optimization as forward compute_worker_core).
    bool umma_tmem_allocated = false;

    while (true) {
        // ---- pop a compute task (group leader) and broadcast to the group ----
        if (group_sm_idx == 0 && thread_id == 0) {
            int task_idx = -1;
            while (true) {
                if constexpr (kStopAtDispatchDone) {
                    // Check if compute progress has reached the threshold to switch to combine.
                    int enqueue_done = ld_acquire_global(state->compute_enqueue_done);
                    if (enqueue_done) {
                        int tail = ld_acquire_global(state->compute_task_tail);
                        int head = ld_acquire_global(state->compute_task_head);
                        if (tail == 0 || head * 100 >= tail * state->combine_start_head_percent) {
                            task_idx = -3;
                            break;
                        }
                    }
                }
                int head = ld_acquire_global(state->compute_task_head);
                int tail = ld_acquire_global(state->compute_task_tail);
                if (head >= tail) {
                    if constexpr (!kStopAtDispatchDone) {
                        if (ld_acquire_global(state->compute_enqueue_done))
                            task_idx = -2;
                    }
                    break;
                }
                if (atomicCAS(state->compute_task_head, head, head + 1) == head) {
                    task_idx = head;
                    break;
                }
            }
            st_release_gpu_global(&state->compute_group_task_idx[group_id], task_idx);
        }
        compute_group_sync(state, group_id, group_size);

        int task_idx = ld_acquire_global(&state->compute_group_task_idx[group_id]);
        if (task_idx == -2 || task_idx == -3) {
            // Dealloc TMEM before exiting the backward persistent loop.
            if (umma_tmem_allocated) {
                char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
                umma::umma_dealloc(cluster_smem);
            }
            break;
        }
        if (task_idx < 0) {
            if (group_sm_idx == 0 && thread_id == 0)
                __nanosleep(128);
            compute_group_sync(state, group_id, group_size);
            continue;
        }

        ComputeTask task = state->compute_tasks[task_idx];
        int expert_id = task.expert_id;
        int start_slot = task.start_slot;
        int batch_size = task.num_tokens;


        constexpr bool kUseUmmaCompute = (MK_COMPUTE_KERNEL != 0);
        static_assert(kUseUmmaCompute, "WMMA compute path removed; MK_COMPUTE_KERNEL must be 1 (1-CTA) or 2 (2-CTA UMMA)");
        constexpr bool kUseUmmaBwdGemm = kUseUmmaCompute;
        const bool use_umma_bwd_for_group =
            kUseUmmaBwdGemm && group_size == COMPUTE_GROUP_SIZE &&
            state->group_input_tma != nullptr && bs->compute_bwd_tma != nullptr &&
            bs->wgrad_dgu_a_tma != nullptr && batch_size <= COMPUTE_BATCH_SIZE;
        // Backward compute requires the full UMMA path (full group + built bwd TMA atoms).
        EP_DEVICE_ASSERT(use_umma_bwd_for_group);
        constexpr int kUmmaClusterDim = (MK_COMPUTE_KERNEL == 2 ? 2 : 1);
        constexpr int kUmmaClustersPerGroup = COMPUTE_GROUP_SIZE / kUmmaClusterDim;

        if constexpr (kUseUmmaBwdGemm) {
            if (use_umma_bwd_for_group && local_warp_id == 0) {
                const umma::InputTmaAtom_t& prefetch_atom = state->group_input_tma[group_id];
                cute::prefetch_tma_descriptor(&prefetch_atom.a);
                cute::prefetch_tma_descriptor(&prefetch_atom.act_cd);
                cute::prefetch_tma_descriptor(&prefetch_atom.gu_a);
                cute::prefetch_tma_descriptor(&prefetch_atom.down_cd);
                cute::prefetch_tma_descriptor(&bs->compute_bwd_tma->wdown[expert_id]);
                cute::prefetch_tma_descriptor(&bs->compute_bwd_tma->wgateup[expert_id]);
            }
        }

        // ---- gather per-row (recv_token, single-hit flag, route weight) ----
        for (int i = thread_id; i < batch_size; i += blockDim.x) {
            int base_offset = state->expert_slot_base[expert_id] + start_slot + i;
            int recv_token = ld_acquire_global(&state->recv_token_source_info[base_offset * 2]);
            int topk_slot = ld_acquire_global(&state->recv_token_source_info[base_offset * 2 + 1]);
            int expected = ld_acquire_global(&state->token_compute_expected[recv_token]);
            s_recv_token_idx[i] = recv_token;
            s_topk_slot[i] = topk_slot;
            s_is_single[i] = static_cast<unsigned char>(expected == 1);
            s_route_w[i] = ld_nc_global(&state->combine_input_topk_weights[recv_token * num_topk + topk_slot]);
        }
        for (int i = batch_size + thread_id; i < COMPUTE_BATCH_SIZE; i += blockDim.x) {
            s_recv_token_idx[i] = -1;
            s_topk_slot[i] = -1;
            s_is_single[i] = 0;
            s_route_w[i] = 0.0f;
        }
        __syncthreads();

        // ---- stage weight-grad inputs (wgrad_dz_slot = grad_down, wgrad_x_slot = X) ----
        const int64_t slot_base64 = (int64_t)state->expert_slot_base[expert_id] + start_slot;
        const int4* bwd_x_i4 = reinterpret_cast<const int4*>(bs->bwd_fc1_input);               // = X
        // grad_act is TMA-fed from recv_tokens (backward dispatch staged grad_down there),
        // so input_buf is not filled here. This loop only stages the per-slot weight-grad
        // inputs for the wgrad GEMMs. Padding rows are unused (dSwiGLU/wgrad bounded by batch_size).
        const int4* rds_recv_tokens_i4 = reinterpret_cast<const int4*>(state->recv_tokens);
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int rt = s_recv_token_idx[row];
            const int64_t slot = slot_base64 + row;
            const int4 grad_vec = rds_recv_tokens_i4[slot * hidden_int4 + v];
            const int4 x_vec = bwd_x_i4[(int64_t)rt * hidden_int4 + v];
            reinterpret_cast<int4*>(bs->wgrad_dz_slot)[slot * hidden_int4 + v] = grad_vec;
            reinterpret_cast<int4*>(bs->wgrad_x_slot)[slot * hidden_int4 + v] = x_vec;
        }
        compute_group_sync(state, group_id, group_size);

        const __nv_bfloat16* Wgu_e = &state->W_gateup[(size_t)expert_id * twoI * hidden];       // [2I,hidden]
        const __nv_bfloat16* Wd_e  = &state->W_down[(size_t)expert_id * hidden * intermediate]; // [hidden,I]

        // Saved-PreAct is mandatory now (WMMA gate/up recompute fallback removed):
        // dSwiGLU below reads gate/up directly from bs->bwd_preact.
        EP_DEVICE_ASSERT(bs->bwd_preact != nullptr);
        __nv_bfloat16* dgu_dst = bs->wgrad_dgu_slot + (size_t)slot_base64 * twoI;
        compute_group_sync(state, group_id, group_size);

        // grad_act = grad_down @ W_down -> up_buf [M,I]. UMMA saved-preact baseline.
        if (use_umma_bwd_for_group) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];
            uint32_t grad_act_accum_iter = 0;
            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            // grad_act A operand = grad_down. Direct-staging TMA-feeds straight from recv_tokens
            // (backward dispatch staged grad_down there) with the task's row base.
            const CUtensorMap* gradact_a_desc = &state->recv_tokens_a_tma;
            uint32_t gradact_a_m_base = (uint32_t)slot_base64;
            umma::umma_dgrad_mn_persistent(
                gradact_a_desc,
                &bs->compute_bwd_tma->wdown[expert_id],
                &in_atom.act_cd,
                COMPUTE_BATCH_SIZE, intermediate, hidden,
                cluster_in_group, num_clusters,
                cluster_smem, grad_act_accum_iter,
                gradact_a_m_base);
            compute_group_sync(state, group_id, group_size);
        }

        // SwiGLU backward and route-probability gradient. One warp owns each route row,
        // so dTopKWeight needs only a warp reduction and one scalar store (no atomics).
        const int lane_id = get_lane_id();
        const bool dswiglu_packed4 = ((intermediate & 3) == 0);
        const int intermediate_i4 = intermediate >> 2;
        for (int m = group_warp_id; m < batch_size; m += group_num_warps) {
            const float route = s_route_w[m];
            const int slot = state->expert_slot_base[expert_id] + start_slot + m;
            const int recv_token_m = s_recv_token_idx[m];
            const int topk_slot_m = s_topk_slot[m];
            // Read saved gate/up values by translating the stable (recv_token, topk_slot)
            // key to the compact forward slot.
            int fwd_preact_slot = state->fwd_slot_map[(int64_t)recv_token_m * num_topk + topk_slot_m];
            if (fwd_preact_slot < 0) fwd_preact_slot = 0;  // safety net; should not occur for a real hit
            const __nv_bfloat16* gu_src = bs->bwd_preact + (int64_t)fwd_preact_slot * twoI;
            float route_grad = 0.0f;
            if (dswiglu_packed4) {
                const int4* gu4 = reinterpret_cast<const int4*>(gu_src);
                const int2* ga4 = reinterpret_cast<const int2*>(up_buf + (size_t)m * intermediate);
                int4* dgu4 = reinterpret_cast<int4*>(dgu_dst + (size_t)m * twoI);
                int2* wgrad_act4 = reinterpret_cast<int2*>(bs->wgrad_act_slot + (int64_t)slot * intermediate);
                for (int q = lane_id; q < intermediate_i4; q += 32) {
                    const int4 gu = gu4[q];
                    const int2 ga = ga4[q];
                    uint32_t o0, o1, o2, o3, a01, a23;
                    fused_moe_bwd_swiglu_pair2_f32x2(
                        static_cast<uint32_t>(gu.x), static_cast<uint32_t>(gu.y),
                        static_cast<uint32_t>(ga.x), route, o0, o1, a01, route_grad);
                    fused_moe_bwd_swiglu_pair2_f32x2(
                        static_cast<uint32_t>(gu.z), static_cast<uint32_t>(gu.w),
                        static_cast<uint32_t>(ga.y), route, o2, o3, a23, route_grad);
                    const int4 out = make_int4(static_cast<int>(o0), static_cast<int>(o1),
                                               static_cast<int>(o2), static_cast<int>(o3));
                    dgu4[q] = out;
                    wgrad_act4[q] = make_int2(static_cast<int>(a01), static_cast<int>(a23));
                }
            } else {
                for (int i = lane_id; i < intermediate; i += 32) {
                    float gate = __bfloat162float(gu_src[2 * i]);
                    float up = __bfloat162float(gu_src[2 * i + 1]);
                    float ga = __bfloat162float(up_buf[m * intermediate + i]);
                    float sig = 1.0f / (1.0f + __expf(-gate));
                    float silu = gate * sig;
                    float activation = silu * up;
                    float g_pre = ga * route;
                    float g_up = g_pre * silu;
                    float dsilu = sig * (1.0f + gate * (1.0f - sig));
                    float g_gate = g_pre * up * dsilu;
                    route_grad += ga * activation;
                    bs->wgrad_act_slot[(int64_t)slot * intermediate + i] =
                        __float2bfloat16(route * activation);
                    dgu_dst[(size_t)m * twoI + 2 * i] = __float2bfloat16(g_gate);
                    dgu_dst[(size_t)m * twoI + 2 * i + 1] = __float2bfloat16(g_up);
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                route_grad += __shfl_down_sync(0xffffffff, route_grad, offset);
            if (lane_id == 0) {
                state->combine_input_topk_weights[recv_token_m * num_topk + topk_slot_m] = route_grad;
            }
        }
        compute_group_sync(state, group_id, group_size);

        // grad_xperm = grad_gu @ W_gateup -> down_buf [M,hidden].
        if (use_umma_bwd_for_group) {
            const int cluster_in_group = group_sm_idx / kUmmaClusterDim;
            const int num_clusters = kUmmaClustersPerGroup;
            char* cluster_smem = reinterpret_cast<char*>(smem_wmma_buf);
            const umma::InputTmaAtom_t& in_atom = state->group_input_tma[group_id];
            const int dgu_batch_id = start_slot / COMPUTE_BATCH_SIZE;
            const CUtensorMap* dgu_a_tma =
                &bs->wgrad_dgu_a_tma[(int64_t)expert_id * state->max_batches_per_expert + dgu_batch_id];
            uint32_t grad_x_accum_iter = 0;
            if (!umma_tmem_allocated) {
                umma::dg_init_barriers_tmem<umma::kDgRunMulticast>(cluster_smem);
                umma_tmem_allocated = true;
            } else {
                umma::dg_reinit_barriers<umma::kDgRunMulticast>(cluster_smem);
            }
            umma::umma_dgrad_mn_persistent(
                dgu_a_tma,
                &bs->compute_bwd_tma->wgateup[expert_id],
                &in_atom.down_cd,
                batch_size, hidden, twoI,
                cluster_in_group, num_clusters,
                cluster_smem, grad_x_accum_iter);
            compute_group_sync(state, group_id, group_size);
        }

        // ---- per-slot output scatter (identical to forward down output) ----
        const int4* down_i4 = reinterpret_cast<const int4*>(down_buf);
        int4* slot_out_i4 = reinterpret_cast<int4*>(state->compute_output_slot);
        int4* token_out_i4 = reinterpret_cast<int4*>(state->combine_input);
        const int slot_base = state->expert_slot_base[expert_id] + start_slot;
        for (int idx = group_thread_id; idx < batch_size * hidden_int4; idx += group_num_threads) {
            int row = idx / hidden_int4;
            int v = idx - row * hidden_int4;
            int slot = slot_base + row;
            int4 data = down_i4[idx];
            if (s_is_single[row])
                token_out_i4[(int64_t)s_recv_token_idx[row] * hidden_int4 + v] = down_i4[idx];
            else
                slot_out_i4[(int64_t)slot * hidden_int4 + v] = down_i4[idx];
        }
        __threadfence();

        // ---- signal per-token completion (identical to forward) ----
        for (int row = group_thread_id; row < batch_size; row += group_num_threads) {
            const int recv_token = s_recv_token_idx[row];
            if (s_is_single[row]) {
                // Single-hit: sole contributor; output already fenced device-wide by the
                // __threadfence + compute_group_sync above. Publish directly — no per-row
                // fence, no token_done_count bump (only multi-hit uses it in the gather worker).
                atomicExch(&state->combine_token_ready[recv_token], 1);
            } else {
                atomicAdd(&state->token_done_count[recv_token], 1);
            }
        }
        compute_group_sync(state, group_id, group_size);

    }
    if constexpr (kStopAtDispatchDone) {
        asm volatile("barrier.sync 15, %0;" :: "r"(static_cast<int>(blockDim.x)) : "memory");
    }
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void compute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int compute_sm_idx,
    int num_compute_sms,
    uint8_t* smem_buffer
) {
    compute_backward_worker_core<kComputeDType, false>(
        bs, sm_id, compute_sm_idx, num_compute_sms, 0, smem_buffer);
}

template <ComputeDType kComputeDType>
__device__ __forceinline__ void combine_precompute_backward_worker(
    MegaKernelBackwardState* bs,
    int sm_id,
    int combine_sm_idx,
    int num_combine_sms,
    uint8_t* smem_buffer
) {
    TeraMoEState* state = bs->bwd_device_state;
    const int post_group_count =
        (state->num_compute_sms + state->num_dispatch_sms + COMPUTE_GROUP_SIZE - 1) / COMPUTE_GROUP_SIZE;
    compute_backward_worker_core<kComputeDType, true>(
        bs, sm_id, combine_sm_idx, num_combine_sms, post_group_count, smem_buffer);
}

// Backward megakernel: same role layout as teramoe_fused_forward_kernel (dispatch / combine /
// scheduler / gather all reused verbatim on the patched bwd state); only the compute
// role is swapped for compute_backward_worker.
template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
__global__ void __launch_bounds__(MegaKernelRdmaConfig<kNumRDMARanks>::kMegaKernelNumThreads, 1) teramoe_fused_backward_kernel(
    MegaKernelBackwardState* bs
) {
    TeraMoEState* state = bs->bwd_device_state;
    const int sm_id = blockIdx.x;
    const int num_dispatch_sms = state->num_dispatch_sms;
    const int num_combine_sms = state->num_combine_sms;
    const int num_compute_sms = state->num_compute_sms;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    SmRole role;
    int role_idx;
    const int compute_begin = num_dispatch_sms + num_combine_sms + COMPUTE_SCHEDULER_SMS;
    const int gather_begin = compute_begin + num_compute_sms;
    const int total_compute_sms_after_dispatch = num_compute_sms + num_dispatch_sms;
    if (sm_id < num_dispatch_sms) {
        role = SmRole::kDispatch;
        role_idx = sm_id;
    } else if (sm_id < num_dispatch_sms + num_combine_sms) {
        role = SmRole::kCombine;
        role_idx = sm_id - num_dispatch_sms;
    } else if (sm_id < compute_begin) {
        role = SmRole::kScheduler;
        role_idx = sm_id - num_dispatch_sms - num_combine_sms;
    } else if (sm_id < gather_begin) {
        role = SmRole::kCompute;
        role_idx = sm_id - compute_begin;
    } else {
        role = SmRole::kGather;
        role_idx = sm_id - gather_begin;
    }

    switch (role) {
        case SmRole::kDispatch:
            dispatch_worker<kNumRDMARanks, kStage, kComputeDType, true>(sm_id, role_idx, state, bs);
            break;
        case SmRole::kCombine:
            combine_precompute_backward_worker<kComputeDType>(bs, sm_id, role_idx, num_combine_sms, smem_buffer);
            combine_worker<kNumRDMARanks, kStage>(role_idx, state);
            break;
        case SmRole::kScheduler:
            compute_scheduler_worker(state, role_idx, COMPUTE_SCHEDULER_SMS);
            break;
        case SmRole::kCompute:
            MK_BACKWARD_COMPUTE_WORKER(
                kComputeDType, bs, sm_id, role_idx,
                total_compute_sms_after_dispatch, smem_buffer);
            break;
        case SmRole::kGather:
            gather_worker(state, role_idx);
            break;
        default:
            break;
    }
}

// Build the backward state with an independent v7 allocation. Only the saved forward
// activation is shared; routing inputs and Buffer-owned transport pointers are reused as
// allocator inputs, while all derived routing, workspace, FIFO, and counter storage is new.
// Host-side backward allocation uses fs.compute_batch_size from the forward state directly.
#undef COMPUTE_BATCH_SIZE
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
    const TeraMoEState* cached_fwd_host_state
) {
    TeraMoEState fs;
    if (cached_fwd_host_state != nullptr) {
        // Fast path: use the host-cached forward state snapshot directly,
        // avoiding a synchronous D2H cudaMemcpy that stalls the pipeline.
        fs = *cached_fwd_host_state;
    } else {
        // Fallback: synchronous D2H (legacy path, should not be hit in training).
        CUDA_CHECK(cudaMemcpy(&fs, fwd_device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    }

    const int hidden = fs.hidden_dim;
    const int intermediate = fs.intermediate_dim;
    const int twoI = 2 * intermediate;
    const int num_local_experts = fs.num_local_experts;
    EP_HOST_ASSERT(fs.num_compute_groups > 0);
    EP_HOST_ASSERT(total_sms >= fs.num_compute_groups * COMPUTE_GROUP_SIZE);

    EP_HOST_ASSERT(host_expert_count != nullptr);
    EP_HOST_ASSERT(host_context != nullptr);
    std::vector<int> h_bwd_expert_count(host_expert_count, host_expert_count + num_local_experts);
    std::vector<int> h_bwd_expert_slot_base(num_local_experts);
    size_t total_bwd_slots = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        h_bwd_expert_slot_base[e] = static_cast<int>(total_bwd_slots);
        total_bwd_slots += static_cast<size_t>(h_bwd_expert_count[e]);
    }
    if (total_bwd_slots == 0) total_bwd_slots = 1;

    auto* host_ctx = new MegaKernelBackwardHostContext{};
    umma::ComputeBackwardTmaAtoms* d_compute_bwd_tma = nullptr;
    if (num_local_experts <= umma::kMaxLocalExperts) {
        umma::build_compute_backward_tma_atoms(
            host_ctx->compute_bwd_tma_atoms, fs.W_down, fs.W_gateup,
            num_local_experts, hidden, intermediate);
        CUDA_CHECK(teramoe_caching_alloc(
            reinterpret_cast<void**>(&d_compute_bwd_tma), sizeof(umma::ComputeBackwardTmaAtoms)));
        CUDA_CHECK(cudaMemcpyAsync(d_compute_bwd_tma, &host_ctx->compute_bwd_tma_atoms,
                                   sizeof(umma::ComputeBackwardTmaAtoms),
                                   cudaMemcpyHostToDevice, stream));
    }

    TeraMoEState* bwd_device_state = ::teramoe::allocate_teramoe_fused_state(
        reinterpret_cast<const int4*>(grad_output), nullptr,
        fs.topk_idx, fs.topk_weights, fs.is_token_in_rank,
        fs.rdma_channel_prefix_matrix, fs.recv_rdma_rank_prefix_sum,
        fs.gbl_channel_prefix_matrix, fs.recv_gbl_rank_prefix_sum,
        fs.rdma_buffer_ptr, fs.buffer_ptrs, fs.allocator_combine_buffer_ptrs,
        fs.num_tokens, fs.hidden_dim, fs.hidden_int4, fs.intermediate_dim,
        0, fs.num_topk, fs.num_experts, fs.num_local_experts,
        fs.num_ranks, fs.rank, fs.scale_token_stride, fs.scale_hidden_stride,
        fs.num_max_rdma_chunked_send_tokens, fs.num_max_rdma_chunked_recv_tokens,
        fs.num_max_nvl_chunked_send_tokens, fs.num_max_nvl_chunked_recv_tokens,
        fs.num_max_combine_rdma_chunked_send_tokens,
        fs.num_max_combine_rdma_chunked_recv_tokens,
        fs.num_max_combine_nvl_chunked_send_tokens,
        fs.num_max_combine_nvl_chunked_recv_tokens,
        fs.W_gateup, fs.W_down, fs.compute_dtype,
        nullptr, nullptr, nullptr, nullptr,
        fs.allocator_num_dispatch_sms, fs.allocator_num_forwarder_sms,
        fs.allocator_num_compute_sms, fs.allocator_num_combine_sms,
        fs.allocator_num_logical_channels,
        fs.allocator_max_tokens_per_expert, fs.allocator_max_total_recv_tokens,
        fs.allocator_num_rdma_bytes, fs.allocator_num_nvl_bytes,
        h_bwd_expert_count.data(),
        nullptr,
        const_cast<__nv_bfloat16*>(fs.bwd_fc1_input),
        const_cast<__nv_bfloat16*>(fs.bwd_preact),
        fs.fwd_slot_map,
        reinterpret_cast<int4*>(grad_input),
        reinterpret_cast<float*>(grad_topk_weights),
        &host_ctx->bwd_state,
        fs.rdma_reuse_dispatch_quiet_done,
        fs.rdma_reuse_combine_clear_done,
        1 /* rdma_reuse_prelude_enable: backward reuses the dispatch RDMA region too.
             Backward shares dispatch_worker / combine_worker, so the same
             dispatch_channel_barrier==2 gate + quiet + phase-B protocol applies. */,
        fs.compute_batch_size,
        fs.combine_start_head_percent);

    MegaKernelBackwardState& hs = host_ctx->backward_state;
    hs.fwd = fwd_device_state;
    hs.bwd_device_state = bwd_device_state;
    hs.bwd_fc1_input = fs.bwd_fc1_input;
    hs.bwd_preact = fs.bwd_preact;
    hs.compute_bwd_tma = d_compute_bwd_tma;
    hs.grad_output = reinterpret_cast<const __nv_bfloat16*>(grad_output);
    hs.grad_input = reinterpret_cast<__nv_bfloat16*>(grad_input);
    hs.grad_w_gateup = reinterpret_cast<__nv_bfloat16*>(grad_w_gateup);
    hs.grad_w_down = reinterpret_cast<__nv_bfloat16*>(grad_w_down);
    hs.grad_topk_weights = reinterpret_cast<float*>(grad_topk_weights);
    // Compact Family B (wgrad) scratch: Σ count, same per-expert slot layout as the backward
    // state. wgrad_dgu_slot gets one extra compute_batch_size of padding so the last batch's
    // CBS-row TMA descriptor tile stays within the allocation.
    const int cbs = fs.compute_batch_size;
    const size_t num_dgu_batch_tmas =
        (size_t)fs.num_local_experts * fs.max_batches_per_expert;
    EP_HOST_ASSERT(wgrad_x_slot != nullptr && wgrad_act_slot != nullptr);
    EP_HOST_ASSERT(wgrad_dz_slot != nullptr && wgrad_dgu_slot != nullptr);
    hs.wgrad_x_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_x_slot);
    hs.wgrad_act_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_act_slot);
    hs.wgrad_dz_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_dz_slot);
    hs.wgrad_dgu_slot = reinterpret_cast<__nv_bfloat16*>(wgrad_dgu_slot);
    host_ctx->wgrad_dgu_a_tma.resize(num_dgu_batch_tmas);
    for (int expert = 0; expert < fs.num_local_experts; ++expert) {
        const int ebase = h_bwd_expert_slot_base[expert];
        const int ecnt = h_bwd_expert_count[expert];
        const int active_batches = (ecnt + cbs - 1) / cbs;
        EP_HOST_ASSERT(active_batches <= fs.max_batches_per_expert);
        for (int batch = 0; batch < active_batches; ++batch) {
            const int row = ebase + batch * cbs;
            const __nv_bfloat16* dgu_batch = hs.wgrad_dgu_slot + (size_t)row * twoI;
            host_ctx->wgrad_dgu_a_tma[(size_t)expert * fs.max_batches_per_expert + batch] =
                umma::dg_make_a_desc(dgu_batch, cbs, twoI);
        }
    }
    CUtensorMap* d_wgrad_dgu_a_tma = nullptr;
    CUDA_CHECK(teramoe_caching_alloc(
        reinterpret_cast<void**>(&d_wgrad_dgu_a_tma),
        num_dgu_batch_tmas * sizeof(CUtensorMap)));
    CUDA_CHECK(cudaMemcpyAsync(d_wgrad_dgu_a_tma, host_ctx->wgrad_dgu_a_tma.data(),
                               num_dgu_batch_tmas * sizeof(CUtensorMap),
                               cudaMemcpyHostToDevice, stream));
    hs.wgrad_dgu_a_tma = d_wgrad_dgu_a_tma;

    MegaKernelBackwardState* device_bs;
    CUDA_CHECK(teramoe_caching_alloc(
        reinterpret_cast<void**>(&device_bs), sizeof(MegaKernelBackwardState)));
    CUDA_CHECK(cudaMemcpyAsync(device_bs, &hs, sizeof(MegaKernelBackwardState),
                               cudaMemcpyHostToDevice, stream));

    // Single sync covers all prior async H2Ds (compute_bwd_tma, allocate_state's
    // state+fill_descs, wgrad_dgu_a_tma, device_bs) plus the single fused_fill_kernel
    // launch inside allocate_teramoe_fused_state. This mirrors the forward pattern:
    // batch all async ops, sync once at the end.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EP_HOST_ASSERT(host_context != nullptr);
    *host_context = host_ctx;
    return device_bs;
}

void free_teramoe_fused_backward_state(
    MegaKernelBackwardState* device_bs,
    const MegaKernelBackwardHostContext* host_context
) {
    if (device_bs == nullptr)
        return;

    EP_HOST_ASSERT(device_bs != nullptr);
    MegaKernelBackwardState hs;
    if (host_context != nullptr) {
        hs = host_context->backward_state;
    } else {
        CUDA_CHECK(cudaMemcpy(&hs, device_bs, sizeof(MegaKernelBackwardState), cudaMemcpyDeviceToHost));
    }
    // wgrad_* buffers are borrowed from caller-owned Torch tensors.
    CUDA_CHECK(teramoe_caching_free(hs.compute_bwd_tma));
    CUDA_CHECK(teramoe_caching_free(hs.wgrad_dgu_a_tma));
    free_teramoe_fused_state(
        hs.bwd_device_state,
        host_context != nullptr ? &host_context->bwd_state : nullptr);
    CUDA_CHECK(teramoe_caching_free(device_bs));
}

void free_teramoe_backward_host_context(MegaKernelBackwardHostContext* host_context) {
    delete host_context;
}

// Establish the post-allocation state for every launch. Routing inputs, weights and the
// saved forward activation are immutable across replay and are intentionally preserved.
static void prepare_teramoe_backward_communication_replay_host(
    const TeraMoEState& hs,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    const int combine_hidden_int4 = hs.combine_hidden / (sizeof(int4) / sizeof(__nv_bfloat16));

    // Use the megakernel notify entry so ordinary DeepEP keeps the stock cached_notify path.
    // Besides the cross-rank barriers, this owns the same RDMA/NVL metadata layout and cleanup sizes.
    ::teramoe::teramoe_cached_notify(
        hs.hidden_int4, hs.num_scales, hs.num_topk + 1, hs.num_topk,
        hs.num_ranks, hs.num_logical_channels, 0, nullptr, nullptr, nullptr, nullptr,
        hs.rdma_buffer_ptr, hs.num_max_rdma_chunked_recv_tokens, hs.buffer_ptrs,
        hs.num_max_nvl_chunked_recv_tokens, dispatch_barrier_signal_ptrs, hs.rank,
        stream, hs.num_rdma_bytes, hs.num_nvl_bytes, true, false);
    ::teramoe::teramoe_cached_notify(
        combine_hidden_int4, 0, 0, hs.num_topk,
        hs.num_ranks, hs.num_logical_channels, hs.num_tokens,
        hs.send_rdma_head, hs.combine_rdma_channel_prefix_matrix,
        hs.combine_rdma_rank_prefix_sum, hs.send_nvl_head,
        hs.combine_rdma_buffer_ptr, hs.num_max_combine_rdma_chunked_recv_tokens,
        hs.combine_buffer_ptrs, hs.num_max_combine_nvl_chunked_recv_tokens,
        combine_barrier_signal_ptrs, hs.rank, stream, hs.num_rdma_bytes,
        hs.num_nvl_bytes, false, false);
}

void prepare_megakernel_communication_replay(
    TeraMoEState* device_state,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    TeraMoEState hs;
    CUDA_CHECK(cudaMemcpy(&hs, device_state, sizeof(TeraMoEState), cudaMemcpyDeviceToHost));
    prepare_teramoe_backward_communication_replay_host(
        hs, dispatch_barrier_signal_ptrs, combine_barrier_signal_ptrs, stream);
}


static void initialize_megakernel_launch_state(const TeraMoEState& hs, cudaStream_t stream) {
    const int E = hs.num_local_experts;
    const size_t MTR = (size_t)hs.max_total_recv_tokens;
    const int TK = hs.num_topk;
    const int NG = hs.num_compute_groups;
    const int NR = hs.num_ranks;
    const int kRDMA = hs.num_ranks / NUM_MAX_NVL_PEERS;
    const int NLC = hs.num_logical_channels;
    const int NPC = hs.num_dispatch_channels;
    const int NPUB = hs.num_pub_warps_total;
    const int MBE = (hs.max_tokens_per_expert + hs.compute_batch_size - 1) / hs.compute_batch_size;
    // Per-expert-slot buffers are sized by the compact Σ expert_count (see allocator);
    // reset must cover exactly that, not the legacy num_local_experts*max_tpe.
    const size_t total_slots = (size_t)hs.total_expert_slots;
    const int round_slots = (NLC + NPC - 1) / NPC;
    const int combine_rdma_head_stride = hs.num_tokens * kRDMA;
    const int combine_nvl_head_stride = (hs.num_tokens * TK) * NUM_MAX_NVL_PEERS;

    auto z = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, stream)); };
    auto f = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0xff, bytes, stream)); };

    z(hs.expert_recv_count, (size_t)E * sizeof(int));
    z(hs.expert_slot_ready, total_slots * sizeof(int));
    z(hs.timeout_log_counters, kTimeoutLogCount * sizeof(int));
    z(hs.expert_token_offsets, (size_t)E * sizeof(int));
    f(hs.recv_token_source_info, total_slots * 2 * sizeof(int));
    z(hs.compute_group_barrier, (size_t)NG * sizeof(int));
    z(hs.compute_group_phase, (size_t)NG * sizeof(int));
    z(hs.compute_task_head, sizeof(int));
    z(hs.compute_task_tail, sizeof(int));
    z(hs.compute_task_reserve_tail, sizeof(int));
    z(hs.compute_enqueue_done, sizeof(int));
    z(hs.scheduler_done_count, sizeof(int));
    z(hs.priority_scheduler_done, sizeof(int));
    z(hs.expert_enqueue_cursor, (size_t)E * sizeof(int));
    f(hs.compute_group_task_idx, (size_t)NG * sizeof(int));
    z(hs.token_done_count, MTR * sizeof(int));
    z(hs.gather_claimed, MTR * sizeof(int));
    z(hs.combine_token_ready, MTR * sizeof(int));
    f(hs.gather_ready_queue, MTR * sizeof(int));
    z(hs.gather_ready_head, sizeof(int));
    z(hs.gather_ready_tail, sizeof(int));
    z(hs.gather_ready_reserve_tail, sizeof(int));
    z(hs.gather_scan_cursor, (size_t)COMPUTE_SCHEDULER_SMS * GATHER_SCHED_MAX_WARPS * sizeof(int));
    z(hs.gather_task_count, sizeof(int));
    z(hs.gather_task_tokens, MTR * sizeof(int));
    z(hs.gather_task_nhits, MTR * sizeof(int));
    z(hs.combine_done_count, sizeof(int));
    z(hs.combine_all_done, sizeof(int));
    f(hs.pending_topk_idx, MTR * TK * sizeof(int));
    z(hs.pending_topk_weights, MTR * TK * sizeof(float));
    f(hs.pending_slot, MTR * TK * sizeof(int));
    z(hs.pub_ring, (size_t)NPUB * PUB_RING_DEPTH * sizeof(int));
    z(hs.pub_ring_head, (size_t)NPUB * sizeof(int));
    z(hs.pub_ring_tail, (size_t)NPUB * sizeof(int));
    z(hs.recv_warp_done, (size_t)NPUB * sizeof(int));
    z(hs.publish_warp_done, (size_t)NPUB * sizeof(int));
    z(hs.publish_done_count, sizeof(int));
    z(hs.publish_all_done, sizeof(int));
    z(hs.token_compute_expected, MTR * sizeof(int));
    z(hs.compute_output_slot, total_slots * hs.hidden_dim * sizeof(__nv_bfloat16));
    z(hs.token_nhits, MTR * sizeof(int));
    f(hs.token_slot_list, MTR * TK * sizeof(int));
    z(hs.expert_batch_enqueued, (size_t)E * MBE * sizeof(int));
    f(hs.ready_batch_queue, (size_t)E * MBE * sizeof(int));
    z(hs.ready_batch_reserve_tail, sizeof(int));
    z(hs.expert_batch_ready_count, (size_t)E * MBE * sizeof(int));
    z(hs.combine_input, MTR * hs.hidden_dim * sizeof(__nv_bfloat16));
    z(hs.combine_input_topk_weights, MTR * TK * sizeof(float));
    z(hs.output_accum, (size_t)hs.num_tokens * hs.hidden_dim * sizeof(float));
    f(hs.send_rdma_head, (size_t)NLC * combine_rdma_head_stride * sizeof(int));
    f(hs.send_nvl_head, (size_t)NLC * combine_nvl_head_stride * sizeof(int));
    z(hs.channel_dispatch_done, (size_t)NLC * sizeof(int));
    z(hs.channel_normalized, (size_t)NLC * sizeof(int));
    z(hs.dispatch_channel_barrier, (size_t)NLC * sizeof(int));
    z(hs.rdma_reuse_prelude_done, sizeof(int));
    z(hs.dispatch_round_barrier, (size_t)round_slots * sizeof(int));
    z(hs.combine_channel_barrier, (size_t)NLC * sizeof(int));
    z(hs.recv_rdma_channel_prefix_matrix, (size_t)kRDMA * NLC * sizeof(int));
    z(hs.recv_gbl_channel_prefix_matrix, (size_t)NR * NLC * sizeof(int));
    z(hs.recv_rdma_channel_token_count, (size_t)kRDMA * NLC * sizeof(int));
    z(hs.recv_gbl_channel_token_count, (size_t)NR * NLC * sizeof(int));
}

static void reset_megakernel_post_notify_state(const TeraMoEState& hs, cudaStream_t stream) {
    const int kRDMA = hs.num_ranks / NUM_MAX_NVL_PEERS;
    const int NLC = hs.num_logical_channels;
    const int TK = hs.num_topk;
    const int combine_rdma_head_stride = hs.num_tokens * kRDMA;
    const int combine_nvl_head_stride = (hs.num_tokens * TK) * NUM_MAX_NVL_PEERS;
    auto f = [&](void* p, size_t bytes) { CUDA_CHECK(cudaMemsetAsync(p, 0xff, bytes, stream)); };

    // Backward allocates a fresh state whose ordinary queues/barriers/slot buffers were already
    // initialized by allocate_teramoe_fused_state. The cached_notify replay can write the combine
    // send heads, so restore only those before launching the persistent kernel.
    f(hs.send_rdma_head, (size_t)NLC * combine_rdma_head_stride * sizeof(int));
    f(hs.send_nvl_head, (size_t)NLC * combine_nvl_head_stride * sizeof(int));
}

template <int kNumRDMARanks, int kStage, ComputeDType kComputeDType>
static void launch_teramoe_fused_backward_case(
    MegaKernelBackwardState* device_bs,
    int total_sms,
    int smem_size,
    cudaStream_t stream
) {
    using RdmaCfg = MegaKernelRdmaConfig<kNumRDMARanks>;
    constexpr int kThreads = RdmaCfg::kMegaKernelNumThreads;
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(teramoe_fused_backward_kernel<kNumRDMARanks, kStage, kComputeDType>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    constexpr int num_gather_sms = GATHER_SMS;
    const int launch_total_sms = total_sms + num_gather_sms;
#ifndef DISABLE_SM90_FEATURES
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = launch_total_sms;
    cfg.blockDim = kThreads;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeCooperative;
    attr[0].val.cooperative = 1;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = MK_COMPUTE_CLUSTER_DIM;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, teramoe_fused_backward_kernel<kNumRDMARanks, kStage, kComputeDType>, device_bs));
#else
    teramoe_fused_backward_kernel<kNumRDMARanks, kStage, kComputeDType><<<launch_total_sms, kThreads, smem_size, stream>>>(device_bs);
#endif
    CUDA_CHECK(cudaGetLastError());
}

void prepare_teramoe_backward_communication_replay(
    const MegaKernelBackwardHostContext* host_context,
    int** dispatch_barrier_signal_ptrs,
    int** combine_barrier_signal_ptrs,
    cudaStream_t stream
) {
    EP_HOST_ASSERT(host_context != nullptr);
    EP_HOST_ASSERT(host_context->backward_state.bwd_device_state != nullptr);
    prepare_teramoe_backward_communication_replay_host(
        host_context->bwd_state, dispatch_barrier_signal_ptrs,
        combine_barrier_signal_ptrs, stream);
}

void launch_teramoe_fused_backward(
    MegaKernelBackwardState* backward_state,
    const MegaKernelBackwardHostContext* host_context,
    int total_sms,
    int smem_size,
    int stage,
    ComputeDType compute_dtype,
    cudaStream_t stream
) {
    EP_HOST_ASSERT(backward_state != nullptr);
    EP_HOST_ASSERT(host_context != nullptr);
    EP_HOST_ASSERT(compute_dtype == ComputeDType::kBF16 &&
                   "megakernel debug backward currently supports BF16 only");

    // The backward-specific fills (send_rdma_head, send_nvl_head, grad_input) are now
    // merged into allocate_teramoe_fused_backward_state and executed before the sync there.
    // No separate fused_fill_kernel launch is needed here.

    const MegaKernelBackwardState& hbs = host_context->backward_state;
    const TeraMoEState& hstate = host_context->bwd_state;

    CUDA_CHECK(cudaGetLastError());

    const int num_ranks = hstate.num_ranks;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);

    // The public API receives the physical GPU SM count, while the forward launch uses
    // only its configured role SMs and adds two gather blocks internally. Reconstruct
    // that same active count here; adding gather blocks to the physical SM count makes
    // a cooperative grid larger than the device residency limit.
    const int active_total_sms = hstate.num_dispatch_sms + hstate.num_combine_sms +
        COMPUTE_SCHEDULER_SMS + hstate.num_compute_sms;
    EP_HOST_ASSERT(active_total_sms > 0 && active_total_sms <= total_sms);

#define MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, kStage, kComputeDType) \
    launch_teramoe_fused_backward_case<kNumRDMARanks, kStage, kComputeDType>(backward_state, active_total_sms, smem_size, stream); \
    break

#define MEGAKERNEL_BWD_CASE_WITH_DTYPE(kNumRDMARanks, kComputeDType) \
    switch (stage) { \
        case 1: MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, 1, kComputeDType); \
        case 2: MEGAKERNEL_BWD_STAGE_CASE(kNumRDMARanks, 2, kComputeDType); \
        default: EP_HOST_ASSERT(false && "Unsupported megakernel stage"); \
    } \
    break

#define MEGAKERNEL_BWD_CASE(kNumRDMARanks) \
    MEGAKERNEL_BWD_CASE_WITH_DTYPE(kNumRDMARanks, ComputeDType::kBF16)

    SWITCH_RDMA_RANKS(MEGAKERNEL_BWD_CASE);

    CUDA_CHECK(cudaGetLastError());


    // Weight-gradient operands are consumed by the host-side QuACK grouped GEMMs.
    // Keep this CUDA entry point limited to the megakernel backward itself.

#undef MEGAKERNEL_BWD_CASE
#undef MEGAKERNEL_BWD_CASE_WITH_DTYPE
#undef MEGAKERNEL_BWD_STAGE_CASE
}

namespace detail {
namespace state_cache {

TeraMoEState* alloc_host() {
    return new TeraMoEState{};
}

void copy_host(TeraMoEState* dst, const TeraMoEState* src) {
    *dst = *src;
}

void free_host(TeraMoEState* ptr) {
    delete ptr;
}

}  // namespace state_cache
}  // namespace detail

}  // namespace teramoe

