#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cute/arch/simd_sm100.hpp>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

CUTLASS_DEVICE float dswiglu_fast_sigmoid(float x) {
    float exp_neg_x;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n"
                 : "=f"(exp_neg_x) : "f"(x * -1.4426950408889634f));
    float sig;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n"
                 : "=f"(sig) : "f"(exp_neg_x + 1.0f));
    return sig;
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_swiglu_from_gate(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                             const uint32_t& tmem_base_addr,
                             const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                             const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                             const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                             const cute::TmaDescriptor& tensor_map_cd,
                             const cutlass::bfloat16_t* gate_ptr,
                             const float* route_ptr,
                             uint32_t stride_n) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "SwiGLU store only supports BF16 output");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);
            const auto lane_m_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            const float route = route_ptr[lane_m_idx];

            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                uint32_t tmem_addr = tmem_base_addr +
                                     w * BLOCK_N +
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                auto smem_ptr = smem_base_ptr +
                                epilogue_warp_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                uint32_t values[kNumElemsPerBankGroup];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                    values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7]);
                cutlass::arch::fence_view_async_tmem_load();

                uint32_t packed[kNumElemsPerBankGroup / 2];
                #pragma unroll
                for (uint32_t j = 0; j < kNumElemsPerBankGroup; j += 2) {
                    float up0 = __bfloat162float(__float2bfloat16_rn(*reinterpret_cast<float*>(&values[j + 0])));
                    float up1 = __bfloat162float(__float2bfloat16_rn(*reinterpret_cast<float*>(&values[j + 1])));
                    float gate0 = static_cast<float>(gate_ptr[lane_m_idx * stride_n + n_idx + i * kNumElemsPerBankGroup + j + 0]);
                    float gate1 = static_cast<float>(gate_ptr[lane_m_idx * stride_n + n_idx + i * kNumElemsPerBankGroup + j + 1]);
                    float act0 = (gate0 * (1.0f / (1.0f + expf(-gate0)))) * up0 * route;
                    float act1 = (gate1 * (1.0f / (1.0f + expf(-gate1)))) * up1 * route;
                    auto bf16x2 = __float22bfloat162_rn({act0, act1});
                    packed[j / 2] = *reinterpret_cast<uint32_t*>(&bf16x2);
                }
                ptx::st_shared(smem_ptr, packed[0], packed[1], packed[2], packed[3]);
            }

            if (w == kNumMWaves - 1 and s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cute::tma_store_fence();

            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------
// sm100_store_dswiglu_from_gu - GEMM2 epilogue for backward SwiGLU.
//
// The accumulator tile is grad_act[M,I] from dY @ W_down^T.  The epilogue reads
// the original interleaved gate/up buffer GU[M,2I] from GMEM and writes the
// interleaved dGU[M,2I] result directly through the TMA store path:
//   out[2*j]   = d_silu(gate[j]) * grad_act[j] * up[j]
//   out[2*j+1] = silu(gate[j]) * grad_act[j]
// This removes the separate up_buf materialization and scalar dSwiGLU kernel.
// ---------------------------------------------------------------------------
 template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_dswiglu_from_gu(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                            const uint32_t& tmem_base_addr,
                            const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                            const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                            const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                            const cute::TmaDescriptor& tensor_map_cd,
                            const cutlass::bfloat16_t* gu_ptr,
                            const float* route_ptr,
                            uint32_t shape_m, uint32_t shape_n, uint32_t stride_n) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "dSwiGLU store only supports BF16 output");
    DG_STATIC_ASSERT(STORE_BLOCK_N % (2 * kNumElemsPerBankGroup) == 0, "Invalid dSwiGLU store width");
    DG_STATIC_ASSERT((BLOCK_N * 2) % STORE_BLOCK_N == 0, "Invalid dSwiGLU expansion width");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto out_n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx * 2 + s * STORE_BLOCK_N);
            const auto lane_m_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;

            auto store_bank_group = [&](uint32_t i, uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3) {
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);
                auto smem_ptr = smem_base_ptr +
                                epilogue_warp_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                ptx::st_shared(smem_ptr, p0, p1, p2, p3);
            };

            auto make_dswiglu_pair = [&](float gate, float up, float grad_act) -> uint32_t {
                float sig = 1.0f / (1.0f + __expf(-gate));
                float silu = gate * sig;
                float silu_ga = silu * grad_act;
                float dgate = ((sig + (-silu) * sig) * grad_act + silu_ga) * up;
                auto bf16x2 = __float22bfloat162_rn({dgate, silu_ga});
                return *reinterpret_cast<uint32_t*>(&bf16x2);
            };

            constexpr uint32_t kNumLoadGroups = STORE_BLOCK_N / (2 * kNumElemsPerBankGroup);
            #pragma unroll
            for (uint32_t lg = 0; lg < kNumLoadGroups; ++ lg) {
                const uint32_t out_col0 = s * STORE_BLOCK_N + lg * 2 * kNumElemsPerBankGroup;
                const uint32_t in_col0 = out_col0 / 2;
                uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + in_col0;

                uint32_t grad_values[kNumElemsPerBankGroup];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                    grad_values[0], grad_values[1], grad_values[2], grad_values[3],
                    grad_values[4], grad_values[5], grad_values[6], grad_values[7]);
                cutlass::arch::fence_view_async_tmem_load();

                uint32_t packed[kNumElemsPerBankGroup];
                #pragma unroll
                for (uint32_t q = 0; q < kNumElemsPerBankGroup; ++ q) {
                    float grad_acc = *reinterpret_cast<float*>(&grad_values[q]);
                    float grad_act = __bfloat162float(__float2bfloat16_rn(grad_acc));
                    const uint32_t src_n = base_n_idx + in_col0 + q;
                    if (lane_m_idx < shape_m && src_n < shape_n) {
                        float gate = static_cast<float>(gu_ptr[(size_t)lane_m_idx * stride_n + 2 * src_n + 0]);
                        float up = static_cast<float>(gu_ptr[(size_t)lane_m_idx * stride_n + 2 * src_n + 1]);
                        if (route_ptr != nullptr)
                            grad_act *= route_ptr[lane_m_idx];
                        packed[q] = make_dswiglu_pair(gate, up, grad_act);
                    } else {
                        packed[q] = 0u;
                    }
                }

                store_bank_group(lg * 2 + 0, packed[0], packed[1], packed[2], packed[3]);
                store_bank_group(lg * 2 + 1, packed[4], packed[5], packed[6], packed[7]);
            }

            if (w == kNumMWaves - 1 and s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, out_n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, out_n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

// ---------------------------------------------------------------------------
// sm100_store_dswiglu_packed_from_gu - Sonic-MoE-style GEMM2 epilogue.
//
// CD is a logical uint32[M,I] tensor where each 32-bit element physically stores
// bf16x2(dgate, dup).  The destination bytes are still consumable as bf16[M,2I]
// by GEMM3, but the epilogue store avoids the slow I -> 2I column expansion.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          bool kEmitSideOutputs,
          bool kRouteGradOnly,
          bool kRouteGradPartial,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_dswiglu_packed_from_gu(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                                   const uint32_t& tmem_base_addr,
                                   const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                                   const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                                   const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                                   const cute::TmaDescriptor& tensor_map_cd,
                                   const cutlass::bfloat16_t* gu_ptr,
                                   const float* route_ptr,
                                   cutlass::bfloat16_t* wgrad_act_ptr,
                                   cutlass::bfloat16_t* wgrad_dgu_ptr,
                                   float* route_grad_ptr,
                                   uint32_t shape_m, uint32_t shape_n, uint32_t stride_n) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4 and cute::is_same_v<cd_dtype_t, float>,
                     "packed dSwiGLU store uses 4-byte packed bf16x2 slots");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

    constexpr bool kEmitWgradSideOutputs = kEmitSideOutputs && !kRouteGradOnly;

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    auto bf16_lane = [] (uint32_t v, int lane) -> float {
        return static_cast<float>(reinterpret_cast<const cutlass::bfloat16_t*>(&v)[lane]);
    };
    auto cvt_f32x2_bf16x2 = [] (float lo, float hi) -> uint32_t {
        uint32_t out;
        asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;\n"
                     : "=r"(out) : "f"(hi), "f"(lo));
        return out;
    };
    auto make_dswiglu_pair = [&] (float gate, float up, float grad_act) -> uint32_t {
        float sig = dswiglu_fast_sigmoid(gate);
        float silu = gate * sig;
        float silu_grad = silu * grad_act;
        float d_silu_grad = (sig - silu * sig) * grad_act + silu_grad;
        return cvt_f32x2_bf16x2(d_silu_grad * up, silu_grad);
    };
    auto make_dswiglu_pair2 = [&] (uint32_t gu0, uint32_t gu1, uint32_t grad01, float route,
                                  uint32_t& out0, uint32_t& out1,
                                  float2& activation, float2& grad_raw) {
        float2 gate = {bf16_lane(gu0, 0), bf16_lane(gu1, 0)};
        float2 up = {bf16_lane(gu0, 1), bf16_lane(gu1, 1)};
        grad_raw = {bf16_lane(grad01, 0), bf16_lane(grad01, 1)};
        float2 grad = {grad_raw.x * route, grad_raw.y * route};
        float2 sig = {dswiglu_fast_sigmoid(gate.x), dswiglu_fast_sigmoid(gate.y)};
        float2 silu, silu_grad, sig_minus_silu_sig, d_silu_grad, dgate;
        float2 neg_sig = {-sig.x, -sig.y};
        cute::mul(silu, gate, sig);
        activation = {silu.x * up.x, silu.y * up.y};
        cute::mul(silu_grad, silu, grad);
        cute::fma(sig_minus_silu_sig, silu, neg_sig, sig);
        cute::fma(d_silu_grad, sig_minus_silu_sig, grad, silu_grad);
        cute::mul(dgate, d_silu_grad, up);
        out0 = cvt_f32x2_bf16x2(dgate.x, silu_grad.x);
        out1 = cvt_f32x2_bf16x2(dgate.y, silu_grad.y);
    };

    const uint32_t gu_packed_stride = stride_n / 2;
    const uint32_t* gu_packed = reinterpret_cast<const uint32_t*>(gu_ptr);

    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
        const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
        const auto lane_m_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
        const bool lane_m_valid = lane_m_idx < shape_m;
        float route = 1.0f;
        if (lane_m_valid) {
            if constexpr (kEmitSideOutputs)
                route = route_ptr[lane_m_idx];
            else if (route_ptr != nullptr)
                route = route_ptr[lane_m_idx];
        }
        float row_route_grad = 0.0f;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx + s * STORE_BLOCK_N);
            const bool full_store = (m_idx + STORE_BLOCK_M <= shape_m) &&
                                    (base_n_idx + (s + 1) * STORE_BLOCK_N <= shape_n);
            float side_route_grad = 0.0f;

            if (full_store) {
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                    auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                    col ^= row % (kSwizzleCDMode / 16);

                    const uint32_t src_n = base_n_idx + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                    uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                    auto smem_ptr = smem_base_ptr +
                                    epilogue_warp_idx * 32 * kSwizzleCDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                    uint32_t grad_values[kNumElemsPerBankGroup];
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                        grad_values[0], grad_values[1], grad_values[2], grad_values[3]);
                    cutlass::arch::fence_view_async_tmem_load();

                    uint32_t grad01 = static_cast<uint32_t>(math::cast_into_bf16_and_pack(grad_values[0], grad_values[1]));
                    uint32_t grad23 = static_cast<uint32_t>(math::cast_into_bf16_and_pack(grad_values[2], grad_values[3]));
                    const size_t gu_base = (size_t)lane_m_idx * gu_packed_stride + src_n;
                    uint4 gu4 = make_uint4(0u, 0u, 0u, 0u);
                    if (lane_m_valid)
                        gu4 = reinterpret_cast<const uint4*>(gu_packed)[gu_base / 4];

                    uint32_t packed[kNumElemsPerBankGroup];
                    float2 activation01 = {0.0f, 0.0f}, activation23 = {0.0f, 0.0f};
                    float2 grad_raw01 = {0.0f, 0.0f}, grad_raw23 = {0.0f, 0.0f};
                    make_dswiglu_pair2(gu4.x, gu4.y, grad01, route, packed[0], packed[1], activation01, grad_raw01);
                    make_dswiglu_pair2(gu4.z, gu4.w, grad23, route, packed[2], packed[3], activation23, grad_raw23);
                    if constexpr (kEmitSideOutputs) {
                        side_route_grad += grad_raw01.x * activation01.x + grad_raw01.y * activation01.y +
                                           grad_raw23.x * activation23.x + grad_raw23.y * activation23.y;
                    }
                    if constexpr (kEmitWgradSideOutputs) {
                        uint32_t* wgrad_dgu_packed = reinterpret_cast<uint32_t*>(wgrad_dgu_ptr);
                        if (wgrad_act_ptr != nullptr) {
                            wgrad_act_ptr[(size_t)lane_m_idx * shape_n + src_n + 0] = cutlass::bfloat16_t(route * activation01.x);
                            wgrad_act_ptr[(size_t)lane_m_idx * shape_n + src_n + 1] = cutlass::bfloat16_t(route * activation01.y);
                            wgrad_act_ptr[(size_t)lane_m_idx * shape_n + src_n + 2] = cutlass::bfloat16_t(route * activation23.x);
                            wgrad_act_ptr[(size_t)lane_m_idx * shape_n + src_n + 3] = cutlass::bfloat16_t(route * activation23.y);
                        }
                        if (wgrad_dgu_packed != nullptr) {
                            wgrad_dgu_packed[(size_t)lane_m_idx * shape_n + src_n + 0] = packed[0];
                            wgrad_dgu_packed[(size_t)lane_m_idx * shape_n + src_n + 1] = packed[1];
                            wgrad_dgu_packed[(size_t)lane_m_idx * shape_n + src_n + 2] = packed[2];
                            wgrad_dgu_packed[(size_t)lane_m_idx * shape_n + src_n + 3] = packed[3];
                        }
                    }
                    ptx::st_shared(smem_ptr, packed[0], packed[1], packed[2], packed[3]);
                }
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                    auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                    col ^= row % (kSwizzleCDMode / 16);

                    const uint32_t src_n = base_n_idx + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                    uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                    auto smem_ptr = smem_base_ptr +
                                    epilogue_warp_idx * 32 * kSwizzleCDMode +
                                    row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                    uint32_t grad_values[kNumElemsPerBankGroup];
                    cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_addr,
                        grad_values[0], grad_values[1], grad_values[2], grad_values[3]);
                    cutlass::arch::fence_view_async_tmem_load();

                    uint32_t grad01 = static_cast<uint32_t>(math::cast_into_bf16_and_pack(grad_values[0], grad_values[1]));
                    uint32_t grad23 = static_cast<uint32_t>(math::cast_into_bf16_and_pack(grad_values[2], grad_values[3]));

                    uint32_t packed[kNumElemsPerBankGroup];
                    #pragma unroll
                    for (uint32_t q = 0; q < kNumElemsPerBankGroup; ++ q)
                        packed[q] = 0u;
                    if (lane_m_idx < shape_m) {
                        uint32_t* wgrad_dgu_packed = nullptr;
                        if constexpr (kEmitWgradSideOutputs)
                            wgrad_dgu_packed = reinterpret_cast<uint32_t*>(wgrad_dgu_ptr);
                        #pragma unroll
                        for (uint32_t q = 0; q < kNumElemsPerBankGroup; ++ q) {
                            if (src_n + q < shape_n) {
                                const uint32_t gu = gu_packed[(size_t)lane_m_idx * gu_packed_stride + src_n + q];
                                const uint32_t grad = q < 2 ? grad01 : grad23;
                                const uint32_t grad_lane = q & 1;
                                const float gate = bf16_lane(gu, 0);
                                const float up = bf16_lane(gu, 1);
                                const float grad_raw = bf16_lane(grad, grad_lane);
                                packed[q] = make_dswiglu_pair(gate, up, grad_raw * route);
                                if constexpr (kEmitSideOutputs) {
                                    const float sig = dswiglu_fast_sigmoid(gate);
                                    const float activation = gate * sig * up;
                                    side_route_grad += grad_raw * activation;
                                    if constexpr (kEmitWgradSideOutputs) {
                                        if (wgrad_act_ptr != nullptr)
                                            wgrad_act_ptr[(size_t)lane_m_idx * shape_n + src_n + q] = cutlass::bfloat16_t(route * activation);
                                        if (wgrad_dgu_packed != nullptr)
                                            wgrad_dgu_packed[(size_t)lane_m_idx * shape_n + src_n + q] = packed[q];
                                    }
                                }
                            }
                        }
                    }
                    ptx::st_shared(smem_ptr, packed[0], packed[1], packed[2], packed[3]);
                }
            }

            if constexpr (kEmitSideOutputs) {
                row_route_grad += side_route_grad;
            }

            if (w == kNumMWaves - 1 and s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
        if constexpr (kEmitSideOutputs) {
            if (lane_m_valid && route_grad_ptr != nullptr) {
                if constexpr (kRouteGradPartial) {
                    const uint32_t partial_stride = (shape_n + BLOCK_N - 1) / BLOCK_N;
                    const uint32_t partial_idx = base_n_idx / BLOCK_N;
                    route_grad_ptr[(size_t)lane_m_idx * partial_stride + partial_idx] = row_route_grad;
                } else {
                    atomicAdd(&route_grad_ptr[lane_m_idx], row_route_grad);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sm100_store_swiglu_interleaved — quack-style in-TMEM interleaved SwiGLU store.
//
// The accumulator N dimension (BLOCK_N columns of TMEM) is the ELEMENT-INTERLEAVED
// gate/up product GU = A @ Wgu^T, where Wgu rows are [g0,u0,g1,u1,...] (gran=1).
// So a BLOCK_N=128 TMEM tile holds 64 (gate,up) pairs: TMEM col 2j = gate_j,
// 2j+1 = up_j. This mirrors quack's GemmGatedMixin.epi_visit_subtile:
//   tRS_rD_pair = flat_divide(tRS_rD, 2); gate = pair[0], up = pair[1];
//   out = act_fn(gate, up)   (== silu(gate)*up here, times per-row route).
//
// Unlike sm100_store_swiglu_from_gate (which reads gate back from GMEM), gate and
// up BOTH live in the same TMEM tile, so nothing round-trips through GMEM. The
// output is HALF width (BLOCK_N/2 act columns) and uses the STANDARD DeepGEMM
// 128B-swizzle bf16 store path, identical to sm100_store_cd, so the CD TMA
// descriptor is a normal act[M,I] descriptor (STORE_BLOCK_N=64, swizzle 128B).
//
// Works unchanged for 1-CTA and 2-CTA: the interleave is along N while the 2-CTA
// split is along M (mcast-on-A) or N-tiles (mcast-on-B); either way each CTA owns
// a full BLOCK_N interleaved tile locally, so flat gate/up pairing stays intact.
//
//   STORE_BLOCK_N : OUTPUT act atom width (=64 for bf16 128B swizzle)
//   requires BLOCK_N == 2 * (act block width) and one store per M-wave.
//   base_n_idx : GU-space (2I) tile base = n_block*BLOCK_N. act col = base_n_idx/2.
// ---------------------------------------------------------------------------
template <uint32_t BLOCK_M, uint32_t BLOCK_N,
          uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N,
          uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages,
          uint32_t kNumUMMAStoreThreads,
          GemmType kGemmType, bool kWithAccumulation,
          typename cd_dtype_t,
          typename epilogue_type_t,
          typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_swiglu_interleaved(const utils::PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx,
                               const uint32_t& tmem_base_addr,
                               const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                               const cute::TmaDescriptor& tensor_map_cd,
                               const float* route_ptr,
                               cutlass::bfloat16_t* preact_ptr = nullptr,
                               const int* preact_recv_idx = nullptr,
                               const int* preact_topk_idx = nullptr,
                               uint32_t num_topk = 0,
                               uint32_t preact_stride = 0,
                               uint32_t valid_rows = 0xffffffffu) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);  // 8 for bf16
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8 and cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Interleaved SwiGLU store only supports BF16 output");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    // OUTPUT act width per block = BLOCK_N / 2 (one act col per gate/up pair).
    constexpr uint32_t OUT_BLOCK_N = BLOCK_N / 2;
    DG_STATIC_ASSERT(OUT_BLOCK_N % STORE_BLOCK_N == 0, "Invalid interleaved block sizes");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
    #pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++ w) {
        constexpr uint32_t kNumStores = OUT_BLOCK_N / STORE_BLOCK_N;
        #pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++ s, advance_store_pipeline()) {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            // act output column: GU tile base halved (interleave collapses 2->1).
            const auto n_idx = epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(base_n_idx / 2 + s * STORE_BLOCK_N);
            const auto lane_m_idx = m_idx + epilogue_warp_idx * 32 + lane_idx;
            const bool lane_m_valid = lane_m_idx < valid_rows;
            const float route = lane_m_valid ? route_ptr[lane_m_idx] : 0.0f;
            cutlass::bfloat16_t* preact_row = nullptr;
            if (preact_ptr != nullptr && lane_m_valid && preact_recv_idx != nullptr && preact_topk_idx != nullptr) {
                const int recv_idx = preact_recv_idx[lane_m_idx];
                const int topk_idx = preact_topk_idx[lane_m_idx];
                if (recv_idx >= 0 && topk_idx >= 0)
                    preact_row = preact_ptr + ((size_t)recv_idx * num_topk + topk_idx) * preact_stride;
            }

            #pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++ i) {
                // Output swizzle math IDENTICAL to sm100_store_cd (8 act bf16 per group).
                auto bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto row = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto col = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                // This bank group emits 8 act cols starting at out_col0; each act col j
                // reads TMEM cols (2*out_col_j)=gate, (+1)=up. 8 act -> 16 interleaved
                // TMEM cols -> two 8-wide TMEM loads.
                const uint32_t out_col0 = s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;  // in OUT_BLOCK_N space
                const uint32_t tmem_col0 = 2u * out_col0;
                uint32_t tmem_addr0 = tmem_base_addr + w * BLOCK_N + tmem_col0;       // cols c..c+7  = g,u,g,u,g,u,g,u
                uint32_t tmem_addr1 = tmem_addr0 + kNumElemsPerBankGroup;             // cols c+8..c+15

                uint32_t v0[kNumElemsPerBankGroup];
                uint32_t v1[kNumElemsPerBankGroup];
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr0,
                    v0[0], v0[1], v0[2], v0[3], v0[4], v0[5], v0[6], v0[7]);
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr1,
                    v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], v1[6], v1[7]);
                cutlass::arch::fence_view_async_tmem_load();

                // v0 -> act out_col0+0..3 ; v1 -> act out_col0+4..7.
                // Within each 8-wide load: [g0,u0,g1,u1,g2,u2,g3,u3].
                auto swiglu_pack = [&](const uint32_t* v, uint32_t act_col_base, uint32_t& p0, uint32_t& p1) {
                    float g0 = *reinterpret_cast<const float*>(&v[0]);
                    float u0 = *reinterpret_cast<const float*>(&v[1]);
                    float g1 = *reinterpret_cast<const float*>(&v[2]);
                    float u1 = *reinterpret_cast<const float*>(&v[3]);
                    float g2 = *reinterpret_cast<const float*>(&v[4]);
                    float u2 = *reinterpret_cast<const float*>(&v[5]);
                    float g3 = *reinterpret_cast<const float*>(&v[6]);
                    float u3 = *reinterpret_cast<const float*>(&v[7]);
                    float a0 = (g0 * (1.0f / (1.0f + expf(-g0)))) * u0 * route;
                    float a1 = (g1 * (1.0f / (1.0f + expf(-g1)))) * u1 * route;
                    float a2 = (g2 * (1.0f / (1.0f + expf(-g2)))) * u2 * route;
                    float a3 = (g3 * (1.0f / (1.0f + expf(-g3)))) * u3 * route;
                    auto b01 = __float22bfloat162_rn({a0, a1});
                    auto b23 = __float22bfloat162_rn({a2, a3});
                    p0 = *reinterpret_cast<uint32_t*>(&b01);
                    p1 = *reinterpret_cast<uint32_t*>(&b23);
                    if (preact_row != nullptr) {
                        const uint32_t act_base = base_n_idx / 2 + act_col_base;
                        const uint32_t act_cols = preact_stride / 2;
                        // The 4 (gate,up) pairs are 8 contiguous BF16 = 16 bytes at
                        // preact_row + 2*act_base. act_base is a multiple of 4 here
                        // (base_n_idx is a multiple of BLOCK_N, act_col_base a multiple
                        // of 4), and preact rows are 2I-strided (16B aligned), so the
                        // address is 16B aligned. Emit one vectorized int4 store
                        // instead of 8 scalar BF16 stores. Verified bytewise-identical
                        // to the scalar path and ~2.8x cheaper in compute_ref
                        // preact_save_bench (M=1024,I=3072).
                        auto pack_gu = [](float g, float u) -> uint32_t {
                            __nv_bfloat162 b = __float22bfloat162_rn({g, u});
                            return *reinterpret_cast<uint32_t*>(&b);
                        };
                        if (act_base + 3 < act_cols) {
                            uint4 packed = make_uint4(pack_gu(g0, u0), pack_gu(g1, u1),
                                                      pack_gu(g2, u2), pack_gu(g3, u3));
                            *reinterpret_cast<uint4*>(&preact_row[2 * act_base]) = packed;
                        } else {
                            if (act_base + 0 < act_cols) {
                                preact_row[2 * (act_base + 0)] = cutlass::bfloat16_t(g0);
                                preact_row[2 * (act_base + 0) + 1] = cutlass::bfloat16_t(u0);
                            }
                            if (act_base + 1 < act_cols) {
                                preact_row[2 * (act_base + 1)] = cutlass::bfloat16_t(g1);
                                preact_row[2 * (act_base + 1) + 1] = cutlass::bfloat16_t(u1);
                            }
                            if (act_base + 2 < act_cols) {
                                preact_row[2 * (act_base + 2)] = cutlass::bfloat16_t(g2);
                                preact_row[2 * (act_base + 2) + 1] = cutlass::bfloat16_t(u2);
                            }
                            if (act_base + 3 < act_cols) {
                                preact_row[2 * (act_base + 3)] = cutlass::bfloat16_t(g3);
                                preact_row[2 * (act_base + 3) + 1] = cutlass::bfloat16_t(u3);
                            }
                        }
                    }
                };
                uint32_t packed[kNumElemsPerBankGroup / 2];  // 4 x bf16x2 = 8 act
                swiglu_pack(v0, out_col0, packed[0], packed[1]);
                swiglu_pack(v1, out_col0 + 4, packed[2], packed[3]);

                auto smem_ptr = smem_base_ptr +
                                epilogue_warp_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                ptx::st_shared(smem_ptr, packed[0], packed[1], packed[2], packed[3]);
            }

            if (w == kNumMWaves - 1 and s == kNumStores - 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
                if constexpr (kGemmType == GemmType::Batched) {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                } else {
                    using cute_tma_t = cute::conditional_t<kWithAccumulation,
                                            cute::SM90_TMA_REDUCE_ADD_2D, cute::SM90_TMA_STORE_2D>;
                    cute_tma_t::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K_,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages_,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t kKAlignment,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation,           typename cd_dtype_t,
          uint64_t kTensorCoreUtilControl,
          bool kFuseSwiGLU = false,
          bool kFuseSwiGLUInterleaved = false,
          uint32_t kPhysicalNumThreads = kNumNonEpilogueThreads + kNumEpilogueThreads,
          bool kFuseDSwiGLU = false,
          bool kFuseDSwiGLUPacked = false,
          bool kEmitDSwiGLUSideOutputs = false,
          bool kDSwiGLURouteGradOnly = false,
          bool kDSwiGLURouteGradPartial = false>

CUTLASS_GLOBAL void __launch_bounds__(kPhysicalNumThreads, 1)
sm100_bf16_gemm_impl(int* grouped_layout,
                     uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                     const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                     const cutlass::bfloat16_t* gate_ptr = nullptr,
                     const float* route_ptr = nullptr,
                     uint32_t stride_n = 0,
                     cutlass::bfloat16_t* wgrad_act_ptr = nullptr,
                     cutlass::bfloat16_t* wgrad_dgu_ptr = nullptr,
                     float* route_grad_ptr = nullptr,
                     cutlass::bfloat16_t* preact_ptr = nullptr,
                     const int* preact_recv_idx = nullptr,
                     const int* preact_topk_idx = nullptr,
                     uint32_t preact_num_topk = 0,
                     uint32_t preact_stride = 0) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    // Enlarge `BLOCK_K` for some cases
    // NOTES: this is for reducing the `umma_arrive()` overhead
    constexpr bool kDoMergeStages =
        kNumStages_ >= 8 and kGemmType == GemmType::Normal and
        kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K;
    // Ensure there are at least `kNumMinStages` stages after merge
    constexpr uint32_t kNumMinStages = 8;
    constexpr uint32_t kNumStagesPerMerge = kDoMergeStages ? kNumStages_ / kNumMinStages : 1;
    constexpr uint32_t BLOCK_K = BLOCK_K_ * kNumStagesPerMerge;
    constexpr uint32_t kNumStages = kNumStages_ / kNumStagesPerMerge;

    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // C/D type: BF16 and FP32 are supported, with or without accumulation
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>, "Invalid C/D data dtype");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    DG_STATIC_ASSERT(BLOCK_K_ == 64, "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(kKAlignment % UMMA_K == 0, "K alignment must be divisible by UMMA K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    constexpr uint32_t kNumTMAStoreStages = 2;
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // Share memory sizes
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE = LOAD_BLOCK_M * BLOCK_K * sizeof(cutlass::bfloat16_t);
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE = LOAD_BLOCK_N * BLOCK_K * sizeof(cutlass::bfloat16_t);
    DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 and SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0, 
                     "Shared memory of A/B must be aligned to 1024 bytes");
    DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    static constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(nv_bfloat16);
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory out of bound for UMMA");

    // Real tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = kNumEpilogueStages * UMMA_N;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols>();
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // D/A/B shared memory
    auto smem_cd = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    auto smem_a  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_CD_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + SMEM_CD_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE));
    auto full_barriers              = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers             = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto tmem_full_barriers         = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + i); });
    auto tmem_empty_barriers        = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages * 2 + kNumEpilogueStages + i); });
    auto tensor_core_full_barrier   = barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2;

    // Fill the tensor memory pointer
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 + 1);
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive only at the leader CTA
            full_barriers[i]->init(kNumMulticast);
            // Arrive at all CTAs
            empty_barriers[i]->init(1);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            tmem_full_barriers[i]->init(1);
            // Arrive only at the leader CTA
            tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
        }
        if constexpr (kTensorCoreUtilControl < 100)
            tensor_core_full_barrier->init(1);

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    // NOTES: BF16 has no SF, so `kSFKSpan` is unused here; pass `kKAlignment` explicitly to avoid relying on the default.
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kEnsureZeroPadding, kKAlignment, kKAlignment>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0, tensor_core_phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = (stage_idx + 1) % kNumStages;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                empty_barriers[stage_idx]->wait(phase ^ 1);

                // Compute offsets
                // NOTES: the group is always concatenated with the outer dimension
                uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked), sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                uint32_t n_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K), sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                // NOTES: `k_idx` is actually the k index default for K-major, while `k_b_idx` may be MN-major
                // And for all m-grouped GEMMs, A must be K-majored
                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                uint32_t k_idx = k_block_idx * BLOCK_K;
                uint32_t k_a_idx = scheduler.template get_global_idx<(kMajorA == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::MN), sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                // Issue TMAs
                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                if constexpr (kMajorA == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], k_a_idx, m_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorA == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_a, full_barriers[stage_idx], smem_a[stage_idx], m_idx, k_a_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::K)
                    tma::copy<BLOCK_K, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], k_b_idx, n_idx, kNumMulticast, batch_idx);
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, cutlass::bfloat16_t, kIsBatchedMM>(
                        &tensor_map_b, full_barriers[stage_idx], smem_b[stage_idx], n_idx, k_b_idx, kNumMulticast, batch_idx);

                // Arrive at full barriers
                constexpr uint32_t kNumArrivalBytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;
                if (is_leader_cta) {
                    full_barriers[stage_idx]->arrive_and_expect_tx(kNumArrivalBytes * kNumMulticast);
                } else {
                    full_barriers[stage_idx]->arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                                                                UMMA_M, UMMA_N, kMajorA, kMajorB>();

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        // Merged stages only happens in NT normal GEMM cases
        constexpr uint32_t BLOCK_ATOM_K = BLOCK_K / kNumStagesPerMerge;
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_ATOM_K, kSwizzleAMode>(smem_a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_ATOM_K, kSwizzleBMode>(smem_b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait tensor memory empty barrier arrival
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // UMMA and empty barrier arrival alias
            auto umma_arrive = [](const uint64_t* barrier) {
                if constexpr (kNumMulticast == 1) {
                    cutlass::arch::umma_arrive(barrier);
                } else {
                    constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                    cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                }
            };
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            constexpr bool kMayHaveTailKBlock = is_k_grouped_contiguous(kGemmType) ? (kKAlignment % BLOCK_K != 0) : (SHAPE_K == 0 or SHAPE_K % BLOCK_K != 0);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait TMA arrival
                full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                // Issue UMMA in the leader CTA
                using mma_t = cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_F16BF16_SS, ptx::SM100_MMA_F16BF16_2x1SM_SS>;
                const auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
                const auto a_desc_base_lo = __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
                const auto b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
                if (cute::elect_one_sync()) {
                    auto issue_umma = [&]<uint32_t kUMMAKIdx>() {
                        constexpr uint32_t kAtomKIdx = kUMMAKIdx * UMMA_K / BLOCK_ATOM_K;
                        constexpr uint32_t kInnerKIdx = kUMMAKIdx * UMMA_K % BLOCK_ATOM_K;
                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, cutlass::bfloat16_t>(
                                        a_desc_base_lo, kAtomKIdx * LOAD_BLOCK_M * BLOCK_ATOM_K, kInnerKIdx);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, cutlass::bfloat16_t>(
                                        b_desc_base_lo, kAtomKIdx * LOAD_BLOCK_N * BLOCK_ATOM_K, kInnerKIdx);
                        if (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N,
                                       kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N,
                                       kUMMAKIdx > 0 or k_block_idx > 0, runtime_instr_desc);
                        }
                    };
                    auto issue_full_k_block = [&]() {
                        utils::for_each_static_until<BLOCK_K / UMMA_K>(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), issue_umma);
                    };

                    if constexpr (kMayHaveTailKBlock) {
                        auto issue_tail_k_block = [&](const uint32_t& remaining_k) {
                            const auto num_valid_umma_k = math::ceil_div(remaining_k, UMMA_K);
                            // Prefix expansion uses switch only for small cases to avoid long SASS.
                            utils::for_each_static_prefix(std::make_integer_sequence<uint32_t, BLOCK_K / UMMA_K>(), num_valid_umma_k, issue_umma);
                        };
                        const auto is_last_k_block = k_block_idx == num_total_k_blocks - 1;
                        if (is_last_k_block) {
                            const auto remaining_k = scheduler.current_shape_k - k_block_idx * BLOCK_K;
                            if (remaining_k < BLOCK_K)
                                issue_tail_k_block(remaining_k);
                            else
                                issue_full_k_block();
                        } else {
                            issue_full_k_block();
                        }
                    } else {
                        issue_full_k_block();
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);

                // Let tensor cores relax for lower possibility of frequency drop
                DG_STATIC_ASSERT(kTensorCoreUtilControl > 0, "Invalid tensor utilization control");
                if constexpr (kTensorCoreUtilControl < 100) {
                    // For utilization control
                    umma_arrive(reinterpret_cast<uint64_t*>(tensor_core_full_barrier));
                    __syncwarp();

                    // Wait for last UMMA to be done
                    tensor_core_full_barrier->wait(tensor_core_phase);
                    tensor_core_phase ^= 1;

                    // Sleep for certain cycles
                    constexpr static uint64_t kNumUMMACycles = (2ull * UMMA_M * UMMA_N * BLOCK_K) / 8192ull;
                    constexpr static uint64_t kNumDummyCycles = (100ull - kTensorCoreUtilControl) * kNumUMMACycles / kTensorCoreUtilControl;
                    const auto start_clock = clock64();
                    if (cute::elect_one_sync())
                        while (clock64() - start_clock < kNumDummyCycles) {}
                    __syncwarp();
                }
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            // Load from tensor memory into registers, and write shared memory with STSM
            const auto tmem_base_addr = accum_stage_idx * UMMA_N;
            const auto base_m_idx = scheduler.template get_global_idx<
                (not is_m_grouped_contiguous(kGemmType)), sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            if constexpr (kSwapAB) {
                const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                    kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                    kGemmType, kWithAccumulation,
                    cd_dtype_t, epilogue::transform::EpilogueIdentity>
                (smem_cd, tma_stage_idx, tmem_base_addr,
                 base_m_idx, base_n_idx, scheduler.current_group_idx,
                 effective_m,
                 epilogue_warp_idx, lane_idx,
                 tmem_empty_barriers[accum_stage_idx],
                 tensor_map_cd);
            } else {
                if constexpr (kFuseDSwiGLUPacked) {
                    sm100_store_dswiglu_packed_from_gu<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue::transform::EpilogueIdentity,
                        kEmitDSwiGLUSideOutputs, kDSwiGLURouteGradOnly, kDSwiGLURouteGradPartial>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd, gate_ptr, route_ptr,
                     wgrad_act_ptr, wgrad_dgu_ptr, route_grad_ptr,
                     shape_m, shape_n, stride_n);
                } else if constexpr (kFuseDSwiGLU) {
                    sm100_store_dswiglu_from_gu<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue::transform::EpilogueIdentity>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd, gate_ptr, route_ptr, shape_m, shape_n, stride_n);
                } else if constexpr (kFuseSwiGLUInterleaved) {
                    sm100_store_swiglu_interleaved<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue::transform::EpilogueIdentity>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd, route_ptr,
                     preact_ptr, preact_recv_idx, preact_topk_idx,
                     preact_num_topk, preact_stride, shape_m);
                } else if constexpr (kFuseSwiGLU) {
                    sm100_store_swiglu_from_gate<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue::transform::EpilogueIdentity>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd, gate_ptr, route_ptr, stride_n);
                } else {
                    epilogue::sm100_store_cd<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N,
                        kSwizzleCDMode, kNumTMAStoreStages, kNumUMMAStoreThreads,
                        kGemmType, kWithAccumulation,
                        cd_dtype_t, epilogue::transform::EpilogueIdentity>
                    (smem_cd, tma_stage_idx, tmem_base_addr,
                     base_m_idx, base_n_idx, scheduler.current_group_idx,
                     epilogue_warp_idx, lane_idx,
                     tmem_empty_barriers[accum_stage_idx],
                     tensor_map_cd);
                }
            }
        }
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
