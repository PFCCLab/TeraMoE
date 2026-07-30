#include "moe_extension.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>

#include <chrono>
#include <algorithm>
#include <memory>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

namespace shared_memory {
void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id = idx;
        access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void SharedMemoryAllocator::free(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
}  // namespace shared_memory

namespace deep_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      shared_memory_allocator(use_fabric) {
    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif

    // Get device info
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Memory layout per rank:
        // [nvl_data | barrier_signals | buffer_ptrs | barrier_signal_ptrs |
        //  combine_nvl_data | combine_barrier_signals | combine_buffer_ptrs | combine_barrier_signal_ptrs]
        int64_t per_half = num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes;
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank], 2 * per_half);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);

        auto base = static_cast<uint8_t*>(buffer_ptrs[nvl_rank]);

        // Dispatch region
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(base + num_nvl_bytes);
        buffer_ptrs_gpu = reinterpret_cast<void**>(base + num_nvl_bytes + barrier_signal_bytes);
        barrier_signal_ptrs_gpu = reinterpret_cast<int**>(base + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // Combine region
        auto combine_base = base + per_half;
        combine_buffer_ptrs[nvl_rank] = combine_base;
        combine_barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(combine_base + num_nvl_bytes);
        combine_buffer_ptrs_gpu = reinterpret_cast<void**>(combine_base + num_nvl_bytes + barrier_signal_bytes);
        combine_barrier_signal_ptrs_gpu = reinterpret_cast<int**>(combine_base + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
        CUDA_CHECK(cudaMemsetAsync(combine_barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier (both dispatch and combine)
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        intranode::barrier(combine_barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local buffer and error flag
        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
    }

    // Free NVSHMEM
#ifndef DISABLE_NVSHMEM
    if (is_available() and num_rdma_bytes > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        if (rdma_reuse_dispatch_quiet_done != nullptr)
            internode::free(rdma_reuse_dispatch_quiet_done);
        if (rdma_reuse_combine_clear_done != nullptr)
            internode::free(rdma_reuse_combine_clear_done);
        if (enable_shrink) {
            internode::free(mask_buffer_ptr);
            internode::free(sync_buffer_ptr);
        }
        internode::finalize();
    }
#endif

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());

        int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
        int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
        int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);
        int64_t per_half = num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes;

        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                // Dispatch barrier signal
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
                // Combine pointers
                combine_buffer_ptrs[i] = static_cast<uint8_t*>(buffer_ptrs[i]) + per_half;
                combine_barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + per_half + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(combine_buffer_ptrs_gpu, combine_buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(combine_barrier_signal_ptrs_gpu, combine_barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory
#ifndef DISABLE_NVSHMEM
    if (num_rdma_bytes > 0) {
        // Initialize NVSHMEM
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        internode::barrier();

        // RDMA pool. Megakernel combine now reuses this single region (the in-kernel combine
        // prelude serializes dispatch->clean->combine and barriers across same-nvl peers), so the
        // old second "combine" half is no longer needed. Regular low-latency APIs also use only
        // this region.
        int64_t rdma_alloc_bytes = num_rdma_bytes;
        rdma_buffer_ptr = internode::alloc(rdma_alloc_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, rdma_alloc_bytes));

        // RDMA-buffer-reuse mailboxes: symmetric memory, indexed by rdma_rank.
        // Allocated like the RDMA pool so peers can IBGDA-write into the local copy.
        {
            int64_t mailbox_bytes = static_cast<int64_t>(num_rdma_ranks) * sizeof(int);
            rdma_reuse_dispatch_quiet_done = reinterpret_cast<int*>(
                internode::alloc(mailbox_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            rdma_reuse_combine_clear_done = reinterpret_cast<int*>(
                internode::alloc(mailbox_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            CUDA_CHECK(cudaMemset(rdma_reuse_dispatch_quiet_done, 0, mailbox_bytes));
            CUDA_CHECK(cudaMemset(rdma_reuse_combine_clear_done, 0, mailbox_bytes));
        }

        // Allocate and clean shrink buffer
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
        }

        // Barrier
        internode::barrier();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
            num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
        <= num_nvl_bytes);
    intranode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        comm_stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        rank_prefix_matrix,
                        channel_prefix_matrix,
                        recv_x,
                        recv_src_idx,
                        recv_channel_prefix_matrix,
                        send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_expert,
                         cached_channel_prefix_matrix,
                         cached_rank_prefix_matrix,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     comm_stream);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       comm_stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4,
                                 num_scales,
                                 num_topk,
                                 num_topk,
                                 num_ranks,
                                 num_channels,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                 rank,
                                 comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes,
                                 true,
                                 low_latency_mode);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens,
                                   num_worst_tokens,
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(),
                                   recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(),
                                   recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode);

        // Synchronize total received tokens and tokens per expert
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(),
                        recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(),
                        recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
                        cached_mode,
                        comm_stream,
                        num_channels,
                        low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        recv_x,
                        rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_rdma_rank,
                         num_tokens_per_expert,
                         cached_rdma_channel_prefix_matrix,
                         cached_recv_rdma_rank_prefix_sum,
                         cached_gbl_channel_prefix_matrix,
                         cached_recv_gbl_rank_prefix_sum,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales,
                         recv_rdma_channel_prefix_matrix,
                         recv_gbl_channel_prefix_matrix,
                         send_rdma_head,
                         send_nvl_head,
                         recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_meta,
    const torch::Tensor& is_combined_token_in_rank,
    const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum,
    const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& combined_rdma_head,
    const torch::Tensor& combined_nvl_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
#ifndef DISABLE_NVSHMEM
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4,
                             0,
                             0,
                             num_topk,
                             num_ranks,
                             num_channels,
                             num_combined_tokens,
                             combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(),
                             rdma_rank_prefix_sum.data_ptr<int>(),
                             combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr,
                             config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu,
                             config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu,
                             rank,
                             comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes,
                             false,
                             low_latency_mode);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());

    internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       combined_x.data_ptr(),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(),
                       combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(),
                       rdma_channel_prefix_matrix.data_ptr<int>(),
                       rdma_rank_prefix_sum.data_ptr<int>(),
                       gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens,
                       num_combined_tokens,
                       hidden,
                       num_topk,
                       rdma_buffer_ptr,
                       config.num_max_rdma_chunked_send_tokens,
                       config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens,
                       rank,
                       num_ranks,
                       comm_stream,
                       num_channels,
                       low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        src_meta,
                        is_combined_token_in_rank,
                        rdma_channel_prefix_matrix,
                        rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        combined_x,
                        combined_rdma_head,
                        combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                           clean_meta_0.second,
                                           clean_meta_1.first,
                                           clean_meta_1.second,
                                           rank,
                                           num_ranks,
                                           mask_buffer_ptr,
                                           sync_buffer_ptr,
                                           at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x,
                             const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool round_scale,
                             bool use_ue8m0,
                             bool async,
                             bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    auto packed_recv_src_info =
        torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<topk_idx_t>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank,
            num_ranks,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace,
            num_device_sms,
            launch_stream,
            phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer,
                              buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(),
                              topk_idx.data_ptr<topk_idx_t>(),
                              topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(),
                              layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first,
                              next_clean_meta.second,
                              num_combined_tokens,
                              hidden,
                              num_max_dispatch_tokens_per_rank,
                              num_topk,
                              num_experts,
                              rank,
                              num_ranks,
                              use_logfmt,
                              workspace,
                              num_device_sms,
                              launch_stream,
                              phases,
                              zero_copy);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
}

TeraMoEAutogradContext::TeraMoEAutogradContext(::teramoe::TeraMoEState* state,
                                                     int num_tokens,
                                                     int hidden_dim,
                                                     int intermediate_dim,
                                                     int num_topk,
                                                     int num_local_experts,
                                                     std::vector<int> expert_counts)
    : state_(state),
      num_tokens_(num_tokens),
      hidden_dim_(hidden_dim),
      intermediate_dim_(intermediate_dim),
      num_topk_(num_topk),
      num_local_experts_(num_local_experts),
      expert_counts_(std::move(expert_counts)) {}

TeraMoEAutogradContext::~TeraMoEAutogradContext() {
#ifndef DISABLE_NVSHMEM
    if (state_ != nullptr)
        ::teramoe::free_teramoe_fused_state(
            state_, cached_host_state_);
#endif
    if (cached_host_state_ != nullptr)
        ::teramoe::detail::state_cache::free_host(cached_host_state_);
}

::teramoe::TeraMoEState* TeraMoEAutogradContext::state() const {
    return state_;
}

int TeraMoEAutogradContext::num_tokens() const {
    return num_tokens_;
}

int TeraMoEAutogradContext::hidden_dim() const {
    return hidden_dim_;
}

int TeraMoEAutogradContext::intermediate_dim() const {
    return intermediate_dim_;
}

int TeraMoEAutogradContext::num_topk() const {
    return num_topk_;
}

int TeraMoEAutogradContext::num_local_experts() const {
    return num_local_experts_;
}

const std::vector<int>& TeraMoEAutogradContext::expert_counts() const {
    return expert_counts_;
}

void TeraMoEAutogradContext::retain_layout_tensors(std::vector<torch::Tensor> tensors) {
    retained_layout_tensors_ = std::move(tensors);
}

const ::teramoe::TeraMoEState& TeraMoEAutogradContext::cached_host_state() const {
    EP_HOST_ASSERT(cached_host_state_ != nullptr);
    return *cached_host_state_;
}

void TeraMoEAutogradContext::set_cached_host_state(const ::teramoe::TeraMoEState& hs) {
    if (cached_host_state_ != nullptr)
        ::teramoe::detail::state_cache::free_host(cached_host_state_);
    cached_host_state_ = ::teramoe::detail::state_cache::alloc_host();
    ::teramoe::detail::state_cache::copy_host(cached_host_state_, &hs);
}

std::tuple<torch::Tensor, std::shared_ptr<TeraMoEAutogradContext>> Buffer::teramoe_fused_forward_impl(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& W_gateup,
    const torch::Tensor& W_down,
    int num_experts,
    int num_dispatch_sms,
    int num_combine_sms,
    int total_sms,
    int stage,
    const Config& dispatch_config,
    const Config& combine_config,
    const pybind11::object& hidden_states_scales_obj,
    const pybind11::object& W_gateup_fp8_obj,
    const pybind11::object& W_down_fp8_obj,
    const pybind11::object& W_gateup_fp8_sf_obj,
    const pybind11::object& W_down_fp8_sf_obj,
    bool retain_state,
    int compute_batch_size,
    int combine_start_head_percent) {
#ifndef DISABLE_NVSHMEM
    auto optional_tensor = [](const pybind11::object& obj) -> torch::Tensor {
        if (obj.is_none()) return torch::Tensor();
        return obj.cast<torch::Tensor>();
    };
    torch::Tensor hidden_states_scales = optional_tensor(hidden_states_scales_obj);
    torch::Tensor W_gateup_fp8 = optional_tensor(W_gateup_fp8_obj);
    torch::Tensor W_down_fp8 = optional_tensor(W_down_fp8_obj);
    torch::Tensor W_gateup_fp8_sf = optional_tensor(W_gateup_fp8_sf_obj);
    torch::Tensor W_down_fp8_sf = optional_tensor(W_down_fp8_sf_obj);

    pybind11::gil_scoped_release release;

    // Input validation
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(W_gateup.dim() == 3 and W_gateup.is_contiguous());
    EP_HOST_ASSERT(W_down.dim() == 3 and W_down.is_contiguous());
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(num_ranks > 0 and num_experts % num_ranks == 0);
    EP_HOST_ASSERT(stage >= 1 and stage <= 3);

    const bool use_fp8_compute = x.scalar_type() == torch::kFloat8_e4m3fn;
    EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16 || use_fp8_compute);
    EP_HOST_ASSERT(use_fp8_compute == hidden_states_scales.defined());
    if (!use_fp8_compute) {
        EP_HOST_ASSERT(W_gateup.scalar_type() == torch::kBFloat16);
        EP_HOST_ASSERT(W_down.scalar_type() == torch::kBFloat16);
    }

    const int num_tokens = x.size(0);
    const int hidden_dim = x.size(1);
    const int hidden_int4 = hidden_dim * x.element_size() / sizeof(int4);
    const int num_topk = topk_idx.size(1);
    const int intermediate_dim = W_gateup.size(1) / 2;
    const int num_local_experts = num_experts / num_ranks;

    EP_HOST_ASSERT(topk_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights.size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights.size(1) == num_topk);
    EP_HOST_ASSERT((hidden_dim * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(W_gateup.size(0) == num_local_experts and W_down.size(0) == num_local_experts);
    EP_HOST_ASSERT(W_gateup.size(1) % 2 == 0);
    EP_HOST_ASSERT(W_gateup.size(2) == hidden_dim);
    EP_HOST_ASSERT(W_down.size(1) == hidden_dim and W_down.size(2) == intermediate_dim);

    const bool has_all_fp8_tensors = W_gateup_fp8.defined() && W_down_fp8.defined() &&
        W_gateup_fp8_sf.defined() && W_down_fp8_sf.defined();
    const bool has_any_fp8_tensors = W_gateup_fp8.defined() || W_down_fp8.defined() ||
        W_gateup_fp8_sf.defined() || W_down_fp8_sf.defined();
    EP_HOST_ASSERT(!has_any_fp8_tensors || has_all_fp8_tensors);
    EP_HOST_ASSERT(use_fp8_compute == has_all_fp8_tensors);
    const bool build_fp8_compute = use_fp8_compute;
    const int fp8_hidden_scale_k_packed = (hidden_dim + 128 * 4 - 1) / (128 * 4);
    const int fp8_intermediate_scale_k_packed = (intermediate_dim + 128 * 4 - 1) / (128 * 4);

    const uint32_t* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (use_fp8_compute) {
        EP_HOST_ASSERT(hidden_states_scales.dim() == 2 and hidden_states_scales.is_contiguous());
        EP_HOST_ASSERT(hidden_states_scales.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(hidden_states_scales.size(0) == num_tokens);
        EP_HOST_ASSERT(hidden_states_scales.size(1) == fp8_hidden_scale_k_packed);
        num_scales = static_cast<int>(hidden_states_scales.size(1));
        x_scales_ptr = reinterpret_cast<const uint32_t*>(hidden_states_scales.data_ptr<int>());
        scale_token_stride = static_cast<int>(hidden_states_scales.stride(0));
        scale_hidden_stride = static_cast<int>(hidden_states_scales.stride(1));
    }

    if (build_fp8_compute) {
        EP_HOST_ASSERT(W_gateup_fp8.dim() == 3 and W_gateup_fp8.is_contiguous());
        EP_HOST_ASSERT(W_down_fp8.dim() == 3 and W_down_fp8.is_contiguous());
        EP_HOST_ASSERT(W_gateup_fp8_sf.dim() == 3 and W_gateup_fp8_sf.is_contiguous());
        EP_HOST_ASSERT(W_down_fp8_sf.dim() == 3 and W_down_fp8_sf.is_contiguous());
        EP_HOST_ASSERT(W_gateup_fp8.scalar_type() == torch::kUInt8 || W_gateup_fp8.scalar_type() == torch::kFloat8_e4m3fn);
        EP_HOST_ASSERT(W_down_fp8.scalar_type() == torch::kUInt8 || W_down_fp8.scalar_type() == torch::kFloat8_e4m3fn);
        EP_HOST_ASSERT(W_gateup_fp8_sf.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(W_down_fp8_sf.scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(W_gateup_fp8.size(0) == num_local_experts && W_gateup_fp8.size(1) == 2 * intermediate_dim && W_gateup_fp8.size(2) == hidden_dim);
        EP_HOST_ASSERT(W_down_fp8.size(0) == num_local_experts && W_down_fp8.size(1) == hidden_dim && W_down_fp8.size(2) == intermediate_dim);
        EP_HOST_ASSERT(W_gateup_fp8_sf.size(0) == num_local_experts && W_gateup_fp8_sf.size(1) == 2 * intermediate_dim && W_gateup_fp8_sf.size(2) == fp8_hidden_scale_k_packed);
        EP_HOST_ASSERT(W_down_fp8_sf.size(0) == num_local_experts && W_down_fp8_sf.size(1) == hidden_dim && W_down_fp8_sf.size(2) == fp8_intermediate_scale_k_packed);
    }

    // SM allocation: dispatch -> combine -> scheduler -> compute groups -> gather, leaving any remainder reserved.
    constexpr int compute_group_size = teramoe_config::kComputeGroupSize;
    constexpr int compute_cluster_dim = teramoe_config::kComputeClusterDim;
    constexpr int compute_scheduler_sms = teramoe_config::kComputeSchedulerSms;
    constexpr int gather_sms = teramoe_config::kGatherSms;
    const int compute_available_sms = total_sms - num_dispatch_sms - num_combine_sms - compute_scheduler_sms;
    const int num_compute_groups = compute_available_sms / compute_group_size;
    const int num_compute_sms = num_compute_groups * compute_group_size;
    const int num_forwarder_sms = compute_available_sms - num_compute_sms;  // remaining reserved SMs, not launched as a role
    const int active_total_sms = num_dispatch_sms + num_combine_sms + compute_scheduler_sms + num_compute_sms;
    EP_HOST_ASSERT(num_compute_groups > 0);
    EP_HOST_ASSERT(num_combine_sms % 2 == 0);
    EP_HOST_ASSERT(num_dispatch_sms % 2 == 0);
    EP_HOST_ASSERT(compute_cluster_dim == 1 || (active_total_sms % 2 == 0 && "active_total_sms must be even for MK_COMPUTE_KERNEL=2 cluster_dim=2"));

    // MegaKernel uses the same DeepEP config objects as the baseline path, but
    // keeps dispatch and combine parameters separate just like original DeepEP.
    const int num_physical_channels = num_dispatch_sms / 2;  // even/odd SM pairing in dispatch_worker
    // Logical channels = physical channels * stage. notify_dispatch / get_dispatch_layout see
    // the expanded logical-channel count so the prefix matrices are laid out per logical channel,
    // consistent with the dispatch worker (num_logical_channels_per_physical = kStage) and the
    // combine worker. Restores logical!=physical support (was clamped to physical in "add backward").
    const int num_logical_channels = num_physical_channels * stage;
    const int num_channels = num_logical_channels;  // DeepEP notify sees the expanded logical-channel count.
    const int dispatch_num_max_rdma_chunked_send_tokens = dispatch_config.num_max_rdma_chunked_send_tokens;
    const int dispatch_num_max_rdma_chunked_recv_tokens = dispatch_config.num_max_rdma_chunked_recv_tokens;
    const int dispatch_num_max_nvl_chunked_send_tokens = dispatch_config.num_max_nvl_chunked_send_tokens;
    const int dispatch_num_max_nvl_chunked_recv_tokens = dispatch_config.num_max_nvl_chunked_recv_tokens;
    const int combine_num_max_rdma_chunked_send_tokens = combine_config.num_max_rdma_chunked_send_tokens;
    const int combine_num_max_rdma_chunked_recv_tokens = combine_config.num_max_rdma_chunked_recv_tokens;
    const int combine_num_max_nvl_chunked_send_tokens = combine_config.num_max_nvl_chunked_send_tokens;
    const int combine_num_max_nvl_chunked_recv_tokens = combine_config.num_max_nvl_chunked_recv_tokens;

    // Step 1: Compute dispatch layout
    auto num_tokens_per_rank = torch::empty({num_ranks}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_expert_t = torch::empty({num_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, torch::dtype(torch::kBool).device(torch::kCUDA));

    auto stream = at::cuda::getCurrentCUDAStream();

    layout::get_dispatch_layout(
        topk_idx.data_ptr<topk_idx_t>(),
        num_tokens_per_rank.data_ptr<int>(),
        num_tokens_per_rdma_rank.data_ptr<int>(),
        num_tokens_per_expert_t.data_ptr<int>(),
        is_token_in_rank.data_ptr<bool>(),
        num_tokens, num_topk, num_ranks, num_experts, stream);

    // Step 2: notify_dispatch — exchange metadata via NVSHMEM
    auto rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    *moe_recv_counter = -1;
    *moe_recv_rdma_counter = -1;
    for (int i = 0; i < num_local_experts; ++i)
        moe_recv_expert_counter[i] = -1;

    // Keep notify_dispatch's buffer cleanup aligned with the megakernel logical-channel layout.
    // The prefix matrices are regenerated below for logical channels, but the cleanup range must
    // cover every logical-channel RDMA/NVL buffer slice before the megakernel starts using them.
    internode::notify_dispatch(
        num_tokens_per_rank.data_ptr<int>(),
        moe_recv_counter_mapped,
        num_ranks,
        num_tokens_per_rdma_rank.data_ptr<int>(),
        moe_recv_rdma_counter_mapped,
        num_tokens_per_expert_t.data_ptr<int>(),
        moe_recv_expert_counter_mapped,
        num_experts,
        is_token_in_rank.data_ptr<bool>(),
        num_tokens,
        0,  // num_worst_tokens
        num_logical_channels,
        hidden_int4,
        num_scales,
        num_topk + 1,  // MK-v7 uses num_topk+1 int slots (src_token_idx + topk_idx)
        1,  // expert_alignment
        rdma_channel_prefix_matrix.data_ptr<int>(),
        recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(),
        recv_gbl_rank_prefix_sum.data_ptr<int>(),
        rdma_buffer_ptr,
        dispatch_num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu,
        dispatch_num_max_nvl_chunked_recv_tokens,
        barrier_signal_ptrs_gpu,
        rank,
        stream,
        num_rdma_bytes,
        num_nvl_bytes,
        low_latency_mode);
    const int source_meta_bytes = internode::get_source_meta_bytes();
    auto get_num_bytes_per_token = [&](int num_topk_idx, int num_topk_weights) {
        return align_up(hidden_int4 * static_cast<int>(sizeof(int4)) + source_meta_bytes +
                            num_topk_idx * static_cast<int>(sizeof(int)) +
                            num_topk_weights * static_cast<int>(sizeof(float)),
                        static_cast<int>(sizeof(int4)));
    };
    auto get_rdma_bytes = [&](int num_topk_idx, int num_topk_weights, int num_max_rdma_chunked_recv_tokens) {
        int64_t data_bytes = static_cast<int64_t>(get_num_bytes_per_token(num_topk_idx, num_topk_weights)) *
            num_max_rdma_chunked_recv_tokens * num_rdma_ranks * 2 * num_logical_channels;
        int64_t meta_bytes = static_cast<int64_t>(NUM_MAX_NVL_PEERS * 2 + 4) *
            num_rdma_ranks * 2 * num_logical_channels * sizeof(int);
        return data_bytes + meta_bytes;
    };
    auto get_nvl_bytes = [&](int num_topk_idx, int num_topk_weights, int num_max_nvl_chunked_recv_tokens) {
        int64_t data_bytes = static_cast<int64_t>(get_num_bytes_per_token(num_topk_idx, num_topk_weights)) *
            num_max_nvl_chunked_recv_tokens * NUM_MAX_NVL_PEERS * num_logical_channels;
        int64_t meta_bytes = static_cast<int64_t>(NUM_MAX_NVL_PEERS) *
            (2 * num_rdma_ranks + 2) * num_logical_channels * sizeof(int);
        return data_bytes + meta_bytes;
    };
    EP_HOST_ASSERT(get_rdma_bytes(num_topk + 1, num_topk, dispatch_num_max_rdma_chunked_recv_tokens) +
                   get_rdma_bytes(0, num_topk, combine_num_max_rdma_chunked_recv_tokens) <= num_rdma_bytes);
    EP_HOST_ASSERT(get_nvl_bytes(num_topk + 1, num_topk, dispatch_num_max_nvl_chunked_recv_tokens) <= num_nvl_bytes);
    EP_HOST_ASSERT(get_nvl_bytes(0, num_topk, combine_num_max_nvl_chunked_recv_tokens) <= num_nvl_bytes);

    // Busy-wait for metadata exchange to complete
    auto wait_start = std::chrono::steady_clock::now();
    while (*moe_recv_counter == -1) {
        auto elapsed = std::chrono::steady_clock::now() - wait_start;
        EP_HOST_ASSERT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() < 30);
    }

    // Clean the combine region the DeepEP way: a clean-only cached_notify (num_combined_tokens=0,
    // null heads => no head normalization). Megakernel keeps this host notify for the NVL combine
    // metadata clean + cross-rank barrier, while the shared RDMA region is cleaned in-kernel after
    // dispatch drains.
    //   RDMA : reuses the single symmetric region (rdma_buffer_ptr); host RDMA clean is skipped.
    //   NVL  combine half : combine_buffer_ptrs_gpu           (base + per_half, see Buffer ctor)
    {
        // RDMA reuse: combine now shares the single RDMA region. This clean-only cached_notify skips
        // RDMA metadata clean so the in-kernel combine prelude is the single owner of that clear.
        void* combine_rdma_ptr = rdma_buffer_ptr;
        ::teramoe::teramoe_cached_notify(hidden_int4,
                                 0,          // num_scales (combine payload carries no scales)
                                 0,          // num_topk_idx
                                 num_topk,   // num_topk_weights
                                 num_ranks,
                                 num_logical_channels,
                                 0,          // num_combined_tokens => clean + barrier only
                                 nullptr,    // combined_rdma_head
                                 nullptr,    // rdma_channel_prefix_matrix
                                 nullptr,    // rdma_rank_prefix_sum
                                 nullptr,    // combined_nvl_head
                                 combine_rdma_ptr,
                                 combine_num_max_rdma_chunked_recv_tokens,
                                 combine_buffer_ptrs_gpu,
                                 combine_num_max_nvl_chunked_recv_tokens,
                                 combine_barrier_signal_ptrs_gpu,
                                 rank,
                                 stream,
                                 num_rdma_bytes,
                                 num_nvl_bytes,
                                 true,       // is_cached_dispatch: clean-only. Makes cached_notify's
                                             // sm_id==1/>=2 head-normalization warps return early
                                            // (teramoe_notify.cu cached_notify), so the null head/prefix
                                            // pointers are never dereferenced. Only sm_id==0 runs,
                                             // which does the NVL clean + cross-rank barrier.
                                             // get_nvl_clean_meta ignores this flag, so the combine
                                             // clean range is unchanged. MK does its own head
                                             // normalization inside the combine worker.
                                 true);     // skip_rdma_clean: MK prelude owns the RDMA clean on
                                             // the shared combine region after dispatch drains.
    }

    // Step 3: Allocate and launch megakernel v7
    const int max_total_recv_tokens = *moe_recv_counter;
    // Per-expert slot budget. A recv token's topk experts are distinct, so a token hits any
    // given local expert AT MOST ONCE. Hence expert_token_offsets[e] (incremented once per
    // (token, expert) hit) is bounded by the number of distinct recv tokens = max_total_recv_tokens.
    // Worst case: all recv tokens route to the same expert -> cap = max_total_recv_tokens.
    // (The previous *num_topk was an over-allocation assuming a token could occupy num_topk slots
    //  on the same expert, which cannot happen.) The in-kernel overflow check `slot >= cap` -> trap()
    // remains as a safety net.

    const int max_tokens_per_expert = std::max(1, max_total_recv_tokens);
    AT_CUDA_CHECK(cudaGetLastError());

    // Per-local-expert received-token counts for compact slot packing (P0 forward path).
    // moe_recv_expert_counter was populated by notify_dispatch above and equals each local
    // expert's exact received-token count (= the in-kernel atomicAdd final value), so the
    // allocator can pack per-expert slot buffers by Σ count instead of num_local_experts*max_tpe.
    std::vector<int> mk_expert_counts(num_local_experts);
    for (int i = 0; i < num_local_experts; ++i) {
        int c = moe_recv_expert_counter[i];  // volatile int* -> plain int read
        mk_expert_counts[i] = c < 0 ? 0 : c;
    }

    // The combine worker writes directly into the returned Torch tensor. This removes the
    // state-owned output allocation and the device-to-device clone that used to follow it.
    auto result = torch::empty(
        {num_tokens, hidden_dim}, x.options().dtype(torch::kBFloat16));

    // host_state_out receives the complete host-side TeraMoEState snapshot from
    // allocate_teramoe_fused_state. The backward uses this directly (via context) instead
    // of a synchronous D2H cudaMemcpy from the device state.
    ::teramoe::TeraMoEState* fwd_host_state_ptr =
        ::teramoe::detail::state_cache::alloc_host();
    auto allocate_state = [&](auto allocator) {
        return allocator(
        reinterpret_cast<const int4*>(x.data_ptr()),
        x_scales_ptr,
        reinterpret_cast<const topk_idx_t*>(topk_idx.data_ptr()),
        topk_weights.data_ptr<float>(),
        is_token_in_rank.data_ptr<bool>(),
        rdma_channel_prefix_matrix.data_ptr<int>(),
        recv_rdma_rank_prefix_sum.data_ptr<int>(),
        gbl_channel_prefix_matrix.data_ptr<int>(),
        recv_gbl_rank_prefix_sum.data_ptr<int>(),
        rdma_buffer_ptr,
        buffer_ptrs_gpu,
        combine_buffer_ptrs_gpu,
        num_tokens,
        hidden_dim,
        hidden_int4,
        intermediate_dim,
        num_scales,
        num_topk,
        num_experts,
        num_local_experts,
        num_ranks,
        rank,
        scale_token_stride,
        scale_hidden_stride,
        dispatch_num_max_rdma_chunked_send_tokens,
        dispatch_num_max_rdma_chunked_recv_tokens,
        dispatch_num_max_nvl_chunked_send_tokens,
        dispatch_num_max_nvl_chunked_recv_tokens,
        combine_num_max_rdma_chunked_send_tokens,
        combine_num_max_rdma_chunked_recv_tokens,
        combine_num_max_nvl_chunked_send_tokens,
        combine_num_max_nvl_chunked_recv_tokens,
        reinterpret_cast<const __nv_bfloat16*>(W_gateup.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W_down.data_ptr()),
        use_fp8_compute ? megakernel::ComputeDType::kFP8E4M3 : megakernel::ComputeDType::kBF16,
        build_fp8_compute ? W_gateup_fp8.data_ptr() : nullptr,
        build_fp8_compute ? W_down_fp8.data_ptr() : nullptr,
        build_fp8_compute ? reinterpret_cast<const uint32_t*>(W_gateup_fp8_sf.data_ptr<int>()) : nullptr,
        build_fp8_compute ? reinterpret_cast<const uint32_t*>(W_down_fp8_sf.data_ptr<int>()) : nullptr,
        num_dispatch_sms,
        num_forwarder_sms,
        num_compute_sms,
        num_combine_sms,
        num_logical_channels,
        max_tokens_per_expert,
        max_total_recv_tokens > 0 ? max_total_recv_tokens : 1,
        num_rdma_bytes,
        num_nvl_bytes,
        mk_expert_counts.data(),
        moe_recv_expert_counter_mapped,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<int4*>(result.data_ptr()),
        nullptr,
        fwd_host_state_ptr,
        rdma_reuse_dispatch_quiet_done,
        rdma_reuse_combine_clear_done,
        1 /* rdma_reuse_prelude_enable (forward) */,
        compute_batch_size,
        combine_start_head_percent);
    };

    void* state = static_cast<void*>(allocate_state(::teramoe::allocate_teramoe_fused_state));

    AT_CUDA_CHECK(cudaGetLastError());

    // Compute shared memory size
    // Match internode.cu dispatch/combine dynamic shared memory requirements.
    int smem_size = std::max(NUM_MAX_NVL_PEERS * 16384, 24 * 9248);
    auto compute_dtype = use_fp8_compute
        ? megakernel::ComputeDType::kFP8E4M3
        : megakernel::ComputeDType::kBF16;

    // Reset RDMA-reuse mailboxes before launch. Barrier so every rank observes the cleared
    // mailbox before any peer's combine prelude writes into it during this iteration.
    if (num_rdma_bytes > 0 && rdma_reuse_dispatch_quiet_done != nullptr) {
        int64_t mailbox_bytes = static_cast<int64_t>(num_rdma_ranks) * sizeof(int);
        CUDA_CHECK(cudaMemsetAsync(rdma_reuse_dispatch_quiet_done, 0, mailbox_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(rdma_reuse_combine_clear_done, 0, mailbox_bytes, stream));
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // internode::barrier();
    }

    ::teramoe::launch_teramoe_fused_forward(
        static_cast<::teramoe::TeraMoEState*>(state),
        fwd_host_state_ptr,
        active_total_sms, smem_size, stage, compute_dtype, stream);
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::shared_ptr<TeraMoEAutogradContext> context;
    if (retain_state) {
        EP_HOST_ASSERT(!use_fp8_compute && "TeraMOE training state currently requires BF16 mode");
        context = std::make_shared<TeraMoEAutogradContext>(
            static_cast<::teramoe::TeraMoEState*>(state),
            num_tokens, hidden_dim, intermediate_dim, num_topk, num_local_experts,
            mk_expert_counts);
        // The backward re-runs dispatch/combine on a fresh v7 state and only reuses the
        // saved-activation buffers (bwd_fc1_input / bwd_preact / fwd_slot_map) and expert_count
        // from this forward state. Release the forward-only working buffers (recv_tokens,
        // compute_output_slot, combine_input, gemm_workspace, output_accum, combine/dispatch
        // heads, ...) now so they don't stay resident across the forward->backward gap.
        // The returned Torch tensor owns combined_x and is not released with the state.
        // Free transient BEFORE caching and freeing fwd_host_state_ptr.
        ::teramoe::free_teramoe_forward_transients_from_host(fwd_host_state_ptr);
        // Cache the host-side state snapshot (with transient pointers already freed) so
        // backward can skip the synchronous D2H.
        context->set_cached_host_state(*fwd_host_state_ptr);
        ::teramoe::detail::state_cache::free_host(fwd_host_state_ptr);
        fwd_host_state_ptr = nullptr;
        // The TeraMoEState keeps only raw data_ptr()s into these notify_dispatch-produced
        // layout tensors, and the backward re-runs dispatch/combine off that state. Retain them so
        // they outlive this forward call (otherwise the backward reads dangling/zeroed memory —
        // notably rdma_channel_prefix_matrix, which stalls the backward NVL dispatch).
        context->retain_layout_tensors({
            is_token_in_rank,
            rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            topk_idx,
            topk_weights,
        });
    } else {
        ::teramoe::free_teramoe_fused_state(
            static_cast<::teramoe::TeraMoEState*>(state),
            fwd_host_state_ptr);
        if (fwd_host_state_ptr != nullptr)
            ::teramoe::detail::state_cache::free_host(fwd_host_state_ptr);
    }

    return {result, context};
#else
    EP_HOST_ASSERT(false && "teramoe_forward requires NVSHMEM support");
    return {torch::Tensor(), nullptr};
#endif
}

torch::Tensor Buffer::teramoe_forward(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& W_gateup,
    const torch::Tensor& W_down,
    int num_experts,
    int num_dispatch_sms,
    int num_combine_sms,
    int total_sms,
    int stage,
    const Config& dispatch_config,
    const Config& combine_config,
    const pybind11::object& hidden_states_scales_obj,
    const pybind11::object& W_gateup_fp8_obj,
    const pybind11::object& W_down_fp8_obj,
    const pybind11::object& W_gateup_fp8_sf_obj,
    const pybind11::object& W_down_fp8_sf_obj,
    int compute_batch_size,
    int combine_start_head_percent) {
    return std::get<0>(teramoe_fused_forward_impl(
        x, topk_idx, topk_weights, W_gateup, W_down, num_experts,
        num_dispatch_sms, num_combine_sms, total_sms, stage, dispatch_config,
        combine_config, hidden_states_scales_obj, W_gateup_fp8_obj,
        W_down_fp8_obj, W_gateup_fp8_sf_obj, W_down_fp8_sf_obj, false,
        compute_batch_size, combine_start_head_percent));
}

std::tuple<torch::Tensor, std::shared_ptr<TeraMoEAutogradContext>> Buffer::teramoe_forward_train(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& W_gateup,
    const torch::Tensor& W_down,
    int num_experts,
    int num_dispatch_sms,
    int num_combine_sms,
    int total_sms,
    int stage,
    const Config& dispatch_config,
    const Config& combine_config,
    int compute_batch_size,
    int combine_start_head_percent) {
    pybind11::object none = pybind11::none();
    return teramoe_fused_forward_impl(
        x, topk_idx, topk_weights, W_gateup, W_down, num_experts,
        num_dispatch_sms, num_combine_sms, total_sms, stage,
        dispatch_config, combine_config, none, none, none, none, none, true,
        compute_batch_size, combine_start_head_percent);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
Buffer::teramoe_backward(
    const std::shared_ptr<TeraMoEAutogradContext>& context,
    const torch::Tensor& grad_output,
    const std::optional<torch::Tensor>& grad_topk_weights,
    int total_sms,
    int stage) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(context != nullptr && context->state() != nullptr);
    EP_HOST_ASSERT(grad_output.dim() == 2 && grad_output.is_contiguous());
    EP_HOST_ASSERT(grad_output.is_cuda() && grad_output.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(stage >= 1 && stage <= 2);

    pybind11::gil_scoped_release release;
    const int num_tokens = context->num_tokens();
    const int hidden = context->hidden_dim();
    const int intermediate = context->intermediate_dim();
    const int num_topk = context->num_topk();
    const int num_local_experts = context->num_local_experts();
    auto grad_input = torch::empty_like(grad_output);
    auto bf16_options = grad_output.options();
    auto fp32_options = grad_output.options().dtype(torch::kFloat32);
    // These are fully overwritten by the QuACK wgrad postprocess, so zero-init is unnecessary.
    auto grad_w_gateup = torch::empty(
        {num_local_experts, 2 * intermediate, hidden}, bf16_options);
    auto grad_w_down = torch::empty(
        {num_local_experts, hidden, intermediate}, bf16_options);
    torch::Tensor grad_topk_weights_out;
    if (grad_topk_weights.has_value()) {
        grad_topk_weights_out = *grad_topk_weights;
        EP_HOST_ASSERT(grad_topk_weights_out.is_cuda() && grad_topk_weights_out.is_contiguous());
        EP_HOST_ASSERT(grad_topk_weights_out.scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(grad_topk_weights_out.dim() == 2);
        EP_HOST_ASSERT(num_tokens == grad_topk_weights_out.size(0));
        EP_HOST_ASSERT(num_topk == grad_topk_weights_out.size(1));
    } else {
        grad_topk_weights_out = torch::zeros({num_tokens, num_topk}, fp32_options);
    }
    const std::vector<int>& expert_token_counts = context->expert_counts();
    EP_HOST_ASSERT(static_cast<int>(expert_token_counts.size()) == num_local_experts);
    std::vector<int> h_cu_seqlens_k(num_local_experts + 1, 0);
    size_t total_slots = 0;
    for (int e = 0; e < num_local_experts; ++e) {
        total_slots += (size_t)expert_token_counts[e];
        h_cu_seqlens_k[e + 1] = static_cast<int>(total_slots);
    }
    const size_t alloc_slots = std::max<size_t>(total_slots, 1);
    const int two_i = 2 * intermediate;
    const int compute_batch_size = ::teramoe::get_teramoe_compute_batch_size_default();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto cu_options = torch::TensorOptions().dtype(torch::kInt32).device(grad_output.device());
    auto cu_seqlens_k = torch::empty({num_local_experts + 1}, cu_options);
    CUDA_CHECK(cudaMemcpyAsync(cu_seqlens_k.data_ptr(), h_cu_seqlens_k.data(),
                               h_cu_seqlens_k.size() * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    auto scratch_x_backing = torch::empty({(int64_t)alloc_slots, hidden}, bf16_options);
    auto scratch_act_backing = torch::empty({(int64_t)alloc_slots, intermediate}, bf16_options);
    auto scratch_dz_backing = torch::empty({(int64_t)alloc_slots, hidden}, bf16_options);
    auto scratch_dgu_backing = torch::empty(
        {(int64_t)alloc_slots + compute_batch_size, two_i}, bf16_options);
    auto scratch_x = scratch_x_backing.narrow(0, 0, total_slots);
    auto scratch_act = scratch_act_backing.narrow(0, 0, total_slots);
    auto scratch_dz = scratch_dz_backing.narrow(0, 0, total_slots);
    auto scratch_dgu = scratch_dgu_backing.narrow(0, 0, total_slots);
    ::teramoe::MegaKernelBackwardHostContext* backward_host_context = nullptr;
    auto* backward_state = ::teramoe::allocate_teramoe_fused_backward_state(
        context->state(), grad_output.data_ptr(), grad_input.data_ptr(),
        grad_w_gateup.data_ptr(), grad_w_down.data_ptr(), grad_topk_weights_out.data_ptr(),
        scratch_x_backing.data_ptr(), scratch_act_backing.data_ptr(), scratch_dz_backing.data_ptr(),
        scratch_dgu_backing.data_ptr(), expert_token_counts.data(), total_sms,
        &backward_host_context, stream,
        &context->cached_host_state());
    ::teramoe::prepare_teramoe_backward_communication_replay(
        backward_host_context, barrier_signal_ptrs_gpu,
        combine_barrier_signal_ptrs_gpu, stream);
    const int smem_size = std::max(NUM_MAX_NVL_PEERS * 16384, 24 * 9248);

    // Reset RDMA-reuse mailboxes before the backward launch (backward runs the same combine
    // prelude). Barrier so every rank observes the cleared mailbox before any peer's backward
    // prelude writes into it, and so forward's leftover "done" flags can't be misread.
    if (num_rdma_bytes > 0 && rdma_reuse_dispatch_quiet_done != nullptr) {
        int64_t mailbox_bytes = static_cast<int64_t>(num_rdma_ranks) * sizeof(int);
        CUDA_CHECK(cudaMemsetAsync(rdma_reuse_dispatch_quiet_done, 0, mailbox_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(rdma_reuse_combine_clear_done, 0, mailbox_bytes, stream));
        // AT_CUDA_CHECK(cudaStreamSynchronize(stream));
        // internode::barrier();
    }

    ::teramoe::launch_teramoe_fused_backward(
        backward_state, backward_host_context, total_sms, smem_size, stage,
        ::teramoe::ComputeDType::kBF16, stream);

    // The backward kernel writes wgrad scratch operands directly into the caller-owned
    // torch tensors (scratch_x/act/dz/dgu). No host-device synchronization is needed here
    // because the caller (QuACK wgrad) launches on the same CUDA stream, so stream ordering
    // guarantees the backward kernel completes before the wgrad GEMMs read the scratch data.
    // Free the backward state using the host-cached copy (no D2H required).
    ::teramoe::free_teramoe_fused_backward_state(backward_state, backward_host_context);
    ::teramoe::free_teramoe_backward_host_context(backward_host_context);
    return {grad_input, grad_w_gateup, grad_w_down, grad_topk_weights_out,
            scratch_x, scratch_act, scratch_dz, scratch_dgu, cu_seqlens_k};
#else
    EP_HOST_ASSERT(false && "megakernel backward requires NVSHMEM support");
    return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
#endif
}


}  // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TeraMOE: a cross-node Mixture-of-Experts (MoE) training engine";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::TeraMoEAutogradContext,
                     std::shared_ptr<deep_ep::TeraMoEAutogradContext>>(
        m, "TeraMoEAutogradContext");

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer)
        .def("teramoe_forward", &deep_ep::Buffer::teramoe_forward,
             py::arg("x"),
             py::arg("topk_idx"),
             py::arg("topk_weights"),
             py::arg("W_gateup"),
             py::arg("W_down"),
             py::arg("num_experts"),
             py::arg("num_dispatch_sms") = 24,
             py::arg("num_combine_sms") = 24,
             py::arg("total_sms") = 148,
             py::arg("stage") = 1,
             py::arg("dispatch_config") = deep_ep::Config(20, 6, 256, 6, 128),
             py::arg("combine_config") = deep_ep::Config(20, 4, 256, 6, 128),
             py::arg("hidden_states_scales") = py::none(),
             py::arg("W_gateup_fp8") = py::none(),
             py::arg("W_down_fp8") = py::none(),
             py::arg("W_gateup_fp8_sf") = py::none(),
             py::arg("W_down_fp8_sf") = py::none(),
             py::arg("compute_batch_size") = 4096,
             py::arg("combine_start_head_percent") = 70)
        .def("teramoe_forward_train", &deep_ep::Buffer::teramoe_forward_train,
             py::arg("x"), py::arg("topk_idx"), py::arg("topk_weights"),
             py::arg("W_gateup"), py::arg("W_down"), py::arg("num_experts"),
             py::arg("num_dispatch_sms") = 24, py::arg("num_combine_sms") = 24,
             py::arg("total_sms") = 148, py::arg("stage") = 1,
             py::arg("dispatch_config") = deep_ep::Config(20, 6, 256, 6, 128),
             py::arg("combine_config") = deep_ep::Config(20, 4, 256, 6, 128),
             py::arg("compute_batch_size") = 4096,
             py::arg("combine_start_head_percent") = 70)
        .def("teramoe_backward", &deep_ep::Buffer::teramoe_backward,
             py::arg("context"), py::arg("grad_output"), py::arg("grad_topk_weights") = py::none(),
             py::arg("total_sms") = 148, py::arg("stage") = 1)
        ;

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));
}
