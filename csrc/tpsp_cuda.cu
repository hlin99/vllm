// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"
#include "cub_helpers.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using bf16 = __nv_bfloat16;

// Each rank exposes an inbox slot to every peer.
// P2P flags are cleared by the receiver after each phase.
constexpr int kTpspInboxReady = 0;
constexpr int kTpspGatherReady = 1;
constexpr int kTpspFlagPhases = kTpspGatherReady + 1;
constexpr int kTpspDirectPipelineSlots = 3;
constexpr double kTpspTailMergeThreshold = 0.5;

struct alignas(16) TpspBf16Vec {
  bf16 data[8];
};

struct TpspChunk {
  int64_t offset;
  int64_t tokens;
};

struct TpspChunkPlan {
  std::vector<TpspChunk> chunks;
  int64_t max_tokens;
};

TpspChunkPlan make_tpsp_chunk_plan(int64_t tokens_per_rank, int64_t target,
                                   int64_t capacity) {
  target = std::min(tokens_per_rank, target);
  const int64_t full_chunks = tokens_per_rank / target;
  const int64_t tail = tokens_per_rank % target;
  // Redistribute small tails only when the larger chunks fit the workspace.
  const bool merge_tail =
      full_chunks > 0 && tail > 0 &&
      static_cast<double>(tail) / target <= kTpspTailMergeThreshold &&
      (tokens_per_rank + full_chunks - 1) / full_chunks <= capacity;
  const int64_t count = full_chunks + (tail > 0 && !merge_tail);
  const int64_t base = merge_tail ? tokens_per_rank / count : target;
  const int64_t larger = merge_tail ? tokens_per_rank % count : 0;
  TpspChunkPlan plan;
  plan.chunks.reserve(count);
  plan.max_tokens = 0;
  for (int64_t i = 0, offset = 0; i < count; ++i) {
    const int64_t size = merge_tail
                             ? base + (i >= count - larger)
                             : std::min(target, tokens_per_rank - offset);
    plan.chunks.push_back({offset, size});
    plan.max_tokens = std::max(plan.max_tokens, size);
    offset += size;
  }
  return plan;
}

// Chunk layout conversion kernels
__global__ void pack_tpsp_chunk(const bf16* input, bf16* packed, int tokens,
                                int width, int tokens_per_rank,
                                int microchunk_tokens_per_rank, int offset,
                                int tp_size) {
  // Group the same token range from each rank, padding the final shard.
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t count = int64_t(tp_size) * microchunk_tokens_per_rank * width;
  if (index >= count) {
    return;
  }
  int row = index / width;
  int rank = row / microchunk_tokens_per_rank;
  int source_row =
      rank * tokens_per_rank + offset + row % microchunk_tokens_per_rank;
  packed[index] = source_row < tokens
                      ? input[int64_t(source_row) * width + index % width]
                      : bf16(0.0f);
}

// Fused reduction, residual, and normalization kernels
template <bool LayerNorm>
__global__ void tpsp_add_norm(const bf16* reduced, const bf16* remote,
                              int64_t remote_stride, int rank, int tp_size,
                              const bf16* residual, const bf16* projection_bias,
                              const bf16* weight, const bf16* norm_bias,
                              bf16* new_residual, bf16* normalized,
                              int hidden_size, float eps) {
  int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    int index = row * hidden_size + col;
    float projected_sum = float(reduced[index]);
    for (int source = 0; source < tp_size; ++source) {
      if (source != rank) {
        projected_sum += float(remote[int64_t(source) * remote_stride + index]);
      }
    }
    bf16 projected = bf16(projected_sum);
    if (projection_bias) {
      projected = bf16(float(projected) + float(projection_bias[col]));
    }
    bf16 value = bf16(float(projected) + float(residual[index]));
    new_residual[index] = value;
    float x = float(value);
    sum += LayerNorm ? x : x * x;
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  sum = BlockReduce(reduce_store).Reduce(sum, CubAddOp{}, blockDim.x);
  __shared__ float mean;
  if constexpr (LayerNorm) {
    if (threadIdx.x == 0) {
      mean = sum / hidden_size;
    }
    __syncthreads();
    sum = 0.0f;
    for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
      float centered = float(new_residual[row * hidden_size + col]) - mean;
      sum += centered * centered;
    }
    sum = BlockReduce(reduce_store).Reduce(sum, CubAddOp{}, blockDim.x);
  }
  __shared__ float inverse_std;
  if (threadIdx.x == 0) {
    inverse_std = rsqrtf(sum / hidden_size + eps);
  }
  __syncthreads();
  for (int col = threadIdx.x; col < hidden_size; col += blockDim.x) {
    int index = row * hidden_size + col;
    float value = float(new_residual[index]);
    if constexpr (LayerNorm) {
      value -= mean;
    }
    value *= inverse_std;
    value *= float(weight[col]);
    if constexpr (LayerNorm) {
      if (norm_bias) {
        value += float(norm_bias[col]);
      }
    }
    normalized[index] = bf16(value);
  }
}

template <bool LayerNorm>
__global__ void tpsp_add_norm_vector(const bf16* reduced, const bf16* remote,
                                     int64_t remote_stride, int rank,
                                     int tp_size, const bf16* residual,
                                     const bf16* projection_bias,
                                     const bf16* weight, const bf16* norm_bias,
                                     bf16* new_residual, bf16* normalized,
                                     int hidden_size, float eps) {
  using Vec = TpspBf16Vec;
  const int row = blockIdx.x;
  const int vec_hidden = hidden_size / 8;
  const auto* reduced_v =
      reinterpret_cast<const Vec*>(reduced + int64_t(row) * hidden_size);
  const auto* residual_v =
      reinterpret_cast<const Vec*>(residual + int64_t(row) * hidden_size);
  const auto* projection_bias_v = reinterpret_cast<const Vec*>(projection_bias);
  const auto* weight_v = reinterpret_cast<const Vec*>(weight);
  const auto* norm_bias_v = reinterpret_cast<const Vec*>(norm_bias);
  auto* new_residual_v =
      reinterpret_cast<Vec*>(new_residual + int64_t(row) * hidden_size);
  auto* normalized_v =
      reinterpret_cast<Vec*>(normalized + int64_t(row) * hidden_size);

  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
    Vec value = reduced_v[idx];
    Vec other = residual_v[idx];
    Vec bias;
    if (projection_bias) {
      bias = projection_bias_v[idx];
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      __nv_bfloat162 pair{value.data[j], value.data[j + 1]};
      float2 sum = __bfloat1622float2(pair);
      for (int source = 0; source < tp_size; ++source) {
        if (source != rank) {
          auto* incoming = reinterpret_cast<const Vec*>(
              remote + int64_t(source) * remote_stride +
              int64_t(row) * hidden_size);
          float2 x = __bfloat1622float2(
              __nv_bfloat162{incoming[idx].data[j], incoming[idx].data[j + 1]});
          sum.x += x.x;
          sum.y += x.y;
        }
      }
      pair = __floats2bfloat162_rn(sum.x, sum.y);
      if (projection_bias) {
        pair += __nv_bfloat162{bias.data[j], bias.data[j + 1]};
      }
      pair += __nv_bfloat162{other.data[j], other.data[j + 1]};
      value.data[j] = pair.x;
      value.data[j + 1] = pair.y;
    }
    new_residual_v[idx] = value;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
      float2 pair =
          __bfloat1622float2(__nv_bfloat162{value.data[j], value.data[j + 1]});
      if constexpr (LayerNorm) {
        variance += pair.x + pair.y;
      } else {
        variance += pair.x * pair.x + pair.y * pair.y;
      }
    }
  }

  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  variance = BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  __shared__ float mean;
  if constexpr (LayerNorm) {
    if (threadIdx.x == 0) {
      mean = variance / hidden_size;
    }
    __syncthreads();
    variance = 0.0f;
    for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
      Vec value = new_residual_v[idx];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float centered = __bfloat162float(value.data[j]) - mean;
        variance += centered * centered;
      }
    }
    variance =
        BlockReduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  }
  __shared__ float inverse_std;
  if (threadIdx.x == 0) {
    inverse_std = rsqrtf(variance / hidden_size + eps);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
    Vec value = new_residual_v[idx];
    Vec weights = weight_v[idx];
    Vec bias;
    if constexpr (LayerNorm) {
      if (norm_bias) {
        bias = norm_bias_v[idx];
      }
    }
    Vec output;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float x = __bfloat162float(value.data[j]);
      if constexpr (LayerNorm) {
        x -= mean;
      }
      float w = __bfloat162float(weights.data[j]);
      float y = x * inverse_std * w;
      if constexpr (LayerNorm) {
        if (norm_bias) {
          y += __bfloat162float(bias.data[j]);
        }
      }
      output.data[j] = __float2bfloat16(y);
    }
    normalized_v[idx] = output;
  }
}

// P2P signaling kernels
__global__ void wait_and_clear_tpsp_signals(unsigned int* signals, int rank,
                                            int tp_size) {
  int peer = blockIdx.x * blockDim.x + threadIdx.x;
  if (peer >= tp_size || peer == rank) {
    return;
  }
  auto start = clock64();
  while (atomicCAS(signals + peer, 1, 0) != 1) {
    if (clock64() - start > 30000000000ULL) {
      printf("TPSP P2P signal timed out\n");
      asm volatile("trap;");
    }
  }
}

__global__ void tpsp_system_fence() { __threadfence_system(); }

// Pipeline and transport state
struct CudaEvent {
  cudaEvent_t handle{};

  CudaEvent() {
    STD_CUDA_CHECK(cudaEventCreateWithFlags(&handle, cudaEventDisableTiming));
  }
  ~CudaEvent() { cudaEventDestroy(handle); }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
};

struct PipelineState {
  // Reuse a scratch slot only after its communication has finished.
  std::array<CudaEvent, kTpspDirectPipelineSlots> gemm_ready;
  std::array<CudaEvent, kTpspDirectPipelineSlots> comm_done;
  cudaStream_t comm_stream{};

  PipelineState() {
    int least_priority;
    int greatest_priority;
    STD_CUDA_CHECK(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    STD_CUDA_CHECK(cudaStreamCreateWithPriority(
        &comm_stream, cudaStreamNonBlocking, greatest_priority));
  }
  ~PipelineState() { cudaStreamDestroy(comm_stream); }
};

thread_local std::unordered_map<int, std::unique_ptr<PipelineState>>
    pipeline_states;

// Per-call scratch buffers for projection and optional row packing.
struct ChunkBuffers {
  std::optional<torch::stable::Tensor> packed;
  torch::stable::Tensor partial;
};

struct TpspIpcHandles {
  cudaIpcMemHandle_t workspace;
  cudaIpcMemHandle_t output;
};

// Local IPC mappings and workspaces for one rank's direct P2P context.
struct TpspP2pContext {
  int device_index;
  int rank;
  int tp_size;
  int64_t max_microchunk_tokens_per_rank;
  int hidden_size;
  int64_t slot_bytes;
  int64_t flag_offset;
  int64_t pipeline_slot_bytes;
  int64_t flag_slot_bytes;
  int64_t max_tokens = 0;
  TpspIpcHandles ipc_handles{};
  bool mapped = false;
  void* workspace = nullptr;
  void* output_workspace = nullptr;
  void* signal_one = nullptr;
  std::vector<void*> peers;
  std::vector<void*> output_peers;

  bf16* peer_inbox(int peer, int slot, int source) const {
    return reinterpret_cast<bf16*>(static_cast<char*>(peers[peer]) +
                                   slot * pipeline_slot_bytes +
                                   source * slot_bytes);
  }

  unsigned int* flags(int peer, int slot, int phase) const {
    return reinterpret_cast<unsigned int*>(static_cast<char*>(peers[peer]) +
                                           flag_offset +
                                           slot * flag_slot_bytes) +
           phase * tp_size;
  }

  bf16* output(int peer, int64_t token_offset) const {
    return static_cast<bf16*>(output_peers[peer]) + token_offset * hidden_size;
  }

  template <bool CheckErrors = true>
  void close_peer_mappings() {
    for (auto* mappings : {&peers, &output_peers}) {
      for (int source = 0; source < tp_size; ++source) {
        if (source != rank && (*mappings)[source]) {
          auto status = cudaIpcCloseMemHandle((*mappings)[source]);
          if constexpr (CheckErrors) {
            STD_CUDA_CHECK(status);
          }
          (*mappings)[source] = nullptr;
        }
      }
    }
  }

  template <bool CheckErrors = true>
  void free_local() {
    for (auto* allocation : {&signal_one, &workspace, &output_workspace}) {
      if (*allocation) {
        auto status = cudaFree(*allocation);
        if constexpr (CheckErrors) {
          STD_CUDA_CHECK(status);
        }
        *allocation = nullptr;
      }
    }
  }

  ~TpspP2pContext() {
    close_peer_mappings<false>();
    free_local<false>();
  }
};

std::mutex tpsp_context_mutex;
std::unordered_map<int64_t, std::shared_ptr<TpspP2pContext>> tpsp_contexts;
int64_t next_tpsp_handle = 1;

std::shared_ptr<TpspP2pContext> get_tpsp_context(int64_t handle) {
  std::lock_guard<std::mutex> lock(tpsp_context_mutex);
  auto it = tpsp_contexts.find(handle);
  STD_TORCH_CHECK(it != tpsp_contexts.end(), "Invalid TPSP P2P context");
  return it->second;
}

void notify_tpsp_peers_and_wait(TpspP2pContext& context, int slot, int phase,
                                cudaStream_t stream) {
  auto* one = reinterpret_cast<const unsigned int*>(context.signal_one);
  tpsp_system_fence<<<1, 1, 0, stream>>>();
  STD_CUDA_CHECK(cudaGetLastError());
  for (int dest = 0; dest < context.tp_size; ++dest) {
    if (dest == context.rank) {
      continue;
    }
    auto* peer_flag = context.flags(dest, slot, phase) + context.rank;
    STD_CUDA_CHECK(cudaMemcpyAsync(peer_flag, one, sizeof(unsigned int),
                                   cudaMemcpyDeviceToDevice, stream));
  }
  wait_and_clear_tpsp_signals<<<(context.tp_size + 255) / 256, 256, 0,
                                stream>>>(
      context.flags(context.rank, slot, phase), context.rank, context.tp_size);
  STD_CUDA_CHECK(cudaGetLastError());
}

torch::stable::Tensor make_bf16(const torch::stable::Tensor& a, int64_t rows,
                                int64_t cols) {
  return torch::stable::empty({rows, cols}, a.scalar_type(), std::nullopt,
                              a.device());
}

torch::stable::Tensor make_p2p_output(const torch::stable::Tensor& a,
                                      void* data, int64_t rows, int64_t cols) {
  return torch::stable::from_blob(data, {rows, cols}, {cols, 1}, a.device(),
                                  a.scalar_type());
}

template <typename Gemm>
cublasStatus_t tpsp_gemm(cublasHandle_t blas, int sm_count_target, Gemm gemm) {
  if (sm_count_target == 0) {
    return gemm();
  }
  int original;
  STD_TORCH_CHECK(
      cublasGetSmCountTarget(blas, &original) == CUBLAS_STATUS_SUCCESS,
      "TPSP failed to read cuBLAS SM target");
  STD_TORCH_CHECK(
      cublasSetSmCountTarget(blas, sm_count_target) == CUBLAS_STATUS_SUCCESS,
      "TPSP failed to set cuBLAS SM target");
  // The callback only calls the non-throwing cuBLAS C API. Restore on failure
  // too.
  cublasStatus_t result = gemm();
  STD_TORCH_CHECK(
      cublasSetSmCountTarget(blas, original) == CUBLAS_STATUS_SUCCESS,
      "TPSP failed to restore cuBLAS SM target");
  return result;
}

void project_tpsp_chunk(cublasHandle_t blas, cudaStream_t stream,
                        int sm_count_target, const bf16* a, const bf16* b,
                        bf16* packed, bf16* partial, int64_t tokens,
                        int64_t width, int64_t hidden, int64_t tokens_per_rank,
                        int tp_size, const TpspChunk& chunk) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  if (tokens % tp_size != 0) {
    const int64_t pack_elems = chunk.tokens * tp_size * width;
    pack_tpsp_chunk<<<(pack_elems + 255) / 256, 256, 0, stream>>>(
        a, packed, tokens, width, tokens_per_rank, chunk.tokens, chunk.offset,
        tp_size);
    STD_CUDA_CHECK(cudaGetLastError());
  }
  const auto status = tpsp_gemm(blas, sm_count_target, [&] {
    if (tokens % tp_size == 0) {
      return cublasGemmStridedBatchedEx(
          blas, CUBLAS_OP_N, CUBLAS_OP_N, hidden, chunk.tokens, width, &alpha,
          b, CUDA_R_16BF, hidden, 0, a + chunk.offset * width, CUDA_R_16BF,
          width, tokens_per_rank * width, &beta, partial, CUDA_R_16BF, hidden,
          chunk.tokens * hidden, tp_size, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, hidden,
                        chunk.tokens * tp_size, width, &alpha, b, CUDA_R_16BF,
                        hidden, packed, CUDA_R_16BF, width, &beta, partial,
                        CUDA_R_16BF, hidden, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  });
  STD_TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "TPSP BF16 GEMM failed");
}

void exchange_tpsp_shards(TpspP2pContext& context, int slot,
                          const bf16* partial, int64_t shard_elems,
                          cudaStream_t stream) {
  for (int dest = 0; dest < context.tp_size; ++dest) {
    if (dest != context.rank) {
      STD_CUDA_CHECK(cudaMemcpyAsync(
          context.peer_inbox(dest, slot, context.rank),
          partial + dest * shard_elems, shard_elems * sizeof(bf16),
          cudaMemcpyDeviceToDevice, stream));
    }
  }
  notify_tpsp_peers_and_wait(context, slot, kTpspInboxReady, stream);
}

void gather_tpsp_shards(TpspP2pContext& context, int slot, int64_t token_offset,
                        int own_tokens, cudaStream_t stream) {
  const size_t bytes = int64_t(own_tokens) * context.hidden_size * sizeof(bf16);
  for (int dest = 0; dest < context.tp_size; ++dest) {
    if (dest != context.rank && bytes > 0) {
      STD_CUDA_CHECK(cudaMemcpyAsync(context.output(dest, token_offset),
                                     context.output(context.rank, token_offset),
                                     bytes, cudaMemcpyDeviceToDevice, stream));
    }
  }
  notify_tpsp_peers_and_wait(context, slot, kTpspGatherReady, stream);
}

void launch_tpsp_add_norm(int norm_kind, int rows, cudaStream_t stream,
                          const bf16* reduced, const bf16* remote,
                          int64_t remote_stride, int rank, int tp_size,
                          const bf16* residual, const bf16* projection_bias,
                          const bf16* weight, const bf16* norm_bias,
                          bf16* new_residual, bf16* normalized, int hidden,
                          float eps) {
  if (rows == 0) {
    return;
  }
  const bool aligned =
      hidden % 8 == 0 && ((reinterpret_cast<uintptr_t>(reduced) |
                           reinterpret_cast<uintptr_t>(remote) |
                           reinterpret_cast<uintptr_t>(residual) |
                           reinterpret_cast<uintptr_t>(projection_bias) |
                           reinterpret_cast<uintptr_t>(weight) |
                           reinterpret_cast<uintptr_t>(norm_bias) |
                           reinterpret_cast<uintptr_t>(new_residual) |
                           reinterpret_cast<uintptr_t>(normalized)) &
                          15) == 0;
  auto kernel =
      norm_kind == 0
          ? (aligned ? tpsp_add_norm_vector<false> : tpsp_add_norm<false>)
          : (aligned ? tpsp_add_norm_vector<true> : tpsp_add_norm<true>);
  kernel<<<rows, 256, 0, stream>>>(
      reduced, remote, remote_stride, rank, tp_size, residual, projection_bias,
      weight, norm_bias, new_residual, normalized, hidden, eps);
  STD_CUDA_CHECK(cudaGetLastError());
}

struct TpspShape {
  int64_t tokens;
  int64_t width;
  int64_t hidden;
  int64_t tokens_per_rank;
};

TpspShape validate_tpsp_inputs(
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& weight, const torch::stable::Tensor& residual,
    const std::optional<torch::stable::Tensor>& projection_bias,
    const std::optional<torch::stable::Tensor>& norm_bias, double eps,
    int64_t norm_kind, int64_t microchunk_tokens_per_rank, int64_t tp_size,
    int64_t p2p_handle) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(
      a.is_cuda() && b.is_cuda() && weight.is_cuda() && residual.is_cuda(),
      "TPSP requires CUDA tensors");
  STD_TORCH_CHECK(a.scalar_type() == ScalarType::BFloat16 &&
                      b.scalar_type() == ScalarType::BFloat16 &&
                      weight.scalar_type() == ScalarType::BFloat16 &&
                      residual.scalar_type() == ScalarType::BFloat16,
                  "TPSP requires BF16 tensors");
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && weight.dim() == 1 &&
                      residual.dim() == 2 && a.is_contiguous() &&
                      b.is_contiguous() && weight.is_contiguous() &&
                      residual.is_contiguous(),
                  "TPSP requires contiguous matrices and vectors");
  STD_TORCH_CHECK(a.device() == b.device() && a.device() == weight.device() &&
                      a.device() == residual.device(),
                  "TPSP tensors must reside on the same CUDA device");
  STD_TORCH_CHECK(norm_kind == 0 || norm_kind == 1,
                  "TPSP norm must be RMSNorm or LayerNorm");
  STD_TORCH_CHECK(!norm_bias || norm_kind == 1,
                  "TPSP norm bias requires LayerNorm");
  for (const auto* bias : {&projection_bias, &norm_bias}) {
    if (*bias) {
      STD_TORCH_CHECK(
          bias->value().is_cuda() &&
              bias->value().scalar_type() == ScalarType::BFloat16 &&
              bias->value().device() == a.device() &&
              bias->value().is_contiguous() && bias->value().dim() == 1 &&
              bias->value().size(0) == b.size(1),
          "TPSP bias must be a contiguous CUDA BF16 hidden-size vector");
    }
  }
  STD_TORCH_CHECK(a.size(1) == b.size(0) && b.size(1) == weight.size(0) &&
                      tp_size >= 2 && microchunk_tokens_per_rank > 0 &&
                      p2p_handle != 0 && eps > 0,
                  "TPSP requires a ready P2P context and valid configuration");
  const int64_t tokens = a.size(0);
  const int64_t width = a.size(1);
  const int64_t hidden = b.size(1);
  const int64_t tokens_per_rank = (tokens + tp_size - 1) / tp_size;
  STD_TORCH_CHECK(tokens > 0 && residual.size(0) == tokens_per_rank &&
                      residual.size(1) == hidden,
                  "TPSP residual shard has an unexpected shape");
  STD_TORCH_CHECK(tokens <= INT32_MAX && width <= INT32_MAX &&
                      hidden <= INT32_MAX && tokens_per_rank <= INT32_MAX &&
                      microchunk_tokens_per_rank <= INT32_MAX &&
                      tp_size <= INT32_MAX,
                  "TPSP dimensions exceed CUDA kernel limits");
  return {tokens, width, hidden, tokens_per_rank};
}

}  // namespace

// Returns 0 when this rank cannot allocate or export its P2P workspaces.
int64_t init_tpsp_p2p(int64_t device_index, int64_t tp_size, int64_t rank,
                      int64_t max_microchunk_tokens_per_rank,
                      int64_t hidden_size, int64_t max_tokens) {
  STD_TORCH_CHECK(tp_size >= 2 && tp_size <= INT32_MAX && rank >= 0 &&
                      rank < tp_size && max_microchunk_tokens_per_rank > 0 &&
                      max_microchunk_tokens_per_rank <= INT32_MAX &&
                      hidden_size > 0 && hidden_size <= INT32_MAX &&
                      max_tokens > 0 &&
                      max_microchunk_tokens_per_rank <=
                          (std::numeric_limits<int64_t>::max() -
                           int64_t(kTpspDirectPipelineSlots) * kTpspFlagPhases *
                               tp_size * sizeof(unsigned int)) /
                              kTpspDirectPipelineSlots / tp_size / hidden_size /
                              int64_t(sizeof(bf16)) &&
                      max_tokens <= std::numeric_limits<int64_t>::max() /
                                        hidden_size / int64_t(sizeof(bf16)),
                  "Invalid TPSP P2P configuration");
  const torch::stable::accelerator::DeviceGuard guard(device_index);
  auto context = std::make_shared<TpspP2pContext>();
  context->device_index = device_index;
  context->rank = rank;
  context->tp_size = tp_size;
  context->max_microchunk_tokens_per_rank = max_microchunk_tokens_per_rank;
  context->hidden_size = hidden_size;
  context->slot_bytes =
      max_microchunk_tokens_per_rank * hidden_size * sizeof(bf16);
  context->pipeline_slot_bytes = tp_size * context->slot_bytes;
  context->flag_slot_bytes = kTpspFlagPhases * tp_size * sizeof(unsigned int);
  context->flag_offset =
      kTpspDirectPipelineSlots * context->pipeline_slot_bytes;
  context->max_tokens = max_tokens;
  context->peers.resize(tp_size, nullptr);
  context->output_peers.resize(tp_size, nullptr);
  size_t bytes = context->flag_offset +
                 kTpspDirectPipelineSlots * context->flag_slot_bytes;
  size_t output_bytes = max_tokens * hidden_size * sizeof(bf16);

  cudaError_t allocated = cudaMalloc(&context->workspace, bytes);
  if (allocated == cudaSuccess) {
    allocated = cudaMalloc(&context->output_workspace, output_bytes);
  }
  if (allocated == cudaSuccess) {
    allocated = cudaMalloc(&context->signal_one, sizeof(unsigned int));
  }
  bool exported = allocated == cudaSuccess &&
                  cudaIpcGetMemHandle(&context->ipc_handles.workspace,
                                      context->workspace) == cudaSuccess &&
                  cudaIpcGetMemHandle(&context->ipc_handles.output,
                                      context->output_workspace) == cudaSuccess;
  if (!exported) {
    context->free_local();
    return 0;
  }

  std::lock_guard<std::mutex> lock(tpsp_context_mutex);
  int64_t handle = next_tpsp_handle++;
  tpsp_contexts.emplace(handle, std::move(context));
  return handle;
}

torch::stable::Tensor get_tpsp_p2p_handles(int64_t handle) {
  auto context = get_tpsp_context(handle);
  auto result = torch::stable::empty(
      {static_cast<int64_t>(sizeof(TpspIpcHandles))},
      torch::headeronly::ScalarType::Byte, std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));
  std::memcpy(result.mutable_data_ptr(), &context->ipc_handles,
              sizeof(TpspIpcHandles));
  return result;
}

bool map_tpsp_p2p(int64_t handle, const torch::stable::Tensor& peer_handles) {
  auto context = get_tpsp_context(handle);
  STD_TORCH_CHECK(
      peer_handles.device().is_cpu() &&
          peer_handles.scalar_type() == torch::headeronly::ScalarType::Byte &&
          peer_handles.is_contiguous() &&
          peer_handles.numel() == context->tp_size * sizeof(TpspIpcHandles) &&
          !context->mapped,
      "Invalid TPSP P2P peer handles");
  std::vector<TpspIpcHandles> handles(context->tp_size);
  std::memcpy(handles.data(), peer_handles.const_data_ptr(),
              context->tp_size * sizeof(TpspIpcHandles));
  const torch::stable::accelerator::DeviceGuard guard(context->device_index);
  context->peers[context->rank] = context->workspace;
  context->output_peers[context->rank] = context->output_workspace;
  bool mapped = true;
  for (int source = 0; source < context->tp_size; ++source) {
    if (source == context->rank) {
      continue;
    }
    if (cudaIpcOpenMemHandle(&context->peers[source], handles[source].workspace,
                             cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
      mapped = false;
      break;
    }
    if (cudaIpcOpenMemHandle(&context->output_peers[source],
                             handles[source].output,
                             cudaIpcMemLazyEnablePeerAccess) != cudaSuccess) {
      mapped = false;
      break;
    }
  }
  if (!mapped) {
    context->close_peer_mappings();
    return false;
  }
  auto stream = get_current_cuda_stream();
  const size_t bytes = context->flag_offset +
                       kTpspDirectPipelineSlots * context->flag_slot_bytes;
  STD_CUDA_CHECK(cudaMemsetAsync(context->workspace, 0, bytes, stream));
  unsigned int one = 1;
  STD_CUDA_CHECK(cudaMemcpyAsync(context->signal_one, &one, sizeof(one),
                                 cudaMemcpyHostToDevice, stream));
  // The CPU rank handshake must not finish before peer-visible flags are ready.
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  context->mapped = true;
  return true;
}

// Python synchronizes the control group before and after destruction.
void destroy_tpsp_p2p(int64_t handle) {
  auto context = get_tpsp_context(handle);
  const torch::stable::accelerator::DeviceGuard guard(context->device_index);
  STD_CUDA_CHECK(cudaDeviceSynchronize());
  context->close_peer_mappings();
  context->free_local();
  std::lock_guard<std::mutex> lock(tpsp_context_mutex);
  tpsp_contexts.erase(handle);
}

// Fused TPSP execution
//
// Processes each chunk through projection, reduction, residual addition,
// normalization, and gather. Returns the updated residual shard and gathered
// normalized output. The gathered output is borrowed until the next TPSP call.
std::tuple<torch::stable::Tensor, torch::stable::Tensor>
tpsp_fused_matmul_reduce_scatter_norm_all_gather(
    const torch::stable::Tensor& a, const torch::stable::Tensor& b,
    const torch::stable::Tensor& weight, const torch::stable::Tensor& residual,
    const std::optional<torch::stable::Tensor>& projection_bias,
    const std::optional<torch::stable::Tensor>& norm_bias, double eps,
    int64_t norm_kind, int64_t microchunk_tokens_per_rank, int64_t tp_size,
    int64_t p2p_handle, int64_t sm_count_target) {
  const auto [tokens, width, hidden, tokens_per_rank] = validate_tpsp_inputs(
      a, b, weight, residual, projection_bias, norm_bias, eps, norm_kind,
      microchunk_tokens_per_rank, tp_size, p2p_handle);
  const torch::stable::accelerator::DeviceGuard guard(a.get_device_index());
  int sm_count;
  STD_CUDA_CHECK(cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, a.get_device_index()));
  STD_TORCH_CHECK(sm_count_target >= 0 && sm_count_target <= sm_count,
                  "TPSP cuBLAS SM target exceeds device capacity");
  cudaStream_t stream = get_current_cuda_stream();
  cublasHandle_t blas = get_current_cuda_blas_handle();
  auto p2p_context = get_tpsp_context(p2p_handle);
  const auto plan =
      make_tpsp_chunk_plan(tokens_per_rank, microchunk_tokens_per_rank,
                           p2p_context->max_microchunk_tokens_per_rank);
  const int64_t num_chunks = plan.chunks.size();
  STD_TORCH_CHECK(
      p2p_context->device_index == a.get_device_index() &&
          p2p_context->tp_size == tp_size &&
          p2p_context->hidden_size == hidden &&
          plan.max_tokens <= p2p_context->max_microchunk_tokens_per_rank &&
          p2p_context->mapped,
      "TPSP P2P context does not match the request");
  int rank = p2p_context->rank;
  PipelineState* pipeline = nullptr;
  if (num_chunks > 1) {
    auto& state = pipeline_states[a.get_device_index()];
    if (!state) {
      state = std::make_unique<PipelineState>();
    }
    pipeline = state.get();
  }
  STD_TORCH_CHECK(
      tokens <= p2p_context->max_tokens && p2p_context->output_workspace,
      "TPSP direct P2P output exceeds workspace capacity");
  std::vector<ChunkBuffers> buffers;
  int buffer_slots = pipeline ? kTpspDirectPipelineSlots : 1;
  const bool needs_packing = tokens % tp_size != 0;
  for (int slot = 0; slot < buffer_slots; ++slot) {
    buffers.push_back({
        needs_packing
            ? std::make_optional(make_bf16(a, plan.max_tokens * tp_size, width))
            : std::nullopt,
        make_bf16(a, plan.max_tokens * tp_size, hidden),
    });
  }
  auto new_residual = make_bf16(a, tokens_per_rank, hidden);
  auto gathered =
      make_p2p_output(a, p2p_context->output_workspace, tokens, hidden);
  auto* a_ptr = reinterpret_cast<const bf16*>(a.const_data_ptr());
  auto* b_ptr = reinterpret_cast<const bf16*>(b.const_data_ptr());
  auto* residual_ptr = reinterpret_cast<const bf16*>(residual.const_data_ptr());
  auto* new_residual_ptr =
      reinterpret_cast<bf16*>(new_residual.mutable_data_ptr());
  auto* weight_ptr = reinterpret_cast<const bf16*>(weight.const_data_ptr());
  auto* projection_bias_ptr =
      projection_bias
          ? reinterpret_cast<const bf16*>(projection_bias->const_data_ptr())
          : nullptr;
  auto* norm_bias_ptr =
      norm_bias ? reinterpret_cast<const bf16*>(norm_bias->const_data_ptr())
                : nullptr;
  for (int64_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
    const auto& chunk = plan.chunks[chunk_index];
    int64_t offset = chunk.offset;
    int slot = static_cast<int>(chunk_index % buffers.size());
    if (pipeline && chunk_index >= int64_t(buffers.size())) {
      STD_CUDA_CHECK(
          cudaStreamWaitEvent(stream, pipeline->comm_done[slot].handle, 0));
    }
    auto& scratch = buffers[slot];
    auto* packed_ptr =
        scratch.packed
            ? reinterpret_cast<bf16*>(scratch.packed->mutable_data_ptr())
            : nullptr;
    auto* partial_ptr =
        reinterpret_cast<bf16*>(scratch.partial.mutable_data_ptr());
    // Stage 1: Project each rank's input tokens; pack only for uneven shards.
    project_tpsp_chunk(blas, stream, sm_count_target, a_ptr, b_ptr, packed_ptr,
                       partial_ptr, tokens, width, hidden, tokens_per_rank,
                       tp_size, chunk);
    cudaStream_t comm_stream = pipeline ? pipeline->comm_stream : stream;
    if (pipeline) {
      STD_CUDA_CHECK(
          cudaEventRecord(pipeline->gemm_ready[slot].handle, stream));
      STD_CUDA_CHECK(cudaStreamWaitEvent(comm_stream,
                                         pipeline->gemm_ready[slot].handle, 0));
    }
    int64_t shard_elems = chunk.tokens * hidden;
    // Stage 2: Exchange projected shards through peer workspaces.
    exchange_tpsp_shards(*p2p_context, slot, partial_ptr, shard_elems,
                         comm_stream);

    auto* residual_chunk = residual_ptr + offset * hidden;
    auto* new_residual_chunk = new_residual_ptr + offset * hidden;
    int own_tokens_per_rank = std::max<int64_t>(
        0, std::min(chunk.tokens, tokens - rank * tokens_per_rank - offset));
    const int64_t token_offset = int64_t(rank) * tokens_per_rank + offset;
    auto* normalized_chunk = p2p_context->output(rank, token_offset);
    auto* reduced_ptr = partial_ptr + rank * shard_elems;
    // Stage 3: Fuse the P2P sum, biases, residual add, and normalization.
    launch_tpsp_add_norm(norm_kind, own_tokens_per_rank, comm_stream,
                         reduced_ptr, p2p_context->peer_inbox(rank, slot, 0),
                         p2p_context->slot_bytes / sizeof(bf16), rank, tp_size,
                         residual_chunk, projection_bias_ptr, weight_ptr,
                         norm_bias_ptr, new_residual_chunk, normalized_chunk,
                         hidden, static_cast<float>(eps));
    // Stage 4: All-gather normalized shards.
    gather_tpsp_shards(*p2p_context, slot, token_offset, own_tokens_per_rank,
                       comm_stream);
    if (pipeline) {
      STD_CUDA_CHECK(
          cudaEventRecord(pipeline->comm_done[slot].handle, comm_stream));
    }
  }
  if (pipeline) {
    STD_CUDA_CHECK(cudaStreamWaitEvent(
        stream, pipeline->comm_done[(num_chunks - 1) % buffers.size()].handle,
        0));
  }
  return {new_residual, gathered};
}
