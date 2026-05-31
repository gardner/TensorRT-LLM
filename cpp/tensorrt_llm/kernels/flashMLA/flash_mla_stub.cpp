#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/flashMLA/flash_mla.h"

#include <cutlass/numeric_types.h>

namespace
{

[[noreturn]] void throwUnsupportedFlashMla()
{
    TLLM_THROW("TensorRT-LLM in-tree FlashMLA kernels are only built for SM90.");
}

} // namespace

void get_mla_metadata_func(Mla_metadata_params& params, cudaStream_t stream)
{
    (void) params;
    (void) stream;
    throwUnsupportedFlashMla();
}

template <typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params& params, cudaStream_t stream)
{
    (void) params;
    (void) stream;
    throwUnsupportedFlashMla();
}

template void run_mha_fwd_splitkv_mla<cutlass::half_t, cutlass::half_t, 576>(
    Flash_fwd_mla_params& params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, cutlass::bfloat16_t, 576>(
    Flash_fwd_mla_params& params, cudaStream_t stream);
template void run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(
    Flash_fwd_mla_params& params, cudaStream_t stream);
