#include "low_latency_gemm_swiglu.h"
#include "tensorrt_llm/common/assert.h"

#if !defined(TRTLLM_HAS_PREBUILT_INTERNAL_CUTLASS_KERNELS)

namespace tensorrt_llm::kernels::internal_cutlass_kernels
{

namespace
{

[[noreturn]] void throwUnsupportedLowLatencyGemmSwiglu()
{
    TLLM_THROW("LowLatencyGemmSwiglu requires the prebuilt internal CUTLASS kernels archive.");
}

} // namespace

template <typename T>
CutlassLowLatencyFp8GemmSwigluRunner<T>::CutlassLowLatencyFp8GemmSwigluRunner()
{
    mSm = 0;
}

template <typename T>
void CutlassLowLatencyFp8GemmSwigluRunner<T>::gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta,
    float scale_d0, float scale_d1, void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio,
    float prefetech_ratio, ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{
    (void) A;
    (void) B;
    (void) alpha;
    (void) beta;
    (void) scale_d0;
    (void) scale_d1;
    (void) C;
    (void) D;
    (void) m;
    (void) n;
    (void) k;
    (void) pdl_overlap_ratio;
    (void) prefetech_ratio;
    (void) gemmConfig;
    (void) workspacePtr;
    (void) workspaceBytes;
    (void) stream;
    throwUnsupportedLowLatencyGemmSwiglu();
}

template <typename T>
size_t CutlassLowLatencyFp8GemmSwigluRunner<T>::dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
    float alpha, float beta, float scale_d0, float scale_d1, void const* C, void* D, int m, int n, int k,
    float pdl_overlap_ratio, float prefetech_ratio, ConfigType gemmConfig, char* workspacePtr,
    size_t const workspaceBytes, cudaStream_t stream)
{
    (void) A;
    (void) B;
    (void) alpha;
    (void) beta;
    (void) scale_d0;
    (void) scale_d1;
    (void) C;
    (void) D;
    (void) m;
    (void) n;
    (void) k;
    (void) pdl_overlap_ratio;
    (void) prefetech_ratio;
    (void) gemmConfig;
    (void) workspacePtr;
    (void) workspaceBytes;
    (void) stream;
    throwUnsupportedLowLatencyGemmSwiglu();
}

template <typename T>
size_t CutlassLowLatencyFp8GemmSwigluRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{
    (void) m;
    (void) n;
    (void) k;
    throwUnsupportedLowLatencyGemmSwiglu();
}

template <typename T>
size_t CutlassLowLatencyFp8GemmSwigluRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k)
{
    (void) m;
    (void) n;
    (void) k;
    throwUnsupportedLowLatencyGemmSwiglu();
}

template <typename T>
std::vector<typename CutlassLowLatencyFp8GemmSwigluRunner<T>::ConfigType>
CutlassLowLatencyFp8GemmSwigluRunner<T>::getConfigs() const
{
    return {};
}

template class CutlassLowLatencyFp8GemmSwigluRunner<__nv_fp8_e4m3>;

} // namespace tensorrt_llm::kernels::internal_cutlass_kernels

#endif
