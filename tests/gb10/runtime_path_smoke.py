from __future__ import annotations

import argparse

import torch

import tensorrt_llm._torch  # noqa: F401
from tensorrt_llm._torch.attention_backend.trtllm_gen import TrtllmGenSupportChecker
from tensorrt_llm._torch.custom_ops.torch_custom_ops import IS_CUBLASLT_AVAILABLE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm.bindings import DataType
from tensorrt_llm.models.modeling_utils import QuantAlgo


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_sm121() -> None:
    require(torch.cuda.is_available(), "CUDA is required for the GB10 runtime smoke")
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print("runtime_path_device", device_name, capability)
    require(capability == (12, 1), f"expected SM121, got {capability}")


def check_trtllm_gen_rejections() -> None:
    supported, reason = TrtllmGenSupportChecker.is_supported(
        q_dtype=torch.float16,
        kv_cache_dtype=DataType.HALF,
        num_heads=8,
        num_kv_heads=8,
        head_size=128,
        out_dtype=torch.float16,
    )
    print("runtime_path_trtllm_gen_attention_supported", supported)
    print("runtime_path_trtllm_gen_attention_reason", reason)
    require(
        not supported and "SM121" in reason,
        "TrtLlmGen attention must reject SM121 with an explicit reason",
    )

    can_moe, moe_reason = TRTLLMGenFusedMoE.can_implement(QuantAlgo.NVFP4)
    print("runtime_path_trtllm_gen_moe_supported", can_moe)
    print("runtime_path_trtllm_gen_moe_reason", moe_reason)
    require(
        not can_moe and "SM121" in (moe_reason or ""),
        "TrtLlmGen MoE must reject SM121 with an explicit reason",
    )


def quantize_nvfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_scale = (448 * 6) / tensor.abs().max().float()
    fp4, block_scale = torch.ops.trtllm.fp4_quantize(
        tensor,
        global_scale,
        16,
        False,
    )
    return fp4, block_scale, global_scale


def check_nvfp4_gemm_backends(dtype: torch.dtype, m: int, n: int, k: int) -> None:
    require(IS_CUBLASLT_AVAILABLE, "cuBLASLt FP4 GEMM backend is not available")
    torch.manual_seed(7)
    x = (torch.randn((m, k), device="cuda", dtype=dtype) * 0.5).contiguous()
    w = (torch.randn((n, k), device="cuda", dtype=dtype) * 0.1).contiguous()
    x_fp4, x_sf, x_global_scale = quantize_nvfp4(x)
    w_fp4, w_sf, w_global_scale = quantize_nvfp4(w)
    alpha = 1.0 / (x_global_scale * w_global_scale)

    with torch.inference_mode():
        cutlass = torch.ops.trtllm.nvfp4_gemm(
            x_fp4,
            w_fp4,
            x_sf,
            w_sf,
            alpha,
            dtype,
            allowed_backends="cutlass",
        )
        cublaslt = torch.ops.trtllm.nvfp4_gemm(
            x_fp4,
            w_fp4,
            x_sf,
            w_sf,
            alpha,
            dtype,
            allowed_backends="cublaslt",
        )

        x_small = x[:8].contiguous()
        x_small_fp4, x_small_sf, x_small_global_scale = quantize_nvfp4(x_small)
        alpha_small = 1.0 / (x_small_global_scale * w_global_scale)
        cuda_core = torch.ops.trtllm.nvfp4_gemm(
            x_small_fp4,
            w_fp4,
            x_small_sf,
            w_sf,
            alpha_small,
            dtype,
            allowed_backends="cuda_core",
        )

    torch.cuda.synchronize()
    require(cutlass.shape == (m, n), f"unexpected CUTLASS output shape {cutlass.shape}")
    require(cublaslt.shape == (m, n), f"unexpected cuBLASLt output shape {cublaslt.shape}")
    require(cuda_core.shape == (8, n), f"unexpected CUDA-core output shape {cuda_core.shape}")
    require(torch.isfinite(cutlass).all().item(), "CUTLASS output contains non-finite values")
    require(torch.isfinite(cublaslt).all().item(), "cuBLASLt output contains non-finite values")
    require(torch.isfinite(cuda_core).all().item(), "CUDA-core output contains non-finite values")

    max_diff = (cublaslt.float() - cutlass.float()).abs().max().item()
    print("runtime_path_nvfp4_cutlass_shape", tuple(cutlass.shape), cutlass.dtype)
    print("runtime_path_nvfp4_cublaslt_shape", tuple(cublaslt.shape), cublaslt.dtype)
    print("runtime_path_nvfp4_cuda_core_shape", tuple(cuda_core.shape), cuda_core.dtype)
    print("runtime_path_nvfp4_cublaslt_vs_cutlass_max_diff", max_diff)
    require(max_diff <= 1e-3, f"cuBLASLt and CUTLASS outputs differ by {max_diff}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()

    require_sm121()
    check_trtllm_gen_rejections()
    check_nvfp4_gemm_backends(torch.bfloat16, args.m, args.n, args.k)
    print("runtime_path_smoke_ok")


if __name__ == "__main__":
    main()
