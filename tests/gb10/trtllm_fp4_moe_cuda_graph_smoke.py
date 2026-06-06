from __future__ import annotations

import argparse
import os

import torch

import tensorrt_llm._torch  # noqa: F401
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.routing import DefaultMoeRoutingMethod
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_sm121() -> None:
    require(torch.cuda.is_available(), "CUDA is required for the GB10 MoE smoke")
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print("fp4_moe_graph_device", device_name, capability)
    require(capability == (12, 1), f"expected SM121, got {capability}")


def fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    return (
        torch.tensor(448 * 6, dtype=torch.float32, device=tensor.device)
        / tensor.abs().max().float()
    )


def quantize_fp4_weight(
    tensor: torch.Tensor,
    global_scale: torch.Tensor,
    scaling_vector_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp4, block_scale = torch.ops.trtllm.fp4_quantize(
        tensor,
        global_scale,
        scaling_vector_size,
        False,
        False,
    )
    return fp4, block_scale.view(tensor.shape[0], -1).view(torch.float8_e4m3fn)


def create_nvfp4_moe_weights(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    x_global_scale: torch.Tensor,
    scaling_vector_size: int = 16,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for expert_id in range(num_experts):
        w1 = (
            torch.randn((intermediate_size, hidden_size), dtype=dtype, device="cuda") * 0.05
        ).contiguous()
        w2 = (
            torch.randn((hidden_size, intermediate_size), dtype=dtype, device="cuda") * 0.05
        ).contiguous()
        w3 = (
            torch.randn((intermediate_size, hidden_size), dtype=dtype, device="cuda") * 0.05
        ).contiguous()

        w1_global = fp4_global_scale(w1)
        w2_global = fp4_global_scale(w2)
        w3_global = fp4_global_scale(w3)
        w3_w1_global = torch.minimum(w1_global, w3_global)

        w1_fp4, w1_block_scale = quantize_fp4_weight(w1, w3_w1_global, scaling_vector_size)
        w2_fp4, w2_block_scale = quantize_fp4_weight(w2, w2_global, scaling_vector_size)
        w3_fp4, w3_block_scale = quantize_fp4_weight(w3, w3_w1_global, scaling_vector_size)

        prefix = f"{expert_id}"
        weights[f"{prefix}.w1.weight"] = w1_fp4
        weights[f"{prefix}.w2.weight"] = w2_fp4
        weights[f"{prefix}.w3.weight"] = w3_fp4
        weights[f"{prefix}.w1.weight_scale"] = w1_block_scale
        weights[f"{prefix}.w2.weight_scale"] = w2_block_scale
        weights[f"{prefix}.w3.weight_scale"] = w3_block_scale

        # TRT-LLM's NVFP4 loader expects checkpoint input_scale values, then
        # stores their reciprocal as the runtime quantization multiplier.
        input_scale = 1.0 / x_global_scale
        weights[f"{prefix}.w1.input_scale"] = input_scale
        weights[f"{prefix}.w2.input_scale"] = input_scale
        weights[f"{prefix}.w3.input_scale"] = input_scale
        weights[f"{prefix}.w1.weight_scale_2"] = 1.0 / w3_w1_global
        weights[f"{prefix}.w2.weight_scale_2"] = 1.0 / w2_global
        weights[f"{prefix}.w3.weight_scale_2"] = 1.0 / w3_w1_global

    return weights


def build_cutlass_nvfp4_moe(
    *,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
    max_num_tokens: int,
) -> CutlassFusedMoE:
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        mapping=Mapping(),
        quant_config=quant_config,
        moe_backend="CUTLASS",
        moe_disable_finalize_fusion=False,
        max_num_tokens=max(256, max_num_tokens),
        use_cuda_graph=True,
    )
    routing_method = DefaultMoeRoutingMethod(top_k=top_k)
    with torch.device("cuda"):
        return CutlassFusedMoE(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=False,
            model_config=model_config,
        )


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    require(
        actual.shape == expected.shape, f"{name}: shape mismatch {actual.shape} != {expected.shape}"
    )
    require(
        actual.dtype == expected.dtype, f"{name}: dtype mismatch {actual.dtype} != {expected.dtype}"
    )
    require(torch.isfinite(actual).all().item(), f"{name}: output contains non-finite values")
    diff = (actual.float() - expected.float()).abs()
    max_diff = diff.max().item()
    print(f"{name}_max_diff", max_diff)
    require(max_diff <= 1e-2, f"{name}: max diff {max_diff} exceeds tolerance")


def run_cuda_graph_smoke(
    *,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    dtype: torch.dtype,
) -> None:
    ok, reason = CutlassFusedMoE.can_implement(QuantAlgo.NVFP4, dtype_activation=dtype)
    print("fp4_moe_graph_cutlass_nvfp4_supported", ok)
    print("fp4_moe_graph_cutlass_nvfp4_reason", reason)
    require(ok, f"Cutlass NVFP4 MoE is not supported: {reason}")

    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    static_x = (
        torch.randn((num_tokens, hidden_size), dtype=dtype, device="cuda") * 0.5
    ).contiguous()
    static_router_logits = torch.randn(
        (num_tokens, num_experts), dtype=dtype, device="cuda"
    ).contiguous()
    x_global_scale = fp4_global_scale(static_x)

    moe = build_cutlass_nvfp4_moe(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        max_num_tokens=num_tokens,
    )
    weights = create_nvfp4_moe_weights(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        x_global_scale=x_global_scale,
    )
    moe.load_weights([weights])
    moe.post_load_weights()
    moe.eval()

    def forward() -> torch.Tensor:
        return moe(static_x, static_router_logits)

    with torch.inference_mode():
        for _ in range(3):
            eager = forward()
        torch.cuda.synchronize()
        eager = eager.clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = forward()
        graph.replay()
        torch.cuda.synchronize()
        replay = graph_out.clone()
        assert_close("fp4_moe_graph_initial_replay", replay, eager)

        next_x = (torch.randn_like(static_x) * 0.5).contiguous()
        next_router_logits = torch.randn_like(static_router_logits).contiguous()
        expected = moe(next_x, next_router_logits).clone()
        static_x.copy_(next_x)
        static_router_logits.copy_(next_router_logits)
        graph.replay()
        torch.cuda.synchronize()
        replay_next = graph_out.clone()
        assert_close("fp4_moe_graph_updated_input_replay", replay_next, expected)

    print("fp4_moe_graph_output_shape", tuple(replay_next.shape), replay_next.dtype)
    print("fp4_moe_cuda_graph_smoke_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("ENABLE_CONFIGURABLE_MOE", "0")
    require_sm121()
    run_cuda_graph_smoke(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype=torch.bfloat16,
    )


if __name__ == "__main__":
    main()
