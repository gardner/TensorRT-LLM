from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def workflow_header(path: str) -> str:
    return "\n".join(read(path).splitlines()[:24])


def test_gb10_release_workflow_builds_wheel_and_container_for_sm121a():
    workflow = read(".github/workflows/gb10-release.yml")

    assert "gb10-trtllm-v*" in workflow
    assert 'default: "121-real"' in workflow
    assert "--cuda_architectures=${{ needs.setup.outputs.cuda_archs }}" in workflow
    assert "docker/build-push-action" in workflow
    assert "tensorrt_llm*.whl" in workflow
    assert "gh release upload" in workflow


def test_cmake_and_docker_defaults_allow_native_sm121a():
    cuda_config = read("cpp/cmake/modules/cuda_configuration.cmake")
    docker_makefile = read("docker/Makefile")
    build_wheel = read("scripts/build_wheel.py")

    assert 'VERSION_GREATER_EQUAL "13.0"' in cuda_config
    assert "ARCHITECTURES_NO_COMPATIBILITY 87 101 121" in cuda_config
    assert "'121-real'" in docker_makefile
    assert '"ENABLE_SM121": "1"' in build_wheel


def test_upstream_bot_schedules_are_manual_only_for_the_fork():
    assert "schedule:" not in workflow_header(
        ".github/workflows/auto-close-inactive-issues.yml"
    )
    assert "schedule:" not in workflow_header(
        ".github/workflows/label_community_pr.yml"
    )


def test_gb10_nvfp4_moe_loader_avoids_typed_storage_data_ptr():
    quantization = read("tensorrt_llm/_torch/modules/fused_moe/quantization.py")

    assert ".storage().data_ptr()" not in quantization
    assert "dst_w3_w1_weight_scale.data_ptr()" in quantization
    assert "dst_w3_w1_weight.data_ptr()" in quantization


def test_gb10_fp8_prequant_and_nvfp4_linear_allow_sm121_cuda_core():
    quantization_ops = read(
        "tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/quant.py"
    )
    linear = read("tensorrt_llm/_torch/modules/linear.py")

    assert "capability in ((8, 9), (12, 0), (12, 1))" in quantization_ops
    assert "capability[0] == 12 and capability[1] in (0, 1)" in linear


def test_gb10_cutlass_moe_filters_sm121_shared_memory_configs():
    heuristic = read("cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp")
    launcher = read(
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/"
        "moe_gemm_tma_ws_launcher.inl"
    )

    assert "kMinSmemForFullTileSet = 120 * 1024" in heuristic
    assert "cudaDevAttrMaxSharedMemoryPerBlockOptin" in heuristic
    assert "std::remove_if(candidate_configs.begin()" in heuristic
    assert "CtaShape128x128x64B" in heuristic
    assert "sizeof(typename GemmKernel_::SharedStorage)" in launcher
    assert "MoE grouped GEMM requires %d bytes shared memory" in launcher


def test_gb10_allreduce_avoids_nccl_symmetric_tactics():
    custom_ops = read("tensorrt_llm/_torch/custom_ops/torch_custom_ops.py")

    assert "def _is_gb10() -> bool:" in custom_ops
    assert '"GB10" in torch.cuda.get_device_name()' in custom_ops
    assert "valid_strategies = [AllReduceStrategy.NCCL.value]" in custom_ops
    assert "if _is_gb10() else AllReduceStrategy.NCCL_SYMMETRIC.value" in custom_ops
