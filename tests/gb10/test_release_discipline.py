from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def workflow_header(path: str) -> str:
    return "\n".join(read(path).splitlines()[:24])


def test_gb10_release_workflow_builds_wheel_and_container_for_sm121a():
    workflow = read(".github/workflows/gb10-release.yml")

    assert '"gb10-release-discipline"' in workflow
    assert '"gb10-integration"' in workflow
    assert "gb10-trtllm-v*" in workflow
    assert "packages: write" in workflow
    assert 'default: "121-real"' in workflow
    assert "--cuda_architectures=${{ needs.setup.outputs.cuda_archs }}" in workflow
    assert "docker/build-push-action" in workflow
    assert "tensorrt_llm*.whl" in workflow
    assert "dist/SHA256SUMS" in workflow
    assert "dist/image-digest.txt" in workflow
    assert "dist/release-metadata.json" in workflow
    assert "steps.image.outputs.digest" in workflow
    assert "gh release upload" in workflow
    assert "--notes-file dist/release-notes.md" in workflow
    assert "needs.setup.outputs.push_image == 'true'" in workflow


def test_gb10_release_passes_trt_llm_version_to_docker_builds():
    workflow = read(".github/workflows/gb10-release.yml")
    dockerfile = read("docker/Dockerfile.multi")
    docker_makefile = read("docker/Makefile")

    assert "trt_llm_ver: ${{ steps.meta.outputs.trt_llm_ver }}" in workflow
    assert 'trt_llm_ver="$(grep' in workflow
    assert 'echo "trt_llm_ver=${trt_llm_ver}"' in workflow
    assert workflow.count("TRT_LLM_VER=${{ needs.setup.outputs.trt_llm_ver }}") == 2
    assert dockerfile.count("ARG TRT_LLM_VER=dev") == 2
    assert (
        "--extra-cmake-vars TRTLLM_USE_PREBUILT_INTERNAL_CUTLASS_KERNELS=OFF"
        in workflow
    )
    assert (
        "--extra-cmake-vars TRTLLM_USE_PREBUILT_INTERNAL_CUTLASS_KERNELS=OFF"
        in docker_makefile
    )


def test_cmake_and_docker_defaults_allow_native_sm121a():
    cuda_config = read("cpp/cmake/modules/cuda_configuration.cmake")
    deep_ep = read("cpp/tensorrt_llm/deep_ep/CMakeLists.txt")
    docker_makefile = read("docker/Makefile")
    build_wheel = read("scripts/build_wheel.py")

    assert 'VERSION_GREATER_EQUAL "13.0"' in cuda_config
    assert "ARCHITECTURES_NO_COMPATIBILITY 87 101 121" in cuda_config
    assert "set(CUDA_ARCH_FEATURE ${CMAKE_MATCH_3})" in deep_ep
    assert "set(CUDA_ARCH_POSTFIX ${CMAKE_MATCH_4})" in deep_ep
    assert '"${CUDA_ARCH_MAJOR}${CUDA_ARCH_MINOR}a${CUDA_ARCH_POSTFIX}"' in deep_ep
    assert "'121-real'" in docker_makefile
    assert '"ENABLE_SM121": "1"' in build_wheel


def test_gb10_cmake_avoids_non_sm121_placeholder_fatbins():
    cuda_config = read("cpp/cmake/modules/cuda_configuration.cmake")
    trtllm_cmake = read("cpp/tensorrt_llm/CMakeLists.txt")
    cutlass_cmake = read("cpp/tensorrt_llm/kernels/cutlass_kernels/CMakeLists.txt")
    flash_mla_cmake = read("cpp/tensorrt_llm/kernels/flashMLA/CMakeLists.txt")
    trtllm_gen_fmha_cmake = read(
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/CMakeLists.txt"
    )
    trtllm_gen_fmha_runner = read(
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.cpp"
    )
    trtllm_gen_fmha_runner_h = read(
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
    )
    plugin_cmake = read("cpp/tensorrt_llm/plugins/CMakeLists.txt")
    thop_cmake = read("cpp/tensorrt_llm/thop/CMakeLists.txt")
    low_latency_gemm_plugin = read(
        "cpp/tensorrt_llm/plugins/lowLatencyGemmPlugin/lowLatencyGemmPlugin.h"
    )
    attention_op_h = read("cpp/tensorrt_llm/common/attentionOp.h")
    moe_dispatch = read(
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/"
        "moe_gemm_template_dispatch.h"
    )

    assert 'PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"' in cuda_config
    assert 'TRTLLM_USE_PREBUILT_INTERNAL_CUTLASS_KERNELS "Use' in trtllm_cmake
    assert "TRTLLM_HAS_PREBUILT_INTERNAL_CUTLASS_KERNELS" in trtllm_cmake
    assert "$<LINK_LIBRARY:WHOLE_ARCHIVE,${INTERNAL_CUTLASS_KERNELS_TARGET}>" in trtllm_cmake
    assert "target_link_libraries(${SHARED_TARGET} PUBLIC ${TRTLLM_LINK_LIBS})" in trtllm_cmake
    assert "set(TRTLLM_PRIVATE_LINK_LIBS" in trtllm_cmake
    assert "set(TRTLLM_PRIVATE_LINK_LIBS ${TRTLLM_PRIVATE_LINK_LIBS} moe_gemm_src)" in trtllm_cmake
    assert "target_link_libraries(${SHARED_TARGET} PRIVATE ${TRTLLM_PRIVATE_LINK_LIBS})" in trtllm_cmake
    assert "target_link_libraries(${PLUGIN_SHARED_TARGET} moe_gemm_src kernels_src cutlass_src)" in plugin_cmake
    assert "target_link_libraries(th_common PRIVATE moe_gemm_src kernels_src cutlass_src)" in thop_cmake
    assert 'glob_src_create_target(120 "120f;121")' in cutlass_cmake
    assert "set_cuda_architectures(fb_gemm_src 89 90 100f 120f 121)" in cutlass_cmake
    assert "set_cuda_architectures(fp8_blockscale_gemm_src 89 90 100f 120f 121)" in cutlass_cmake
    assert "set_cuda_architectures(fp4_gemm_src 100f 120f 121)" in cutlass_cmake
    assert "$<TARGET_OBJECTS:_moe_gemm_launcher>" in cutlass_cmake
    assert "tensorrt_llm/kernels/cutlass_kernels/include/low_latency_gemm.h" in low_latency_gemm_plugin
    assert "namespace low_latency_gemm = tensorrt_llm::kernels::" in low_latency_gemm_plugin
    assert "flash_mla_stub.cpp" in flash_mla_cmake
    assert '"90" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG' in flash_mla_cmake
    assert "TRTLLM_GEN_FMHA_HAS_PREBUILT_KERNELS" in trtllm_gen_fmha_cmake
    assert "if(TRTLLM_GEN_FMHA_HAS_PREBUILT_KERNELS)" in trtllm_gen_fmha_cmake
    assert "PRIVATE TRTLLM_GEN_FMHA_HAS_PREBUILT_KERNELS" in trtllm_gen_fmha_cmake
    assert "#if defined(TRTLLM_GEN_FMHA_HAS_PREBUILT_KERNELS)" in trtllm_gen_fmha_runner
    assert "TrtLlmGen FMHA kernels are only built for SM100/SM103" in trtllm_gen_fmha_runner
    assert '#include "fmhaKernels.h"' not in trtllm_gen_fmha_runner_h
    assert "class TllmGenFmhaKernel;" in trtllm_gen_fmha_runner_h
    assert '#include "cutlass/fast_math.h"' in attention_op_h
    assert '#include "cutlass/numeric_types.h"' in attention_op_h
    assert "#if defined(EXCLUDE_SM_80)" in moe_dispatch
    assert "Fused MoE SM80 launcher is not built for this architecture set" in moe_dispatch


def test_gb10_cutlass_generator_drops_disabled_arch_families():
    generator = read(
        "cpp/tensorrt_llm/kernels/cutlass_kernels/python/generate_kernels.py"
    )
    sm90_body = generator[
        generator.index("def generate_sm90_operations"):
        generator.index("\ndef calc_shape_mnk_sm100")
    ]
    sm80_body = generator[
        generator.index("def generate_sm80_operations"):
        generator.index('\n\nif __name__ == "__main__"')
    ]

    assert "if not is_arch_enabled:\n        return []" in sm90_body
    assert "if not is_arch_enabled:\n        return []" in sm80_body


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
