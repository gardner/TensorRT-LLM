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
