from unittest.mock import patch

import pytest

try:
    from tensorrt_llm._torch.distributed.ops import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping
except ModuleNotFoundError as exc:
    pytest.skip(f"Skipping GB10 allreduce regression test: {exc}",
                allow_module_level=True)


def test_allreduce_ncc_symm_falls_back_to_nccl_on_gb10():
    mapping = Mapping(world_size=2, tp_size=2, rank=0)

    with patch("tensorrt_llm._torch.distributed.ops._is_gb10", return_value=True):
        allreduce = AllReduce(
            mapping=mapping,
            strategy=AllReduceStrategy.NCCL_SYMMETRIC,
        )

    assert allreduce.strategy == AllReduceStrategy.NCCL
