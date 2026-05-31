from __future__ import annotations

import argparse

from tensorrt_llm import LLM, SamplingParams


def run_generation(
    model: str,
    prompt: str,
    max_tokens: int,
    max_input_len: int,
    max_seq_len: int,
    max_num_tokens: int,
) -> None:
    llm = LLM(
        model=model,
        dtype="float16",
        trust_remote_code=False,
        tensor_parallel_size=1,
        max_batch_size=1,
        max_input_len=max_input_len,
        max_seq_len=max_seq_len,
        max_num_tokens=max_num_tokens,
        cuda_graph_config=None,
    )
    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    text = outputs[0].outputs[0].text
    print("minimal_generation_model", model)
    print("minimal_generation_text", repr(text))
    if not isinstance(text, str):
        raise TypeError(f"expected generated text, got {type(text)!r}")
    print("minimal_generation_ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="TensorRT-LLM on GB10")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--max-input-len", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--max-num-tokens", type=int, default=64)
    args = parser.parse_args()
    run_generation(
        args.model,
        args.prompt,
        args.max_tokens,
        args.max_input_len,
        args.max_seq_len,
        args.max_num_tokens,
    )


if __name__ == "__main__":
    main()
