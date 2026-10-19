# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest
from transformers import AutoModelForImageTextToText

from vllm.assets.video import VideoAsset
from vllm.envs import disable_envs_cache

from ....conftest import HfRunner, VllmRunner
from ...utils import check_logprobs_close
from .vlm_utils import model_utils

VIDEO_ASSET = VideoAsset("baby_reading", num_frames=8)

VIDEO_MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-VL-2B-Instruct": {
        "prompt": (
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf": {
        "prompt": (
            "<|im_start|>user <video>\nDescribe this video.<|im_end|>"
            "<|im_start|>assistant\n"
        ),
    },
    "CohereLabs/North-Micro-Vision-Instruct": {
        "prompt": (
            "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
            "<|VISION_START|><|VIDEO_PAD|><|VISION_END|>Describe this video."
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        ),
        "patch_hf_runner": model_utils.cohere_compass_patch_hf_runner,
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
}


@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.8-27B"])
def test_qwen3_5_transformers_dummy_multimodal_generation(vllm_runner, model_id):
    """Hybrid recurrent states work for text, image and video prefill/decode."""
    import numpy as np
    from PIL import Image

    from vllm import SamplingParams
    from vllm.inputs import TextPrompt

    prompts: list[TextPrompt] = [
        {"prompt": "Hello"},
        {
            "prompt": "<|vision_start|><|image_pad|><|vision_end|> Describe.",
            "multi_modal_data": {"image": Image.new("RGB", (64, 64))},
        },
        {
            "prompt": "<|vision_start|><|video_pad|><|vision_end|> Describe.",
            "multi_modal_data": {
                "video": (
                    np.zeros((2, 64, 64, 3), dtype=np.uint8),
                    {
                        "fps": 1.0,
                        "total_num_frames": 2,
                        "frames_indices": [0, 1],
                        "do_sample_frames": False,
                    },
                ),
            },
        },
    ]
    with vllm_runner(
        model_id,
        model_impl="transformers",
        load_format="dummy",
        dtype="bfloat16",
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=1024,
        max_num_seqs=3,
        limit_mm_per_prompt={"image": 1, "video": 1},
        mm_processor_kwargs={"size": {"shortest_edge": 1024, "longest_edge": 4096}},
        hf_overrides={
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
            },
            "vision_config": {"depth": 1, "out_hidden_size": 512},
        },
    ) as model:
        outputs = model.get_llm().generate(
            prompts, SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
        )
        chat_outputs = model.get_llm().chat(
            [{"role": "user", "content": "Hello"}],
            SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
        )
    assert len(outputs) == len(prompts)
    assert all(len(output.outputs[0].token_ids) == 3 for output in outputs)
    assert len(chat_outputs) == 1
    assert len(chat_outputs[0].outputs[0].token_ids) == 3


@pytest.fixture(autouse=True)
def use_spawn_for_video_models(monkeypatch):
    """Single-process workaround for V1 fork safety deadlock issue
    (vllm-project/vllm/issues/17676). Running multiple video models together
    under pytest can cause (possibly flaky) hangs, so they are grouped under
    the same config. Using VLLM_WORKER_MULTIPROC_METHOD=spawn avoids the
    deadlock and allows worker processes to terminate cleanly, and release
    GPU memory between test runs until the issue is fixed."""
    # TODO: Remove monkeypatch once
    # https://github.com/vllm-project/vllm/issues/17676 is fixed.
    disable_envs_cache()
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@pytest.mark.parametrize("model_id", list(VIDEO_MODEL_SETTINGS))
def test_transformers_video_generation(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model_id: str,
):
    prompt = VIDEO_MODEL_SETTINGS[model_id]["prompt"]
    video = (VIDEO_ASSET.np_ndarrays, VIDEO_ASSET.metadata)

    with vllm_runner(
        model_id,
        model_impl="transformers",
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt], 128, num_logprobs=10, videos=[video]
        )

    with hf_runner(
        model_id, dtype="bfloat16", auto_cls=AutoModelForImageTextToText
    ) as hf_model:
        hf_model = model_utils.qwen3_vl_patch_hf_runner(hf_model)
        patch_hf_runner = VIDEO_MODEL_SETTINGS[model_id].get("patch_hf_runner")
        if patch_hf_runner is not None:
            hf_model = patch_hf_runner(hf_model)
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            [prompt], 128, num_logprobs=10, videos=[video]
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
