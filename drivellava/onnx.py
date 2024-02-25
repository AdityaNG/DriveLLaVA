import os

import onnxruntime as ort


def load_model_from_onnx_comma(
    path: str, device="cuda"
) -> ort.InferenceSession:
    # Load the model
    assert device in ["cuda", "cpu"]
    assert os.path.exists(path)

    if device == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    # 'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
        ]
    else:
        providers = ["CPUExecutionProvider"]  # type: ignore

    options = ort.SessionOptions()

    net = ort.InferenceSession(path, options, providers)
    return net
