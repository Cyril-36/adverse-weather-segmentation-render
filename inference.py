"""Run one ONNX inference in an isolated process."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HEIGHT = 360
WIDTH = 640
MASK_BYTES = HEIGHT * WIDTH
FRAME_BYTES = MASK_BYTES * 4
MODELS = {"S", "same", "cross"}


def predict(frame, key):
    import numpy as np
    import onnxruntime as ort

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = np.frombuffer(frame, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3]
    normalized = (rgb.astype(np.float32) / np.float32(255) - mean) / std
    pixels = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        str(ROOT / "weights" / f"{key}.onnx"),
        options,
        providers=["CPUExecutionProvider"],
    )
    return session.run(["labels"], {"pixels": pixels})[0].tobytes()


if __name__ == "__main__":
    key = sys.argv[1]
    frame = sys.stdin.buffer.read(FRAME_BYTES + 1)
    if key not in MODELS or len(frame) != FRAME_BYTES:
        raise SystemExit("Invalid model or frame")
    sys.stdout.buffer.write(predict(frame, key))
