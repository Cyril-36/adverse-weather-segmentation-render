"""Render web service for the adverse-weather segmentation demo."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

from flask import Flask, Response, jsonify, request
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
HEIGHT = 360
WIDTH = 640
FRAME_BYTES = HEIGHT * WIDTH * 4
MODELS = {"S", "same", "cross"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
inference_lock = Lock()

app = Flask(__name__, static_folder=str(ROOT / "public"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = FRAME_BYTES


@lru_cache(maxsize=2)
def load_model(key):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Render's free instance has a 512 MB limit. Release intermediate tensors
    # between requests instead of retaining a separate CPU arena per model.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(
        str(ROOT / "weights" / f"{key}.onnx"),
        options,
        providers=["CPUExecutionProvider"],
    )


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.errorhandler(413)
def oversized_frame(_error):
    return jsonify(error="The input frame is too large."), 413


@app.post("/api/segment")
def segment():
    keys = request.args.getlist("model")
    if not 1 <= len(keys) <= 2 or len(set(keys)) != len(keys) or set(keys) - MODELS:
        return jsonify(error="Select one or two supported models."), 400
    if request.mimetype != "application/octet-stream":
        return jsonify(error="Send an RGBA frame."), 415
    frame = request.get_data(cache=False)
    if len(frame) != FRAME_BYTES:
        return jsonify(error="The RGBA frame must be 640 × 360 pixels."), 400
    if not inference_lock.acquire(blocking=False):
        return jsonify(
            error="The server is processing another frame. Try again shortly."
        ), 503
    try:
        rgb = np.frombuffer(frame, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3]
        normalized = (rgb.astype(np.float32) / np.float32(255) - MEAN) / STD
        pixels = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        masks = [
            load_model(key).run(["labels"], {"pixels": pixels})[0].tobytes()
            for key in keys
        ]
        return Response(
            b"".join(masks),
            mimetype="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
    finally:
        inference_lock.release()
