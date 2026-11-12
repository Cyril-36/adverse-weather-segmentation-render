"""Render web service for the adverse-weather segmentation demo."""

import subprocess
import sys
from threading import Lock

from flask import Flask, Response, jsonify, request

from inference import FRAME_BYTES, MASK_BYTES, MODELS, ROOT

inference_lock = Lock()

app = Flask(__name__, static_folder=str(ROOT / "public"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = FRAME_BYTES


def predict_in_process(frame, key):
    # A short-lived process releases all runtime allocations before the next model.
    result = subprocess.run(
        [sys.executable, str(ROOT / "inference.py"), key],
        input=frame,
        capture_output=True,
        check=True,
        timeout=25,
    )
    if len(result.stdout) != MASK_BYTES:
        raise ValueError("Unexpected mask length")
    return result.stdout


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
        masks = [predict_in_process(frame, key) for key in keys]
        return Response(
            b"".join(masks),
            mimetype="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
    except subprocess.TimeoutExpired:
        return jsonify(error="Inference timed out. Please retry this frame."), 504
    except (subprocess.CalledProcessError, ValueError) as error:
        app.logger.error("Inference process failed: %s", error)
        return jsonify(error="Inference failed. Please retry this frame."), 503
    finally:
        inference_lock.release()
