# Adverse Weather Segmentation on Render

Render serves the demo and returns segmentation masks for 360 × 640 frames. The web client assembles the preview and exports. The service processes request frames in memory. The original recording stays in the browser.

`render.yaml` defines a free Python web service. The build installs the pinned dependencies and Gunicorn starts the service on Render's assigned port. Health is available at `/healthz`. The public web root is `public/` and the ONNX weights are in the private source tree under `weights/`.

The site contains aggregate ACDC development results and a sample based on [Wintry Dashcam by brownpau](https://commons.wikimedia.org/wiki/File:Wintry_Dashcam_(Feb_2016,_720p).webm), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), trimmed and muted with prediction overlays. The original checkpoints were trained on [ACDC](https://acdc.vision.ee.ethz.ch/) starting from [NVIDIA SegFormer-B0](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024). The models are for noncommercial research and evaluation under the [SegFormer license](public/models/LICENSE-SegFormer.txt).
