---
title: Face Antispoof ONNX — live demo
emoji: 🛡️
colorFrom: gray
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Real vs spoof face check, 600 KB ONNX model, runs in-browser
---

Live demo of [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) (Apache-2.0).
Upload a photo or turn on your webcam: every detected face is labelled **REAL** or **SPOOF** (printed photo / screen replay) by the 600 KB quantized MiniFAS model.

Everything runs locally in your browser with ONNX Runtime Web; no image leaves your device. Models and preprocessing are ported 1:1 from the original repository by the Facenox team. Example images are AI-generated for illustration.

---
Live demo: https://huggingface.co/spaces/r69shabh/face-antispoof-onnx-demo
