# model/tiny/

Reserved for tiny-AI ONNX artefacts trained via [`ai/`](../../ai/).

No weights are shipped yet — once a first milestone model is trained and
passes the cross-backend gate, it will land here as
`vmaf_tiny_vN.onnx` with a matching `vmaf_tiny_vN.json` manifest.

Artefacts under this directory must:

- Be reproducible from a commit-hashed training run (recipe pinned in `ai/`).
- Pass the ULP-≤2 cross-backend gate under ONNX Runtime CPU / CUDA / SYCL.
- Be signed via Sigstore keyless as part of the release bundle.
