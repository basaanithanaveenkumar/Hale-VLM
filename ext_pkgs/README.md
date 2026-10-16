# ext_pkgs - vendor SDK sources (git submodules)

Shallow (`--depth 1`) submodules of the upstream toolchains the deployment pipelines target.
They are **reference sources** (samples, model recipes, op support tables, converters to read or
build from); the Python packages themselves are installed through the `deploy-*` extras.

```bash
git submodule update --init --depth 1 ext_pkgs/<name>     # fetch one (never --recursive: nested
                                                          # submodules of executorch etc. are huge)
```

| name | upstream | vendor | purpose | used by (hale_core.deploy stage) |
|---|---|---|---|---|
| TensorRT | NVIDIA/TensorRT | NVIDIA | ONNX parser, plugins, samples, trtexec | `TensorRTEngineStage` |
| TensorRT-LLM | NVIDIA/TensorRT-LLM | NVIDIA | LLM serving (PyTorch backend), `trtllm-serve` | `TensorRTLLMBundleStage` |
| TensorRT-Edge-LLM | NVIDIA/TensorRT-Edge-LLM | NVIDIA | on-device LLM/VLM runtime for Jetson Thor / DRIVE | HF checkpoint from `HuggingFaceCheckpointStage` |
| Model-Optimizer | NVIDIA/Model-Optimizer | NVIDIA | AutoCast mixed precision, ONNX / HF PTQ | `ModelOptAutoCastStage`, `ModelOptQuantizeStage`, `TensorRTLLMBundleStage` |
| Torch-TensorRT | pytorch/TensorRT | NVIDIA / PyTorch | TensorRT inside `torch.export` programs | `TorchTensorRTStage` |
| mlx | ml-explore/mlx | Apple | arrays, `mx.export_function`, fast kernels | `MLXConvertStage`, `MLXFunctionSession` |
| mlx-lm | ml-explore/mlx-lm | Apple | HF LLM → MLX conversion / quantization | `MLXHuggingFaceConvertStage(mlx_lm)` |
| mlx-vlm | Blaizzy/mlx-vlm | community | HF VLM → MLX (supported model types only) | `MLXHuggingFaceConvertStage(mlx_vlm)` |
| AMDMIGraphX | ROCm/AMDMIGraphX | AMD | ONNX → MIGraphX programs (Instinct / Radeon) | `MIGraphXCompileStage`, `MIGraphXSession` |
| Quark | amd/Quark | AMD | ONNX / PyTorch quantization (A16W8, BF16 ...) | `QuarkQuantizeStage` |
| RyzenAI-SW | amd/RyzenAI-SW | AMD | Ryzen AI NPU examples, Vitis AI EP setup | `OnnxRuntimePackageStage` (VitisAI EP) |
| ai-hub-models | qualcomm/ai-hub-models | Qualcomm | AI Hub recipes, device list, app samples | `QAIHubCompileStage` |
| executorch | pytorch/executorch | PyTorch / Qualcomm | on-device runtime, QNN backend, Android demos | `ExecuTorchQNNStage` |
