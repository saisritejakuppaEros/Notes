
## Title: On-Device AI with Qualcomm AI Hub tags: [on-device-ai, edge-ai, qualcomm, segmentation, quantization, tinyml] created: 2026-07-17

# On-Device AI with Qualcomm AI Hub

## 1. Overview

On-device AI refers to running machine learning models **directly on a physical device** (smartphone, embedded board, wearable, etc.) rather than sending data to the cloud for inference. This course/notes set walks through the full lifecycle of taking a trained model and deploying it efficiently onto edge hardware using **Qualcomm AI Hub**.

The end-to-end workflow covered here includes:

1. Building a **computational graph** from a trained model
2. **Compiling** that graph for a specific target device
3. **Running/validating** it efficiently on-device
4. **Quantizing** the model to reduce size and increase speed
5. **Deploying** it into a real-time application (e.g., camera-based segmentation)

---

## 2. Why On-Device AI?

Running inference locally — instead of relying on a cloud API — has strong advantages for certain modalities and use cases:

### Key Modalities Suited for On-Device Inference

1. **Audio and speech** — wake-word detection, transcription, voice assistants
2. **Image and video** — segmentation, detection, enhancement, AR effects
3. **Sensors and text** — IMU data, contextual/NLP tasks

### Core Benefits

- **Multimodal flexibility** — on-device models can be mixed/combined for multimodal AI experiences
- **Cost-effective** — no recurring cloud inference/API costs
- **Privacy-preserving** — sensitive data (voice, camera, sensor data) never leaves the device
- **Personalization** — models can adapt to a user's own local/custom data
- **Low latency & efficiency** — no network round-trip; predictable performance

> **GenAI Applications**: On-device generative AI has strong commercial potential for deployed consumer applications (e.g., photo editing, personal assistants, offline chat).

---

## 3. The "Device-in-the-Loop" Workflow

This is the general methodology for taking any model from training to a working on-device deployment. It requires access to a **physical or cloud-hosted target device** throughout the process.

|Step|Description|
|---|---|
|1. Capture|Represent the trained model as a **computational graph**|
|2. Compile|Compile the graph specifically for the **target device's hardware**|
|3. Validate|Check that on-device **numerics** match the original model (accuracy/precision check)|
|4. Measure|Benchmark **performance** (latency, memory, throughput) on the real device|
|5. Deploy|Package and ship the model into the application|

> ⚠️ You need an actual (or hub-provisioned) device to complete this loop — you can't fully validate on-device behavior with a desktop CPU/GPU alone.

**Qualcomm AI Hub** provides the infrastructure to perform all five of these steps without needing to physically own every target device — it gives remote access to real hardware for compiling, profiling, and validating models.

---

## 4. Real-Time Image Segmentation on Device

### 4.1 Types of Segmentation

|Type|Definition|
|---|---|
|**Semantic Segmentation**|Every pixel is classified into a specific class (e.g., "road", "sky", "car") — instances of the same class are not distinguished|
|**Instance Segmentation**|Every pixel is assigned to a **separate object instance** — two cars are segmented as two distinct entities, not just "car"|

### 4.2 Example: Loading a Segmentation Model (FFNet-40S)

Qualcomm AI Hub Models (`qai_hub_models`) ships pretrained, edge-optimized models ready for profiling and deployment.

```python
from qai_hub_models.models.ffnet_40s import Model
from torchinfo import summary

# Load from pre-trained weights
model = Model.from_pretrained()
input_shape = (1, 3, 1024, 2048)
stats = summary(
    model,
    input_size=input_shape,
    col_names=["num_params", "mult_adds"]
)
print(stats)
```

**Resulting model summary (abridged):**

```
==============================================================================================================
Layer (type:depth-idx)                                       Param #                   Mult-Adds
==============================================================================================================
FFNet40S
├─FFNet: 1-1
│    └─ResNetS: 2-1
│    │    └─Conv2d, BatchNorm2d, ReLU, Sequential blocks ...
│    └─FFNetUpHead: 2-2
│    │    └─Sequential: 3-11                                 1,208,147                 26,571,541,312
==============================================================================================================
Total params: 13,911,283
Trainable params: 13,911,283
Non-trainable params: 0
Total mult-adds (G): 62.38
Input size (MB): 25.17
Forward/backward pass size (MB): 1269.30
Params size (MB): 55.65
Estimated Total Size (MB): 1350.11
==============================================================================================================
```

**Key takeaway:** ~13.9M parameters and ~62.38 GMACs — this gives a sense of the compute budget the target device's NPU/GPU/CPU must handle per inference.

### 4.3 Real-Time Constraint (Latency Budget)

For a **30 FPS** video stream (e.g., a self-driving car camera feed), each frame must be processed within:

$$ \frac{1000\text{ ms}}{30\text{ frames}} \approx 33.3\text{ ms per frame} $$

> The notes round this to **~30 ms** as the target inference budget — if inference takes longer than this, the pipeline cannot sustain real-time 30 FPS throughput.

### 4.4 Popular Real-Time Segmentation Model Families

1. **ResNet** — general-purpose backbone, widely used as a feature extractor
2. **HRNet** — maintains high-resolution representations throughout the network
3. **FANet** — Fast Attention Network, optimized for real-time segmentation
4. **DDRNet** — Dual Dynamic Resolution Network, balances speed and accuracy via dual-resolution branches

**FFNet (Fuss-Free Network):**

- Designed for efficient, "fuss-free" real-time semantic segmentation
- Paper: https://arxiv.org/html/2404.07847v1
- GitHub: https://github.com/Qualcomm-AI-research/FFNet

---

## 5. Setting Up Qualcomm AI Hub

### 5.1 Authentication & Configuration

```python
import qai_hub
from utils import get_ai_hub_api_token

ai_hub_api_token = get_ai_hub_api_token()

!qai-hub configure --api_token $ai_hub_api_token

# Run the built-in demo for the model
%run -m qai_hub_models.models.ffnet_40s.demo
```

### 5.2 Selecting a Target Device

AI Hub exposes a catalog of real, physical devices (mostly Snapdragon-powered Android phones/tablets) that jobs can be submitted to remotely.

```python
devices = [
    "Samsung Galaxy S22 Ultra 5G",
    "Samsung Galaxy S22 5G",
    "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8",
    "Xiaomi 12",
    "Xiaomi 12 Pro",
    "Samsung Galaxy S23",
    "Samsung Galaxy S23+",
    "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24",
    "Samsung Galaxy S24 Ultra",
    "Samsung Galaxy S24+",
]

import random
selected_device = random.choice(devices)
print(selected_device)

# Run the export/profiling job on the selected device
%run -m qai_hub_models.models.ffnet_40s.export -- --device "$selected_device"
```

### 5.3 Interpreting On-Device Results

After the job runs, AI Hub returns:

- **Estimated inference time** (latency, in ms)
- **Estimated peak memory usage**
- **Compute unit(s) used** (CPU / GPU / NPU)
- **PSNR comparison** between on-device output and local-CPU (reference) output

```
Waiting for profile job (jqp4d7wqp) completion. Type Ctrl+C to stop waiting at any time.
✅ SUCCESS

------------------------------------------------------------
Performance results on-device for Ffnet_40s.
------------------------------------------------------------
Device                     : Samsung Galaxy S23 (13)
Runtime                    : TFLITE
Estimated inference time (ms): 22.7
Estimated peak memory usage (MB): [3, 5]
Total # Ops                : 92
Compute Unit(s)            : NPU (92)

Comparing on-device vs. local-cpu inference for Ffnet_40s.

+-------------+-------------------+-------+
| output_name | shape             | psnr  |
+-------------+-------------------+-------+
| output_0    | (1, 19, 128, 256) | 62.96 |
+-------------+-------------------+-------+
```

#### Understanding PSNR (Peak Signal-to-Noise Ratio)

- PSNR quantifies how close the **on-device output** is to the **reference (local CPU/cloud) output**.
- **Rule of thumb: PSNR > 30 dB is considered good.**
- A PSNR of **62.96 dB** (as seen above) indicates the on-device (quantized/compiled) model output is _almost numerically identical_ to the full-precision reference — i.e., negligible accuracy loss from compilation/quantization.
- In this example, **22.7 ms** inference time is well within the **~33 ms** budget needed for 30 FPS real-time performance. ✅

---

## 6. Module 3 — Graph Capture, Compile, Deploy, Evaluate

The core on-device pipeline can be summarized in four stages:

1. **Graph Capture** — Capture the model's computation as a static graph
2. **Compile** — Compile that graph to run on the local/target device
3. **Deploy** — Push the compiled artifact to the device/app
4. **Evaluate** — Measure accuracy (numerics) and performance (latency/memory)

### 6.1 Graph Capture (Tracing)

```python
from qai_hub_models.models.ffnet_40s import Model as FFNet_40s

# Load from pre-trained weights
ffnet_40s = FFNet_40s.from_pretrained()

import torch
input_shape = (1, 3, 1024, 2048)
example_inputs = torch.rand(input_shape)

# Trace the model into a static computational graph
traced_model = torch.jit.trace(ffnet_40s, example_inputs)
```

> `traced_model` now contains a frozen graph representation of all the modules/operations in the model — this is what gets handed off to the compiler.

### 6.2 Device Discovery

```python
import qai_hub
import qai_hub_models
from utils import get_ai_hub_api_token

ai_hub_api_token = get_ai_hub_api_token()
!qai-hub configure --api_token $ai_hub_api_token

# List all devices available through AI Hub (Google, Samsung, Xiaomi, etc.)
for device in qai_hub.get_devices():
    print(device.name)

devices = [
    "Samsung Galaxy S22 Ultra 5G", "Samsung Galaxy S22 5G", "Samsung Galaxy S22+ 5G",
    "Samsung Galaxy Tab S8", "Xiaomi 12", "Xiaomi 12 Pro",
    "Samsung Galaxy S23", "Samsung Galaxy S23+", "Samsung Galaxy S23 Ultra",
    "Samsung Galaxy S24", "Samsung Galaxy S24 Ultra", "Samsung Galaxy S24+",
]

import random
selected_device = random.choice(devices)
print(selected_device)
```

### 6.3 Compiling for the Target Device

```python
device = qai_hub.Device(selected_device)

# Compile for target device
compile_job = qai_hub.submit_compile_job(
    model=traced_model,                   # Traced PyTorch model
    input_specs={"image": input_shape},   # Input specification
    device=device,                        # Target device
)

# Download and save the compiled target model for on-device use
target_model = compile_job.get_target_model()
```

During compilation, AI Hub also performs a **compatibility check** between the traced model's operations and the target device's supported runtime/hardware.

### 6.4 Compiled Artifact Formats

|Format|Target Runtime|
|---|---|
|`.tflite`|TensorFlow Lite runtime — **Android**|
|`.onnx`|ONNX Runtime — **Windows**|
|Qualcomm AI Engine format|**Qualcomm/embedded devices**|

**Reference:** https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-pytorch-to-tflite

### 6.5 Why Qualcomm AI (Snapdragon Platforms)?

1. **Mobile-first** — purpose-built for smartphones/tablets
2. **Low latency** — hardware-accelerated inference paths
3. **Broad device coverage** — wide range of devices to test against
4. **Energy efficient** — critical for battery-powered devices
5. **Hardware acceleration** — via the **Qualcomm AI Engine delegate**, which routes ops to the most efficient compute unit

### 6.6 Compute Units

|Unit|Role|
|---|---|
|**CPU**|Runs most general-purpose operations; most flexible, least efficient for large NN workloads|
|**GPU**|Higher throughput for parallel computation|
|**NPU** (Neural Processing Unit)|Purpose-built silicon for running neural network operations extremely efficiently|

> Modern flagship devices include a dedicated **NPU**, allowing neural network inference to run with far greater power/performance efficiency than CPU or GPU alone.

**Cross-validation technique:** Run the same image through two different backends (e.g., cloud/reference vs. on-device/compiled), then compare **PSNR** between the two outputs to confirm the compiled/quantized model hasn't degraded accuracy.

---

## 7. Module 4 — Quantization

**Quantization** reduces model precision (and therefore size and compute cost) while aiming to preserve accuracy — a critical step for efficient on-device deployment.

### 7.1 Precision Formats

|Format|Bit-width|Notes|
|---|---|---|
|**FP32**|32-bit float|Full precision, largest size, most compute-intensive|
|**INT8**|8-bit integer|~4x smaller than FP32; much faster on NPUs/quantization-aware hardware|

### 7.2 The Quantization Formula

Quantization maps floating-point values to integers using a **scale** and a **zero-point**:

```
scale = 2.35
zero_point = 0

final_value = round(value / scale) - zero_point
```

- **Scale** — determines the step size between representable integer values
- **Zero-point** — an offset allowing asymmetric ranges (e.g., for ReLU outputs which are non-negative) to be represented efficiently
- **Quantization error** — the small discrepancy introduced when converting between integer and floating-point representations; this is the accuracy cost being paid for the efficiency gain

### 7.3 What Gets Quantized

- **Weight quantization** — quantizing the learned model parameters
- **Activation quantization** — quantizing the intermediate outputs (activations) during inference

### 7.4 Common Quantization Schemes

|Scheme|Description|
|---|---|
|**W8A8** (weights INT8 / activations INT8)|Standard mobile-vision quantization scheme|
|**W4A16** (weights INT4 / activations INT16)|Common in **LLM** quantization, where weights dominate memory and can tolerate more aggressive compression|

### 7.5 Quantization Methods

|Method|Description|
|---|---|
|**Post-Training Quantization (PTQ)**|Applied _after_ the model is fully trained; requires only a few hundred calibration samples to estimate activation ranges|
|**Quantization-Aware Training (QAT)**|Quantization effects are **simulated during training**, so the model learns to be robust to the precision loss — generally yields higher accuracy than PTQ, at the cost of retraining|

---

## 8. Deployment Pipeline for Real-Time Segmentation

A full real-time, camera-driven segmentation pipeline typically looks like this:

```
Camera Stream (30 FPS, RGB/YUV)
        │
        ▼
GPU Preprocessing
  - YUV → RGB conversion
  - Downsampling
        │
        ▼
Model Inference (runs on NPU)
        │
        ▼
Output Mask (e.g., 224 × 224)
        │
        ▼
GPU Postprocessing
  - Exponential smoothing (temporal stability across frames)
  - Upsampling (back to display resolution)
  - Thresholding
  - Normalized box blur (edge smoothing)
```

### 8.1 Stage-by-Stage Breakdown

1. **Camera Stream** — Captures frames at ~30 FPS in RGB or YUV color format
2. **GPU Preprocessing** — Uses **OpenCV (GPU-accelerated)** for fast color-space conversion and downsampling before feeding the model
3. **Model Inference** — Executed via a native **runtime API** (C++ or Java/Kotlin) so the model can run with minimal overhead inside a mobile app
4. **Postprocessing (GPU)** — Cleans up the raw segmentation mask: smooths flicker between frames (exponential smoothing), resizes it back up to the original resolution, applies thresholding to binarize/clean the mask, and blurs edges for a more natural visual result
5. **Runtime Packaging** — All hardware-acceleration runtime dependencies must be bundled with the app so it works correctly on the target device

### 8.2 Runtime Dependencies

|Component|Purpose|
|---|---|
|**Java source**|Android app-layer integration|
|**Native source (C++)**|Performance-critical inference code|
|**TFLite (CPU)**|Fallback/default execution path|
|**GPU delegate**|Offloads ops to the mobile GPU|
|**NPU delegate**|Offloads ops to the dedicated Neural Processing Unit for maximum efficiency|

---

## 9. Quick-Reference Summary

- **Workflow:** Capture → Compile → Validate → Measure → Deploy
- **Real-time budget:** ~33 ms/frame for 30 FPS
- **Good numerical fidelity:** PSNR > 30 dB
- **Compute unit hierarchy (efficiency):** NPU > GPU > CPU for neural network ops
- **Quantization trade-off:** INT8 (or lower) shrinks the model and speeds up inference, at the cost of small, controllable quantization error
- **PTQ vs QAT:** PTQ is fast/cheap (post-hoc calibration); QAT is more accurate but requires retraining
- **Deployment formats:** `.tflite` (Android), `.onnx` (Windows), Qualcomm AI Engine (embedded/Qualcomm silicon)

---

## 10. Useful Links

- FFNet Paper: https://arxiv.org/html/2404.07847v1
- FFNet GitHub: https://github.com/Qualcomm-AI-research/FFNet
- Qualcomm AI Hub Compile Docs: https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html#compiling-pytorch-to-tflite


Edge AI Inference: 

https://github.com/google-ai-edge/gallery
https://developer.apple.com/machine-learning/resources/
https://developers.googleblog.com/litert-maximum-performance-simplified/


Diffusion Models on Edge Devices:
https://arxiv.org/pdf/2504.15298
https://intechhouse.com/blog/generative-ai-on-the-edge-devices-efficiency-without-the-cloud
https://developer.nvidia.com/blog/accelerating-diffusion-models-with-an-open-plug-and-play-offering/
https://developer.nvidia.com/blog/new-stable-diffusion-models-accelerated-with-nvidia-tensorrt/
https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/
