
Summary:
### 1. Estimating training time and model size

**Training time (70B model, 15T tokens, 1024 H100s)**

- Total training FLOPs ≈ 6 × params × tokens (6 = ~2 FLOPs/param forward + ~4 FLOPs/param backward)
- Each H100 delivers ~1979 TFLOPS peak (BF16), but effective throughput is halved to account for forward+backward not hitting peak, then further scaled by MFU (~0.5 typical)
- Cluster throughput = per-GPU effective FLOPs × num GPUs × seconds/day
- Days to train = total FLOPs ÷ daily cluster throughput

**MFU (Model FLOP Utilization)**

- MFU = actual FLOPs achieved ÷ theoretical peak FLOPs
- 0.3 = poor, 0.5 = typical, 0.6–0.7 = well-optimized, 0.8+ = excellent/rare

**Max model size (8 H100s, AdamW)**

- Each H100 = 80GB memory
- Memory per parameter = 12 bytes (2 param + 2 grad + 4+4 Adam moments)
- Max params = (80GB × 8) ÷ 12 bytes — this is an **upper bound** since it ignores activations, attention caches, workspaces, communication buffers, etc.

### 2. Tensors

- **Rank** = number of dimensions (vector=1, matrix=2, etc.); transformers use rank-4 tensors (batch, seq, heads, dim)
- Memory = num elements × bytes per element
- **Precision formats:**
    - fp32: default, high precision, 4 bytes/value
    - fp16: 2 bytes, narrow dynamic range → underflow risk
    - bf16: same 2 bytes as fp16, but fp32-like dynamic range, lower resolution — solves fp16's instability
    - Mixed precision: bf16 for params/activations/gradients, fp32 for optimizer state (via PyTorch AMP)
    - fp8 (2022): standardized for ML, two H100 variants (E4M3, E5M2)
    - fp4/nvfp4 (2025): 4 bits/value with per-block scale factors; used in Nemotron 3 Super
- Tensors live on CPU by default; must be explicitly moved `.to(device)` or created directly on GPU

### 3. Einops

- **einsum**: generalized matmul with named dimensions instead of `@`/transpose gymnastics; unnamed output dims get summed; `...` handles broadcasting
- **reduce**: named-dimension reduction ops (sum, mean, max, etc.)
- **rearrange**: splits/merges dimensions (e.g., unpacking `total_hidden` into `heads × hidden1`)

### 4. FLOPs basics

- FLOP = one operation; FLOP/s (FLOPS) = speed metric
- Matmul FLOPs = 2 × B × D × K (one multiply + one add per element triple)
- Actual FLOP/s = FLOPs ÷ measured time, compared against hardware's promised peak (which varies by dtype)

### 5. Memory-bound vs. compute-bound

- Execution time = max(compute time, memory transfer time)
- **Arithmetic Intensity** = FLOPs ÷ bytes moved
- **Accelerator Intensity** = peak FLOPs/s ÷ memory bandwidth (H100's "expected" ratio)
- If arithmetic intensity < accelerator intensity → memory-bound; if greater → compute-bound
- **Examples ranked by intensity:** ReLU (¼) → GeLU (5) → dot product (~½) → matrix-vector (~1) → matrix multiply (~n/3, e.g. 341 at n=1024) — only matmul is compute-bound because data gets reused many times
- **Roofline model**: performance rises with intensity until hitting the compute "roof"
- Transformer **training** = compute-bound (large matmuls); **inference** = memory-bound (matrix-vector ops), explaining why more FLOPS doesn't help inference latency much
- MFU can be estimated as min(1, arithmetic intensity ÷ accelerator intensity)

### 6. Backward pass / gradient computation

- Two-layer network example: forward computes h1=XW1, h2=h1W2, loss=(mean(h2))²
- `loss.backward()` applies chain rule back through the graph; `retain_grad()` needed to inspect non-leaf tensor gradients
- Each linear layer needs ~3 matmuls total: 1 forward (XW) + 2 backward (dW and dX)
- Training ≈ 3× the FLOPs of inference alone
- This is the origin of the **6× parameters × tokens** approximation: ~2 FLOPs/param forward, ~4 FLOPs/param backward

### 7. Deep networks, optimizers, memory, training loop

- **Optimizer family tree:**
    - SGD: uses only current gradient
    - Momentum: SGD + exponential average of past gradients
    - AdaGrad: adapts LR per-parameter using cumulative gradient²
    - RMSProp: AdaGrad + exponential averaging (fixes ever-shrinking LR)
    - Adam/AdamW: Momentum + RMSProp combined (tracks 1st and 2nd moment) — most common today
- **Memory per parameter:**
    - Params (bf16): 2 bytes
    - Gradients (bf16): 2 bytes
    - Optimizer state: AdaGrad = 4 bytes (fp32 grad²), Adam = 8 bytes (two moments)
    - Activations: ~2×B×D×L (bf16), needed for backward pass
    - Total (Adam) ≈ 2P + 2P + 8P + activations
- **Training step FLOPs**: 6 × params × batch size
- **Training loop**: load batch → forward → compute loss → backward → optimizer step → zero gradients → repeat

### 8. Basic neural network (MLP) vs. Transformer

- MLP: linear layers + activation, processes inputs independently, no positional info, lower memory/compute, O(D²L) params and FLOPs
- Transformer: adds multi-head self-attention + FFN, processes sequences jointly (every token attends to every other token), needs positional encoding, much higher memory (KV cache, attention matrices) and compute (attention is O(n²) in sequence length)
- Transformer training FLOPs ≈ 6BND²L + O(BN²D) vs. MLP's ≈ 6BD²L
- Key distinction: MLPs have no cross-input interaction; transformers model contextual relationships via self-attention — this is why they scale to billion/trillion-token regimes (GPT, Llama, Gemini) while MLPs stay suited to simpler tabular/classification tasks


# Detailed Notes

# Estimating Training Time and Maximum Model Size

## Example 1: Training a 70B Parameter Model

**Question:** How long would it take to train a **70B parameter** model on **15 trillion tokens** using **1024 NVIDIA H100 GPUs**?

### Step 1: Compute the Total Training FLOPs

A commonly used approximation for training Transformer models is:

$$
\text{Total FLOPs} \approx 6 \times \text{Number of Parameters} \times \text{Number of Training Tokens}
$$

```python
total_flops = 6 * 70e9 * 15e12
```

where:

- **6** – Approximate FLOPs required per parameter per training token (forward + backward pass)
- **70e9** – 70 billion model parameters
- **15e12** – 15 trillion training tokens

6 is know later in the series.

---

### Step 2: Compute the Effective Compute per GPU

An NVIDIA H100 delivers approximately **1979 TFLOPS** (FP16/BF16 Tensor Core performance).

```python
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
```

The division by **2** is an approximation that accounts for the fact that training (forward + backward pass) does not achieve peak inference throughput.

The **Model FLOP Utilization (MFU)** further adjusts the throughput to reflect real-world efficiency.

---

### What is MFU?

**Model FLOP Utilization (MFU)** measures how efficiently the GPU's theoretical compute capability is utilized during training.

It is defined as

$$
\text{MFU} =
\frac{\text{Actual FLOPs Performed}}
{\text{Theoretical Peak FLOPs}}
$$

Typical values are:

| MFU | Meaning |
|------|---------|
| 0.30 | Poor utilization |
| 0.50 | Typical large-scale training |
| 0.60–0.70 | Well-optimized training |
| 0.80+ | Excellent utilization (rare) |

For example, if an H100 is capable of **1979 TFLOPS** but the training pipeline achieves only **990 TFLOPS**, then

$$
\text{MFU} = \frac{990}{1979} \approx 0.5
$$

---

### Step 3: Compute Cluster Throughput

For **1024 GPUs**, the total effective FLOPs per day are

```python
flops_per_day = (
    h100_flop_per_sec
    * mfu
    * 1024
    * 60 * 60 * 24
)
```

---

### Step 4: Estimate Training Time

```python
days = total_flops / flops_per_day
```

This estimates the number of days required to train the model.

---

## Example 2: Maximum Model Size on 8 H100 GPUs

**Question:** What is the largest model that can fit on **8 NVIDIA H100 GPUs** using the **AdamW optimizer**?

---

### Step 1: Available GPU Memory

Each H100 has approximately **80 GB** of memory.

```python
h100_bytes = 80e9
```

---

### Step 2: Memory Required per Parameter

Each model parameter requires memory for:

| Component | Bytes |
|-----------|------:|
| Model Parameter (BF16/FP16) | 2 |
| Gradient | 2 |
| Adam First Moment | 4 |
| Adam Second Moment | 4 |
| **Total** | **12 Bytes** |

Therefore,

```python
bytes_per_parameter = 2 + 2 + (4 + 4)
```

---

### Step 3: Compute Maximum Number of Parameters

```python
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

This estimates the maximum number of parameters that can fit into the memory of **8 H100 GPUs**.

---

### Caveat

This calculation is an **upper bound** because it assumes that all GPU memory is used exclusively for storing model parameters.

In practice, additional memory is required for:

- Activations
- Attention caches
- Temporary tensors
- CUDA workspaces
- Communication buffers
- Mixed-precision bookkeeping

Therefore, the actual trainable model size will be smaller and depends on factors such as:

- Batch size
- Sequence length
- Activation checkpointing
- Tensor/Data/Pipeline parallelism
- Memory optimization techniques such as **ZeRO** or **FSDP**





# Tensors Basics

Tensors are basic building blocks for the storing things
1. data
2. parameters
3. gradients
4. optimizer state
5. activation

Each tensor has a rank, which is the number of dimensions.

x = torch.zeros(4) # rank 1 tensor (vector)

x = torch.zeros(4, 8) # rank 2 tensor (matrix)

x = torch.zeros(4, 8, 2) # rank 3 tensor

In Transformers, will see tensors of rank 4:

B = 32 # Batch size

S = 16 # Sequence length

H = 16 # Number of heads

D = 64 # Hidden dimension per head

x = torch.zeros(B, S, H, D)




def tensors_memory():

Elements of tensors are generally floating point numbers.

## fp32

[[Wikipedia]](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)

![](https://cs336.stanford.edu/lectures/images/fp32.png)

The fp32 data type (also known as float32 or single precision) is the default.

Traditionally, in scientific computing, fp32 is the baseline; you could use double precision (fp64) in some cases.

In deep learning, you can be a lot sloppier.

Let's examine memory usage of these tensors.

Memory is determined by the 
(i) number of values 
(ii) data type of each value.

x = torch.zeros(4, 8)

assert x.dtype == torch.float32 # Default type

assert x.numel() == 4 * 8

assert x.element_size() == 4 # Float is 4 bytes

assert get_memory_usage(x) == 4 * 8 * 4 # 128 bytes

One matrix in the feedforward layer of GPT-3:

assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024 # 2.3 GB
## fp16

[[Wikipedia]](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

![](https://cs336.stanford.edu/lectures/images/fp16.png)

The fp16 data type (also known as float16 or half precision) cuts down the memory.

x = torch.zeros(4, 8, dtype=torch.float16)

assert x.element_size() == 2

However, the dynamic range (especially for small numbers) isn't great.

x = torch.tensor([1e-8], dtype=torch.float16)

assert x == 0 # Underflow!

If this happens when you train, you can get instability.

## bf16

[[Wikipedia]](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)

![](https://cs336.stanford.edu/lectures/images/bf16.png)

Google Brain developed brain floating point (bf16) in 2018 to address this issue.

bf16 uses the same memory as fp16 but has the same dynamic range as fp32!

The only catch is that the resolution is worse, but this matters less for deep learning.

x = torch.tensor([1e-8], dtype=torch.bfloat16)

assert x != 0 # No underflow!

## Mixed precision

Implications on training:

- Training with fp32 works, but requires lots of memory.

- Training with fp16 and even bf16 is risky, and you can get instability.

Solution: mixed precision training  

[[Micikevicius+ 2017]](https://arxiv.org/pdf/1710.03740.pdf)

- Use bf16 for parameters, activations, and gradients

- Use fp32 for optimizer states

Pytorch has an automatic mixed precision (AMP) library.  

[[docs]](https://pytorch.org/docs/stable/amp.html)

Tries to cast things into bf16 when safe (matmuls, not exp).

with torch.amp.autocast("cuda", dtype=torch.bfloat16):

x = torch.zeros(4, 8)

## fp8

In 2022, fp8 was standardized, motivated by machine learning workloads [primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

![](https://cs336.stanford.edu/lectures/var/files/image-df6d7649a3bdb77cfdc38092d8387a99-https_docs_nvidia_com_deeplearning_transformer-engine_user-guide__images_fp8_formats_png)

H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).

Reference:  

[[Micikevicius+ 2022]](https://arxiv.org/pdf/2209.05433.pdf)

H100s support two variants of FP8: E4M3 (range [-448, 448]) and E5M2 ([-57344, 57344]).

Reference:  

[[Micikevicius+ 2022]](https://arxiv.org/pdf/2209.05433.pdf)

## fp4

In 2025, NVIDIA developed [nvfp4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

Only 4 bits per value!

Values: -6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6

Use a separate scale factor per block, so actually get more dynamic range (but just can't vary freely from neighbors).

Nemotron 3 Super was trained in NVFP4  

[[Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning]](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

Some of this is done in NVIDIA libraries outside of user control.

def tensors_on_gpus():

By default, tensors are stored in CPU memory.

x = torch.zeros(32, 32)

assert x.device == torch.device("cpu")

However, what about GPUs?

![](https://cs336.stanford.edu/lectures/images/cpu-gpu.png)

device = cuda_if_available()

In order to take advantage of the massive parallelism of GPUs, we need to move them to GPU memory.

x = x.to(device)

Or create the tensor directly on the GPU:

with torch.device(device):

x = torch.zeros(32, 32)

assert x.device == device




# Eniops Operation for the matrix multiplications

def einops_einsum():

Einsum is generalized matrix multiplication with good bookkeeping.

x = torch.ones(3, 4) # seq1 hidden

y = torch.ones(4, 3) # hidden seq2

# Old way

z = x @ y # seq1 seq2

# New (einops) way

z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")

Let's try a more complex example...

x = torch.ones(2, 3, 4) # batch seq1 hidden

y = torch.ones(2, 3, 4) # batch seq2 hidden

# Old way

z = x @ y.transpose(-2, -1) # batch seq1 seq2

# New (einops) way

z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")

Dimensions that are not named in the output are summed over.

# Or can use `...` to represent broadcasting over any number of dimensions

z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")

def einops_reduce():

You can reduce a single tensor via some operation (e.g., sum, mean, max, min).

x = torch.ones(2, 3, 4) # batch seq hidden

# Old way

y = x.sum(dim=-1)

# New (einops) way

y = reduce(x, "... hidden -> ...", "sum")

def einops_rearrange():

Sometimes, a dimension represents two dimensions

...and you want to operate on one of them.

x = torch.ones(3, 8) # seq total_hidden

...where `total_hidden` is a flattened representation of `heads * hidden1`

w = torch.ones(4, 4) # hidden1 hidden2

# Break up `total_hidden` into two dimensions (`heads` and `hidden1`

x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)

# Perform the transformation by `w`

x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")

# Combine `heads` and `hidden2` back together

x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")




A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).

Two terribly confusing acronyms (pronounced the same!):

- FLOPs: floating-point operations (measure of computation done)

- FLOP/s: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.




## Linear model

if torch.cuda.is_available():

B = 16384 # Number of points

D = 32768 # Dimension of each point

K = 8192 # Number of outputs

else:

B = 1024

D = 256

K = 64

x = torch.ones(B, D, device=cuda_if_available())

w = torch.randn(D, K, device=cuda_if_available())

y = x @ w

How many FLOPs is this matmul?

We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.

actual_num_flops = 2 * B * D * K

We can also time this operation to see how long it takes.

actual_time = benchmark(lambda: x @ w)

The actual FLOP/s of this operation:

actual_flop_per_sec = actual_num_flops / actual_time

Each GPU has a specification sheet that provides the peak performance.

- Example: 

[[H100 spec]](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)

Note that the FLOP/s depends heavily on the data type!

promised_flop_per_sec = get_promised_flop_per_sec(x.dtype)









# Memory-Bound vs Compute-Bound: A Story of How Fast a GPU Can Execute Your Code

Imagine you have an NVIDIA H100 GPU and you want to know:

> **How long will my operation take?**

At first glance, you might think:

> "Just divide the number of FLOPs by the GPU's FLOPS."

Unfortunately, it isn't that simple.

The execution time of any GPU workload depends on **two resources**:

1. **Compute Throughput (FLOPs/second)** – How fast the GPU can perform arithmetic.
2. **Memory Bandwidth (Bytes/second)** – How fast data can be moved between memory and the GPU.

For an H100,

```python
h100_flop_per_sec = 1979e12 / 2     # Effective BF16 throughput
h100_bytes_per_sec = 3.35e12        # Memory bandwidth
```

Every operation needs **both** computation and memory transfers.

Therefore,

$$
\text{Execution Time}
=
\max(\text{Computation Time},\;
\text{Communication Time})
$$

where

$$
\text{Computation Time}
=
\frac{\text{FLOPs}}
{\text{GPU FLOPs/sec}}
$$

and

$$
\text{Communication Time}
=
\frac{\text{Bytes Moved}}
{\text{Memory Bandwidth}}
$$

The slower of the two determines the total execution time.

---

# Story 1: ReLU

Suppose we execute

```python
y = ReLU(x)
```

What happens?

The GPU performs three steps:

```
Read x from memory
↓

Compute max(x, 0)

↓

Write y back to memory
```

Suppose there are **n** elements.

Memory movement:

- Read **n** BF16 numbers
- Write **n** BF16 numbers

Each BF16 value occupies **2 bytes**.

Therefore,

```python
bytes = (2*n) + (2*n)
```

How much computation?

Only one comparison per element.

```python
flops = n
```

Arithmetic intensity becomes

$$
\frac{n}{4n}
=
\frac14
$$

Very little computation.

Lots of memory movement.

The GPU spends most of its time waiting for memory.

**Conclusion**

✅ ReLU is **memory-bound**.

---

# Story 2: GeLU

Now consider

```python
y = GELU(x)
```

Unlike ReLU, GeLU performs a complicated mathematical expression involving

- multiplication
- addition
- cubic term
- tanh approximation

Approximately

```python
flops = 20*n
```

Memory movement is unchanged.

```python
bytes = 4*n
```

Arithmetic intensity becomes

$$
\frac{20n}{4n}
=
5
$$

Much higher than ReLU.

However...

The H100 accelerator is capable of much more work per byte than **5 FLOPs/byte**.

So the GPU is **still waiting for memory**.

**Conclusion**

✅ GeLU is also **memory-bound**.

Interestingly,

Although GeLU performs far more computation than ReLU, it is often **not significantly slower**, because memory—not computation—is the bottleneck.

---

# Story 3: Dot Product

Now compute

```python
y = x @ w
```

The GPU

```
Read vector x

↓

Read vector w

↓

Multiply corresponding elements

↓

Accumulate

↓

Write one output
```

Memory:

```python
bytes = (2*n) + (2*n)
```

Computation:

```python
flops = 2*n - 1
```

Arithmetic intensity

$$
\approx
\frac12
$$

Still very low.

The GPU spends more time moving vectors than multiplying them.

**Conclusion**

✅ Dot products are memory-bound.

---

# Story 4: Matrix-Vector Multiplication

Now consider

```python
y = W x
```

where

```
W : n × n
x : n
```

Each output requires a dot product.

There are **n** outputs.

Memory

```python
bytes =
2n
+
2n²
+
2n
```

Computation

```python
flops = n(2n-1)
```

Arithmetic intensity

$$
\approx
1
$$

This is better than a dot product.

But still not enough.

The GPU continues waiting for memory.

**Conclusion**

✅ Matrix-vector multiplication is still memory-bound.

---

# Story 5: Matrix Multiplication

Finally,

```python
Y = XW
```

where

```
X : n × n

W : n × n
```

Now something changes.

Each element loaded into memory is reused **many times**.

Instead of reading a value once and discarding it,

the GPU repeatedly multiplies it with many other values.

Memory

```python
bytes =
2n²
+
2n²
+
2n²
```

Computation

```python
flops =
n²(2n-1)
```

Arithmetic intensity becomes

$$
\approx
\frac{n}{3}
$$

When

```
n = 1024
```

Arithmetic intensity is roughly

```
341 FLOPs per byte
```

Now the GPU has far more computation than memory transfers.

The arithmetic units remain busy.

Memory is no longer the bottleneck.

**Conclusion**

✅ Matrix multiplication is **compute-bound**.

---

# The Key Idea: Arithmetic Intensity

We define

$$
\boxed{
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes Moved}}
}
$$

This measures

> **How much computation do we perform for every byte transferred?**

Higher is better.

---

# Accelerator Intensity

Every GPU also has a characteristic number:

$$
\boxed{
\text{Accelerator Intensity}
=
\frac{\text{Peak FLOPs/sec}}
{\text{Memory Bandwidth}}
}
$$

For an H100,

```python
h100_accelerator_intensity =
h100_flop_per_sec /
h100_bytes_per_sec
```

This represents the amount of computation the GPU **expects** to perform per byte moved in order to keep all compute units fully occupied.

---

# Comparing the Two

Now compare

Arithmetic Intensity

vs

Accelerator Intensity.

If

$$
\text{Arithmetic Intensity}
<
\text{Accelerator Intensity}
$$

the GPU cannot keep its arithmetic units busy.

It waits for memory.

✅ Memory-bound.

If

$$
\text{Arithmetic Intensity}
>
\text{Accelerator Intensity}
$$

there is enough computation to fully utilize the GPU.

Memory is no longer the bottleneck.

✅ Compute-bound.

---

# Roofline Model
We can visualize the relationship between arithmetic intensity and performance using roofline plots.

![](https://cs336.stanford.edu/lectures/var/files/image-42d32b9c87939fe9a4b0a268d6d02ea7-https_jax-ml_github_io_scaling-book_assets_img_roofline-improved-1400_webp)



### Left Side

Performance increases as arithmetic intensity increases.

This is the **memory-bound region**.

More computation per byte improves performance.

---

### Right Side

Eventually the graph reaches a flat roof.

The GPU has reached its maximum computational capability.

Adding more computation no longer increases performance.

This is the **compute-bound region**.

---

The point where the slope changes is called the **accelerator intensity**.

It marks the transition from

```
Memory-bound
        ↓
Compute-bound
```

---

# Why Transformers Train Efficiently

Transformer training consists mostly of

- Linear layers
- Attention matrix multiplications
- MLP matrix multiplications

These are **large matrix multiplications**.

Large matrix multiplications have very high arithmetic intensity.

Therefore,

✅ Transformer **training** is mostly **compute-bound**.

The GPU spends most of its time performing useful arithmetic.

---

# Why Inference is Different

During inference,

we usually multiply

```
Large Weight Matrix

×

One Token
```

This is a **matrix-vector multiplication**, not a matrix-matrix multiplication.

Matrix-vector multiplication has low arithmetic intensity.

Therefore,

Transformer **inference** is mostly **memory-bound**.

This explains why increasing FLOPS often has little effect on inference latency—the bottleneck is memory bandwidth.

---

# Connecting Everything Back to MFU

Recall that

$$
\text{MFU}
=
\frac{\text{Actual FLOPs}}
{\text{Peak FLOPs}}
$$

Using the Roofline Model, we can estimate it as

$$
\boxed{
\text{MFU}
=
\min\left(
1,\;
\frac{\text{Arithmetic Intensity}}
{\text{Accelerator Intensity}}
\right)
}
$$

This equation tells us:

- If arithmetic intensity is **much smaller** than the accelerator intensity, MFU is low because the GPU spends most of its time waiting for memory.
- As arithmetic intensity increases, MFU increases because more of the GPU's compute capability is utilized.
- Once arithmetic intensity exceeds the accelerator intensity, the workload becomes compute-bound and MFU approaches **1 (100%)**, meaning the GPU is operating near its maximum computational capacity.




# Understanding the FLOPs Required for Gradient Computation

So far, we have focused on the **forward pass**, where the model takes an input and produces an output.

However, during **training**, this is only half of the work.

Once the loss is computed, we must calculate gradients for every trainable parameter so that the optimizer can update the model weights. This process is called the **backward pass** or **backpropagation**.

The goal of this example is to understand **where the computation in the backward pass comes from** and why training is much more expensive than inference.

---

# A Simple Two-Layer Neural Network

Consider a simple network consisting of two linear layers.

```python
B = 1024      # Batch size
D = 256       # Feature dimension
```

Create an input batch:

```python
x = torch.ones(B, D)
```

Create two trainable weight matrices:

```python
w1 = torch.randn(D, D, requires_grad=True)
w2 = torch.randn(D, D, requires_grad=True)
```

The network can be visualized as

```
Input
   x
   │
   ▼
Linear Layer (W1)
   │
   ▼
Hidden Representation (h1)
   │
   ▼
Linear Layer (W2)
   │
   ▼
Output (h2)
```

---

# Forward Pass

The first layer computes

```python
h1 = x @ w1
```

or equivalently

```python
h1 = einsum(
    x,
    w1,
    "batch in, in out -> batch out"
)
```

Mathematically,

$$
h_1 = XW_1
$$

---

The second layer computes

```python
h2 = h1 @ w2
```

or

```python
h2 = einsum(
    h1,
    w2,
    "batch in, in out -> batch out"
)
```

Mathematically,

$$
h_2 = h_1W_2
$$

---

Finally, define a simple loss

```python
loss = (h2.mean() - 0)**2
```

which can be written as

$$
L =
\left(
\operatorname{mean}(h_2)
\right)^2
$$

The actual choice of loss is not important here—it simply provides a scalar value from which gradients can be computed.

---

# Backward Pass

Once the loss is computed,

```python
loss.backward()
```

PyTorch automatically computes gradients for every trainable parameter.

Internally, this applies the **chain rule** repeatedly to propagate gradients from the loss back to the weights.

---

# What Gradients Are Computed?

The backward pass computes

```
Loss
 │
 ▼
∂L/∂h₂
 │
 ▼
∂L/∂W₂
 │
 ▼
∂L/∂h₁
 │
 ▼
∂L/∂W₁
```

More explicitly,

1. Gradient of the loss with respect to the output:

$$
\frac{\partial L}{\partial h_2}
$$

2. Gradient with respect to the second layer weights:

$$
\frac{\partial L}{\partial W_2}
$$

3. Gradient propagated back to the hidden representation:

$$
\frac{\partial L}{\partial h_1}
$$

4. Gradient with respect to the first layer weights:

$$
\frac{\partial L}{\partial W_1}
$$

---

# Why Call `retain_grad()`?

Normally, PyTorch stores gradients only for **leaf tensors** (such as model parameters).

Intermediate tensors like `h1` and `h2` do not retain their gradients after backpropagation.

For debugging purposes,

```python
h1.retain_grad()
h2.retain_grad()
```

tells PyTorch to also store

```python
h1.grad
```

and

```python
h2.grad
```

after calling

```python
loss.backward()
```

This is useful for inspecting how gradients flow through the network.

---

# Where Do the FLOPs Come From?

During training, computation occurs in two phases:

### 1. Forward Pass

```
XW₁

↓

h₁W₂

↓

Loss
```

This performs **two matrix multiplications**.

---

### 2. Backward Pass

For each linear layer, backpropagation performs **two additional matrix multiplications**:

- Compute the gradient with respect to the weights.
- Compute the gradient with respect to the input activations.

Thus, for every linear layer:

```
Forward:
    XW

Backward:
    dW
    dX
```

This results in approximately **three matrix multiplications per layer**.

---

# Why Is Training More Expensive Than Inference?

Inference performs only the forward pass:

```
Input
↓

Forward

↓

Prediction
```

Training performs

```
Forward

↓

Loss

↓

Backward

↓

Optimizer Update
```

As a result,

- **Inference** requires approximately **1×** the forward computation.
- **Training** requires approximately **3×** the forward computation.

This is the origin of the commonly used approximation:

$$
\boxed{
\text{Training FLOPs}
\approx
6 \times
(\text{Parameters})
\times
(\text{Training Tokens})
}
$$

The factor of **6** comes from:

- ~2 FLOPs per parameter during the forward pass (multiply and add),
- ~4 FLOPs per parameter during the backward pass (gradient computation for weights and activations).

Thus, the backward pass accounts for roughly **twice the computation** of the forward pass, making training significantly more computationally expensive than inference.





# Deep Networks, Optimizers, Memory, and Training Loop

Now that we understand the cost of computing gradients, let's put everything together and see what happens during **training**.

Training a neural network consists of repeatedly performing four steps:

```
Input Batch
      │
      ▼
Forward Pass
      │
      ▼
Compute Loss
      │
      ▼
Backward Pass (Gradients)
      │
      ▼
Optimizer Update
      │
      ▼
Repeat
```

This process is repeated thousands or even millions of times until the model converges.

---

# Building a Deep Network

Consider a deep neural network with

- Input dimension = **D**
- Output dimension = **D**
- Hidden dimension = **D**
- Number of layers = **L**

Every layer performs exactly the same computation.

```
Input
   │
   ▼
Linear
   │
   ▼
ReLU
   │
   ▼
Linear
   │
   ▼
ReLU
   │
   ▼
...
   │
   ▼
Output
```

Each layer is implemented as a simple **Block**.

```python
class Block(nn.Module):
    def forward(self, x):
        x = x @ self.weight
        x = F.relu(x)
        return x
```

Each block contains only

- a weight matrix
- a ReLU activation

---

# The Linear Layer

Suppose the feature dimension is

```
D = 8
```

The weight matrix has size

$$
W \in \mathbb{R}^{D \times D}
$$

Therefore,

the number of parameters in one layer is

$$
D \times D
$$

If the network has

```
L
```

layers,

the total number of parameters becomes

$$
\boxed{
\text{Parameters}
=
D^2L
}
$$

which is exactly what the code verifies.

```python
num_parameters = D * D * L
```

---

# Forward Pass Through the Network

Assume

```python
B = 4
```

samples in a batch.

The input tensor has shape

```
(B,D)
```

Each layer computes

$$
X_{i+1}
=
\text{ReLU}(X_iW_i)
$$

The output of one layer becomes the input to the next.

```
X

↓

Layer 1

↓

Layer 2

↓

...

↓

Layer L
```

---

# Optimizers

After the gradients have been computed, we need to update the parameters.

Different optimizers maintain different amounts of information about previous gradients.

The simplest optimizer is **Stochastic Gradient Descent (SGD)**.

---

## SGD

Parameter update

$$
w
=
w
-
\eta g
$$

where

- \(g\) is the gradient
- \(\eta\) is the learning rate

SGD only remembers the current gradient.

---

## Momentum

Momentum improves SGD by remembering previous gradients.

Instead of using

```
Current Gradient
```

it uses

```
Current Gradient

+

Previous Moving Average
```

This makes optimization smoother and faster.

```
Momentum

=

SGD

+

Exponential Average of Gradients
```

---

## AdaGrad

AdaGrad adapts the learning rate for every parameter.

Instead of remembering gradients,

it remembers

```
Gradient²
```

over the entire training history.

The accumulated squared gradients are

$$
G_t
=
\sum_{i=1}^{t}
g_i^2
$$

The update becomes

$$
w
=
w
-
\eta
\frac{g}
{\sqrt{G_t+\epsilon}}
$$

Large gradients lead to smaller future updates.

Small gradients receive relatively larger updates.

---

## RMSProp

AdaGrad has one drawback.

The accumulated gradients continually grow,

making learning rates eventually become very small.

RMSProp fixes this by replacing the cumulative sum with an exponential moving average.

```
RMSProp

=

AdaGrad

+

Exponential Averaging
```

---

## Adam

Adam combines the best ideas from Momentum and RMSProp.

It stores

- First moment (momentum)
- Second moment (variance)

```
Adam

=

Momentum

+

RMSProp
```

Today, Adam and AdamW are the most commonly used optimizers for training large neural networks.

---

# AdaGrad Implementation

The optimizer stores one additional tensor for every parameter.

```python
g2 += grad²
```

Then the parameter update becomes

```python
p -= lr * grad / sqrt(g2)
```

Thus, AdaGrad needs to remember

```
Running Sum of Gradient²
```

for every parameter.

---

# Memory Usage During Training

Training requires more memory than inference because several tensors must be stored simultaneously.

Suppose the network has

$$
P
=
D^2L
$$

parameters.

---

## Parameters

The model weights themselves occupy memory.

Using BF16,

```
2 bytes / parameter
```

Memory

$$
2P
$$

---

## Gradients

Each parameter also stores its gradient.

Again,

```
2 bytes / parameter
```

Memory

$$
2P
$$

---

## Optimizer State

AdaGrad stores

```
Gradient²
```

using FP32.

```
4 bytes / parameter
```

Memory

$$
4P
$$

Adam stores

```
Momentum

+

Variance
```

Therefore,

```
8 bytes / parameter
```

---

## Activations

During the forward pass,

every layer must store its activations for use during backpropagation.

If

- Batch size = \(B\)
- Hidden dimension = \(D\)
- Layers = \(L\)

then activation memory is approximately

$$
2BDL
$$

because activations are stored in BF16.

---

# Total Memory

Putting everything together,

for AdaGrad,

$$
\boxed{
\text{Total Memory}
=
\text{Parameters}
+
\text{Gradients}
+
\text{Optimizer State}
+
\text{Activations}
}
$$

or

$$
\boxed{
2P
+
2P
+
4P
+
2BDL
}
$$

For Adam,

replace

```
4P
```

with

```
8P
```

because Adam stores two optimizer statistics.

---

# Compute Required for One Training Step

Each layer performs

- Forward pass
- Backward pass
- Gradient computation

Combining these gives approximately

$$
6
\times
(\text{Parameters})
\times
(\text{Batch Size})
$$

Therefore,

```python
flops = 6 * B * num_parameters
```

This is the same approximation used when estimating the cost of training large language models.

---

# The Training Loop

A complete training loop repeatedly executes four steps.

### Step 1 — Load a Batch

```
Dataset

↓

Mini-batch
```

```python
x, y = get_batch()
```

---

### Step 2 — Forward Pass

```
Input

↓

Model

↓

Prediction
```

```python
pred_y = model(x)
```

---

### Step 3 — Compute the Loss

The prediction is compared with the ground truth.

```python
loss = mse(pred_y, y)
```

---

### Step 4 — Backward Pass

Gradients are computed using automatic differentiation.

```python
loss.backward()
```

This computes

```
∂L/∂W
```

for every trainable parameter.

---

### Step 5 — Optimizer Step

The optimizer updates the model weights.

```python
optimizer.step()
```

---

### Step 6 — Clear Gradients

PyTorch accumulates gradients by default.

Therefore,

after every update,

the gradients must be cleared.

```python
optimizer.zero_grad(set_to_none=True)
```

---

The entire training process looks like

```
Load Batch

↓

Forward Pass

↓

Compute Loss

↓

Backward Pass

↓

Optimizer Update

↓

Clear Gradients

↓

Repeat
```

This simple loop is the foundation of training almost every modern deep learning model—from small convolutional networks to billion-parameter Transformers. The main differences in large models are the scale of the network, the amount of memory required, and the distributed training strategies used to efficiently perform these same steps across many GPUs.



# Basic Neural Network vs Transformer

| Feature | Basic Neural Network (MLP) | Transformer |
|---------|----------------------------|-------------|
| **Building Block** | Linear Layer + Activation | Multi-Head Self-Attention + Feed Forward Network (FFN) |
| **Input** | Fixed-size feature vector | Sequence of tokens |
| **Architecture** | Stacked fully connected layers | Stacked Transformer blocks |
| **Information Flow** | Sequential through layers | Self-attention allows every token to interact with every other token |
| **Captures Relationships** | Learns global mapping only | Learns contextual relationships between tokens |
| **Position Awareness** | Not required | Uses Positional Encoding/Embeddings |
| **Main Computation** | Matrix Multiplication | Self-Attention + Matrix Multiplication |
| **Activation Function** | ReLU, GeLU, Sigmoid, Tanh | Mostly GeLU (inside FFN) |
| **Parameter Sharing** | Different weights for every layer | Different weights per layer, attention shares computations across tokens |
| **Parallelization** | Easy | Highly parallel during training |
| **Memory Requirement** | Lower | Higher due to attention matrices and KV cache |
| **Compute Requirement** | Lower | Significantly higher |
| **Time Complexity** | Depends mainly on number of layers | Attention has \(O(n^2)\) complexity with sequence length |
| **Typical Applications** | Classification, Regression, Tabular Data | NLP, Vision, Audio, Multimodal Models, LLMs |
| **Examples** | Feedforward Network, MLP | BERT, GPT, Llama, Gemini, Qwen |

---

# Single Layer Comparison

## Basic Neural Network

```
Input
   │
   ▼
Linear
   │
   ▼
Activation (ReLU / GeLU)
   │
   ▼
Output
```

Mathematically,

$$
y = \sigma(XW + b)
$$

where:

$$
\begin{aligned}
X & : \text{Input} \\
W & : \text{Weight Matrix} \\
b & : \text{Bias} \\
\sigma & : \text{Activation Function}
\end{aligned}
$$

---

## Transformer Block

```
Input
   │
   ▼
Multi-Head Self-Attention
   │
   ▼
Add & LayerNorm
   │
   ▼
Feed Forward Network
   │
   ▼
Add & LayerNorm
   │
   ▼
Output
```

The Feed Forward Network (FFN) itself is just a small MLP:

$$
\text{FFN}(x)
=
W_2(\text{GeLU}(W_1x))
$$

---

# Computational Comparison

| Operation | Basic Neural Network | Transformer |
|-----------|----------------------|-------------|
| Matrix Multiplication | ✅ | ✅ |
| Activation | ✅ | ✅ |
| Self-Attention | ❌ | ✅ |
| Layer Normalization | Optional | ✅ |
| Residual Connections | Rare | ✅ |
| Positional Encoding | ❌ | ✅ |

---

# Complexity Comparison

Suppose

- \(B\) = Batch size
- \(D\) = Hidden dimension
- \(L\) = Number of layers
- \(N\) = Sequence length

| Quantity              | Basic Neural Network | Transformer                                 |
| --------------------- | -------------------- | ------------------------------------------- |
| **Parameters**        | $O(D^2L)$            | $O(D^2L)$ + attention projection parameters |
| **Forward FLOPs**     | $O(BD^2L)$           | $O(BND^2L + BN^2D)$                         |
| **Training FLOPs**    | $\approx 6BD^2L$     | $\approx 6BND^2L + O(BN^2D)$                |
| **Activation Memory** | $O(BDL)$             | $O(BNL)$ + attention activations            |
|                       |                      |                                             |

---

# Key Difference

A Basic Neural Network processes each input **independently**.

```
x₁ → y₁
x₂ → y₂
x₃ → y₃
```

There is **no interaction** between different inputs.

A Transformer processes an entire sequence **jointly**.

```
Token₁ ←→ Token₂ ←→ Token₃ ←→ ... ←→ Tokenₙ
```

Each token can attend to every other token using **self-attention**, allowing the model to capture long-range dependencies and contextual information.

---

# Summary

| Basic Neural Network | Transformer |
|----------------------|-------------|
| Simple architecture | Complex architecture |
| Uses only Linear + Activation | Uses Self-Attention + FFN |
| Lower memory usage | Higher memory usage |
| Lower computational cost | Higher computational cost |
| Suitable for tabular and structured data | Designed for sequential and multimodal data |
| No notion of context | Learns contextual relationships between tokens |
| Scales moderately | Scales to billions of parameters and trillions of tokens (e.g., GPT, Llama, Gemini) |