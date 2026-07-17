


## CS336 Lecture 3 — Topic Flow (for note-making)

### 1. Intro / Framing

- Outline and goals (recap modern transformer, common elements, common variations)
- Course theme: hands-on > learning from others' experience

### 2. Starting Point: Baseline Architectures

- The 'original' transformer (review of choices: sine PE, ReLU FFN, post-norm LayerNorm)
- What you implemented — simple modern variant (pre-norm, RoPE, SwiGLU, no bias)
- Why so many architectures exist (survey of 19+ dense models, 2024–2026 model zoo)
- "Let's look at the data" — comparison table across models; framing questions (what's common, what varies, what to learn)
- What we'll cover: architecture variations, hyperparameters, stability tricks

### 3. Architecture Variations

#### 3.1 Normalization

- Pre-vs-post norm (concept, data/evidence, explanations: gradient attenuation & spikes)
- New trend: 'double'/non-residual post-norm (Grok, Gemma 2, OLMo 2)
- LayerNorm vs RMSNorm (definitions, notable models)
- Why RMSNorm? (FLOP argument, why it's weak — FLOPs ≠ runtime, data movement matters)
- RMSNorm validation (empirical results)
- Dropping bias terms generally (reasons: memory, optimization stability)
- **Recap: LayerNorm summary**

#### 3.2 Activations

- Zoo of activations overview
- Common activations: ReLU, GeLU, SwiGLU/GeGLU (notable models)
- Gated activations (*GLU) — math derivation
- Gated FF variants: GeGLU, SwiGLU (notable models, dff scaling by 2/3)
- Do gated linear units work? (Shazeer 2020 data, Narang et al 2020 corroboration)
- **Recap: Gating/activations summary**

#### 3.3 Serial vs Parallel Layers

- Concept (can we parallelize attention + MLP?)
- Parallel layers (GPT-J origin, tradeoffs, recent models: Command A, Falcon 2, Command R+)

#### 3.4 **Summary: Architectures** (pre/post-norm, RMSNorm, gating, serial/parallel)

### 4. Position Embeddings

- Overview of variations: sine, absolute, relative embeddings (with notable models)
- RoPE motivation (relative position via inner-product invariance)
- RoPE intuition (rotation analogy)
- RoPE: which rotation to pick (pairing coordinates, 2D rotation, Gemma 4 alt)
- The actual RoPE math (rotation matrix formulation)
- Implementation/code walkthrough for RoPE

### 5. Hyperparameters

- Framing questions (ff size, heads, vocab size, regularization, depth vs width)

#### 5.1 Feedforward Ratio

- Consensus hyperparameter 1: dff = 4·dmodel
- Exception 1: GLU variants (8/3 ratio, table of models)
- Exception 2: T5 (64x multiplier) and follow-ups
- Why this range works (Kaplan+ 2020 basin evidence)
- What we learn from dff hyperparam

#### 5.2 Head Dimensions

- Consensus hyperparameter 2: head_dim × num_heads = dmodel
- Data: num heads/head dim/model dim across models

#### 5.3 Aspect Ratio (depth vs width)

- Data across models (dmodel/nlayer)
- Considerations (deep = harder to parallelize/higher latency)
- Evidence on aspect ratio scaling (Kaplan 2020, Tay 2021)

#### 5.4 Vocabulary Size

- Typical sizes: monolingual vs multilingual/production models

#### 5.5 Regularization

- Dropout/weight decay: arguments against regularization in pretraining
- Data: dropout & weight decay usage across models (older vs newer)
- Why weight decay in LLMs? (Andriushchenko et al 2023 — not for overfitting, interacts with LR schedule)
- **Summary: Hyperparameters** (feedforward, head dim, aspect ratio, regularization)

### 6. Stability Tricks

- Framing: attention to stable training (OLMo comparison curves)
- Where issues arise: softmax instability
- Output softmax stability: z-loss (PaLM, other models)
- Attention softmax stability: QK-norm (origin in vision/multimodal, adopters)
- Logit soft-capping (Gemma, tradeoffs)

### 7. Attention Head Variants

- Overview: GQA/MQA, sparse/sliding window, SSM (teaser for next lecture)

#### 7.1 GQA/MQA

- Compute analysis: full training-time attention (arithmetic intensity)
- Incremental/inference case: KV cache concept
- Incremental arithmetic intensity problem
- MQA: shared K/V idea, memory savings
- GQA: middle ground (grouped queries), mention of MLA (DeepSeek v2)
- Does MQA/GQA hurt performance? (Shazeer 2019, Ainslie 2023 data)

#### 7.2 Sparse / Sliding Window Attention

- Concept: quadratic cost, sparse/structured patterns (Child et al 2019)
- Current standard: interleaving full + local/sliding-window attention (Cohere Command A example)
- Other recent examples: Gemma 4, OLMo 3, Qwen 3.5/Qwen 3 Next

### 8. Final Recap

- Overall summary table across all models
- Major remaining differences: position embeddings, activations, tokenization



This is a lecture by tatsu

Model Archs

Designs of the transformer arch

1. positional embeddings
2. FFN: feed forward neural network layers
3. Norm Type: LayerNorm


layernorm is at the front of the block
rope for the positional encodings over sinusoidal encodings
FF uses swiglu over relu
no bias are set for the linear layers and layernorms 



Common Architecture Variants:
 1. activations , FFN
 2. attention variants
 3. positional embeddings


Hyper parameters that matter
	1.ff_dims in the attention blocks
	2. how many vocal elements

Stablility tricks



High level view of the dominance of the llama based papers and 
qk-norm and hybrid attention papers


Pre vs Post Norm:

norm to exist befor the attention block vs after the it

all the llm use the prenorm, but why?

prenorm is actually helping the model to learn things much better in a stable way so that loss keeps going down.

gradient spikes are lesser during the training because of the prenorm things over then post norm.

further update is to put up the double norms, one before the attention and after the attention blocks as well

grok and gemma 2, olmo 2 does this.




# Layer Norm vs RMS Norm

Layer Norm: formala to be in the tex - gpt 3/2/1 , bloom
rmsnorm: doest subtract mena or add a bia ter m - llama, chincilla , T5

rms norms have seen the gains in the papers


further for the FFN things

FFN(x) = max(0, xW1+b1)w2+b2 this is not used rather we use
FFN(x) = sigmoidn(xW1)*W2

memory and the optimization stability that u similary get in the rms norms usually.




LayerNorm: recap • Basically everyone does non-residual norm (often prenorm) • Intuition – keep the good parts of residual connections • Observations – nicer gradient propagation, fewer spike • Some people add a second norm outside the residual stream • Most people do RMSnorm • In practice, works as well as LayerNorm • But, has fewer parameters to move around, which saves on wallclock time • People more generally drop bias terms since the compute/param tradeoffs are not great


mathematical notations for the Gelu, RelU, swiglu, glu parts.


Now you have positional encodings:
1. sinusoidal emebddings
2. relative embeddings
3. RoPE



Concensus Hyperparameter 1:

FFN(x) = max(0, xW1+b1)W+b2
dff= 4 d(model)
There are two dimensions that are relevant – the feedforward dim (𝑑𝑓𝑓) and model dim (𝑑𝑚𝑜𝑑𝑒𝑙). What should their relationship be?


Remember that GLU variants scale down by 2/3rd. This means most GLU variants have 𝑑𝑓𝑓 = 8 3 𝑑𝑚𝑜𝑑𝑒𝑙. This is mostly what happens. Some notable such examples.

As we have (and will) see, most LMs are have boring, conservative hyperparameters. One exception is T5 [Raffel et al 2020] which has some very bold settings. In particular, for the 11B model, they set 𝑑𝑓𝑓 = 65,536 𝑑𝑚𝑜𝑑𝑒𝑙 = 1024


The ‘default’ choices of 𝑑𝑓𝑓 = 4𝑑𝑚𝑜𝑑𝑒𝑙 and 𝑑𝑓𝑓 = 2.66𝑑𝑚𝑜𝑑𝑒𝑙 have worked well for nearly all modern LLMs. • But T5 does show that even radical choices of 𝑑𝑓𝑓 = 64𝑑𝑚𝑜𝑑𝑒𝑙 can work. This hyperparameter choice isn’t written in stone.


This doesn’t have to be true: we can have head-dimensions > model-dim / num-heads. But most models do follow this guideline


How many heads, whats the model dim? Some examples of this hyperparameter Num heads Head dim Model dim Ratio GPT3 96 128 12288 1 T5 128 128 1024 16 T5 v1.1 64 64 4096 1 LaMDA 128 128 8192 2 PaLM 48 258 18432 1.48 LLaMA2 64 128 8192 1 Qwen 3.5 (27B) 24 256 5120 1.2 Most models have ratios around 1 – notable exceptions by some google models.


Aspect ratios Should my model be deep or wide? How deep and how wide? Most models are surprisingly consistent on this one too! Model 𝒅𝒎𝒐𝒅𝒆𝒍/𝒏𝒍𝒂𝒚𝒆𝒓 BLOOM 205 T5 v1.1 171 PaLM (540B) 156 GPT3/OPT/Mistral/Qwen /OLMo 3 128 LLaMA / LLaMA2 102 Gemma 3 87 Gemma 4 61 T5 (11B) 33


Dropout and weight decay in practice * Most of the times papers just don’t discuss dropout. On open models, this closely matches not doing dropout. This may not be true of closed models. Model Dropout* Weight decay Original transformer 0.1 0 GPT2 0.1 0.1 T5 0.1 0 GPT3 0.1 0.1 T5 v1.1 0 0 PaLM 0 (variable) OPT 0.1 0.1 LLaMA 0 0.1 Qwen 14B 0.1 0.1


The z-loss thing in the Palm paper






what is soft capping in the llm

Grouped Query attention 
Multihead query attention computational costs savers and why is it how is it happneing

KV cache so that u can save the inference time for a better output.


Attention mechanisms
1. sparse transformer strided
2. fixed