# Multi-View Image Generation

**Date:** 24 March 2026

## Overview

1. Generated the layout using a diffusion model.
2. Performed LoRA training to improve the composition of different lighting conditions.

## Goals

- I have 30 reference objects and want to arrange them properly on the canvas to create a cinematic shot.

## Diffusion Model

The diffusion model takes random‑view reference images and generates a layout using maps.

## Plan

1. Generate a layout from a pretrained diffusion model with a guidance scale between **1.0** and **0.6**.
2. Once the layout is generated, create a black canvas and place the reference objects according to the layout.
3. Combine the original latents (`og_latents`) with the canvas latents. The structure comes from `og_latents` (guided by the prompt) while textures and shapes are supplied by the canvas latents (the group of reference objects).

## Available Resources

- **Dataset:** Layout bounding boxes.
- **Canvas generator script:** Supports multi‑view objects.

---

## What’s Actually Happening

```
 t = 1.0 → 0.7 : FLUX runs freely – prompt drives structure (og_latents)
 t = 0.7 → 0.0 : LoRA takes over – pulls texture FROM canvas INTO og_latents
```

The `og_latent` already contains the scene’s structure. The canvas holds the object textures. The LoRA’s sole job is **texture transfer** during the second half of denoising.

## Training Objective

At every training step we **sample only from t = 0.7 to t = 0.0**, effectively masking out the high‑noise regime.

### Triplet

- **`og_latent`** – Noisy image latent at the sampled timestep (structure source).
- **`canvas_latent`** – Encoded canvas composite (texture source, clean, no added noise).
- **`target`** – The real scene photo (desired output).

The LoRA sees the `og_latent` being denoised, but at each step it **cross‑attends to `canvas_latent`** to pull texture. The canvas itself is never denoised; it remains a fixed reference throughout.

---

## Day 25 March – Experiments

### Layout LLM

#### Experiment 1

I previously worked with LoRA, performing monkey‑patching to combine multiple repositories using Claude Code, but the output was never logged—only tensorboard visualisations were inspected, and the results were unsatisfactory.

**Repositories used:**
1. *eye‑for‑an‑eye* – matches key features in diffusion space.
2. *easycontrol* – LoRA training framework.

Full Claude Code thread: <https://claude.ai/chat/b0dff679-23b1-4d86-a7e3-c6b3958710cf>

#### Experiment 2

I realized the previous approach was overly complex, so I switched to a simpler setup using **easycontrol** for mask‑based attention, which provides finer control.

> *“This day I got the LoRA working to an extent; the results are good enough.”*

Results can be viewed here: <https://docs.google.com/presentation/d/1i3xkGkh2CCJpCVU_XzKME24apHBvEOsq-dLquj1plpY/edit?usp=sharing>

---
