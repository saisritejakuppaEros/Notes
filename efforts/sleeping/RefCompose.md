
# Multi-View Image Generation
<a id="24-march-2026"></a>
## 24 March 2026

### Overview

1. Generated the layout using a diffusion model.
2. Performed LoRA training to improve the composition of different lighting conditions.

### Goals

- I have 30 reference objects and want to arrange them properly on the canvas to create a cinematic shot.

### Diffusion Model

The diffusion model takes random‑view reference images and generates a layout using maps.

### Plan

1. Generate a layout from a pretrained diffusion model with a guidance scale between **1.0** and **0.6**.
2. Once the layout is generated, create a black canvas and place the reference objects according to the layout.
3. Combine the original latents (`og_latents`) with the canvas latents. The structure comes from `og_latents` (guided by the prompt) while textures and shapes are supplied by the canvas latents (the group of reference objects).

### Available Resources

- **Dataset:** Layout bounding boxes.
- **Canvas generator script:** Supports multi‑view objects.

---

### What’s Actually Happening

```
 t = 1.0 → 0.7 : FLUX runs freely – prompt drives structure (og_latents)
 t = 0.7 → 0.0 : LoRA takes over – pulls texture FROM canvas INTO og_latents
```

The `og_latent` already contains the scene’s structure. The canvas holds the object textures. The LoRA’s sole job is **texture transfer** during the second half of denoising.

### Training Objective

At every training step we **sample only from t = 0.7 to t = 0.0**, effectively masking out the high‑noise regime.

#### Triplet

- **`og_latent`** – Noisy image latent at the sampled timestep (structure source).
- **`canvas_latent`** – Encoded canvas composite (texture source, clean, no added noise).
- **`target`** – The real scene photo (desired output).

The LoRA sees the `og_latent` being denoised, but at each step it **cross‑attends to `canvas_latent`** to pull texture. The canvas itself is never denoised; it remains a fixed reference throughout.

---

## 25 March 2026

### Experiments

### Layout LLM

#### Experiment 1

I previously worked with LoRA, performing monkey‑patching to combine multiple repositories using Claude Code, but the output was never logged—only tensorboard visualisations were inspected, and the results were unsatisfactory.

**Repositories used:**
1. *eye‑for‑an‑eye* – matches key features in diffusion space.
2. *easycontrol* – LoRA training framework.

Full Claude Code thread: <https://claude.ai/chat/b0dff679-23b1-4d86-a7e3-c6b3958710cf>

#### Experiment 2

I realized the previous approach was overly complex, so I switched to a simpler setup using **easycontrol** for mask‑based attention, which provides finer control.

> *"This day I got the LoRA working to an extent; the results are good enough."*

Results can be viewed here: <https://docs.google.com/presentation/d/1i3xkGkh2CCJpCVU_XzKME24apHBvEOsq-dLquj1plpY/edit?usp=sharing>


<a id="26-march-2026"></a>
## 26 March 2026

### Experiment 3 – Augmented Layout
since the original layout is working, i am proceeding with the augmented layout, which means i am distoring the original image with some information, like segmentation, to the core object and then give it for training.

the goal of the model is to learn to map from one perspective to another one. What i mean is i have an object of different pose in one canvas, but the original image is of different shape as well.

The goal is to transfere the texture and shape from the canvas to the original latent. The canvas has the references alone in that.

I understand that the qwen image model is doing multi view, but its overkill and its not doing that good when it comes to that, 

pivoting to use sam3d.

Setting up sam3d is tought, due to dependecy issues, so i have migrated to using hunyun2.1 for generating multiple view of the same object.

i have run hunyain 2.1, but realized the meshes are not looking that great to be said.
Humans are not doing that good in that.

March 2:

working on the synthetic dataset generation for the evaluation. This is because, there is not a global testset for this, so for this we are doing a basic testing set so that we can benchmark 3 different things in the first place.


## July 15

Update:
Submitted Paper to the BMVC and got rejected because of the main things

The reviews for the rejections are kept over here:
https://docs.google.com/document/d/1hhy_V9yZvXOeUHM-jRwu--1eBBx_TJe1/edit?rtpof=true&sd=true


Major flaws include
1. dataset preparation
2. limit in novetly
3. depth based rotation not working out in the first place.
4. Cultural aspect is gone and not maintained that well.

Model is praised for the approach 
1. bringing all the reference images into the canvas so that you are doing token compression

Plan to submit for some workshop paper if things go well. AI for Visual Arts is the one to aim for over now.

Immediate things to do for the and trouble shoot things so that i can make the system work

1. Flux 2.0 model can generate things the right way, its about making it more asthetic and compressed in the right way.
2. Use depth information for the sake of overlapping images not randomly
3. Use dift to translate or rotate in the canvas itself so that it can deal things in the first place along with the depth map

I need to compare against canvas to image model as well for this approach.


Since the depth is not really doing the job, I am using the sift ops to get the object in the place and the depth being given as well to get the structure.


https://colab.research.google.com/github/zszazi/MSD/blob/master/SIFT_feature_matching.ipynb

I have tried using sift algorithm the image key point matching is so bad using this.

LightGlue is doing better when you do the map between the images in the crop and the reference image decently.

the further goal is to make the homography to happen prior so that it can help the model much better and then get the results for the inference better. 

# July 16

I see that the model is now capable of generating pictures from the reference images, but i need the model to pick up the cues from the reference images rather than the prompt. 

Prompt engineering work is to be done so that it can pickup the cues from the reference images over the prompt

1. Parth dataset to be fixed with valid prompts.
2. Teja dataset to be fixed with valid prompts for both the landmarks and synthetic dataset. Both of them.

Dift based inference so that to check if the previous models are capable enough to take care of this?


# July 17

I have realized that the model is working good on a black canvas 

from the depth orientation is got and the from the canvas the textures are retrived.

The dataset is also prepared and pushed, but now the past vm is not working so i am setting things up in the new vm

Things to do:
Paper writing part for the new dataset pipeline and the inference process so that we can opensource the codebase and the make the repo setup for the submission of the paper.

Inference on a different set of images needs to be done so that we are able to get the depth the right way for making things possible.

Get all the reviews and make the paper to be updated first, prepare a seperate benchmark to prove things so that you are having multi reference images to be done.

results to bring closer to the original metrics for that paper that evals the things so that we are closer to the original paper for the progress.

Make the timeline for doing things and the write up the paper for the updates.


Dataset Preperation re writing and then the benchmark to be standard,

make a module running with the samples given a decent way to setup so that the other can run based on the reference images given.

compare with qwen image edit so that we can show the sticker effect aganist the qwen image edit, where realism is more in the paper we have with us.

rewrite the paper based on the insights we got from the reviews.


This is the benchmarking paper:
http://arxiv.org/html/2606.15867v1

Buildup the things based on this so that we can tackle the things further.

Somehow show that u can control the orinetation of the object because we are capable of doing this based on the depth.



July 18 and 19

The picture for the papers  are in progress and has created all the necessary pictures.

