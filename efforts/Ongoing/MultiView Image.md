
---

Day: 24 March


1. generated the layout from the diffusion model.
2. lora training for better composition of different lightning


Goals:
i have 30 reference objects and i wanted to lay them properly in the canvas to make a cinematic shot in the place.

The diffusion model
takes reference images of random view, and the layout is generate by the diffsuon model using maps, 


Plan:
1. layout to be generated from the pretrained diffusion model between 1.0 to 0.6
2. once the layout is generated, u will have to make a black canvas and then place the objects in the layouts of the reference objects,
3. og_latents + canvas latents to be further pushed, so that the structure comes the og_latents(guided by prompt) and the textures/shapes should come from the canvas latents(group of reference objects).


Available things as of now
1. dataset: layout bboxes, canvas generator script( support for the multiview of objects)




## What's Actually Happening

```
t=1.0 → 0.7   : FLUX runs freely, prompt drives structure (og_latents)
t=0.7 → 0.0   : LoRA takes over, pulls texture FROM canvas INTO og_latents
```

The og_latent already has the scene structure. The canvas has the object textures. The LoRA's only job is **texture transfer** in the second half of denoising.
## Training Objective

At every training step, you sample **only from t=0.7 to t=0.0** (mask out the high-noise regime entirely).

Your triplet:

- `og_latent` — noisy image latent at sampled t (structure source)
- `canvas_latent` — encoded canvas composite (texture source, clean, no noise added)
- `target` — the real scene photo (what the blend should look like)

The LoRA sees the og_latent being denoised, but at each step it **cross-attends to canvas_latent** to pull texture. It never denoises the canvas — canvas is a fixed reference throughout.

---


Day 25 March:

Layout LLM:


**Experiment 1:**

I have worked with lora, doing monkey patching, where i tried to combine multiple repos using claude code, but failed in giving the output, irrespective of that, i didnt log the images where, i was into seeing the outputs alone from the tensorboard, but the results were bad.

The repos i used are:
1. eye for an eye - for matching the key features in the diffusion space.
2. easycontrol - for lora training .


The entire claudecode thread for this;
https://claude.ai/chat/b0dff679-23b1-4d86-a7e3-c6b3958710cf



**Experiment 2:**

I realized that i am overkilling the process, so i have decided to stop doing patching and start from a simple tone. so i went with easycontrol, the reason is for masked based attention, so that u can control it.


***This day i have got the lora working to an extent, i see the results are doing good enough.***

you can find the results here:
https://docs.google.com/presentation/d/1i3xkGkh2CCJpCVU_XzKME24apHBvEOsq-dLquj1plpY/edit?usp=sharing

---

