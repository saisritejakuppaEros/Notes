

March 23:
the main things for building genie from the sources fo the google are
https://x.com/seti_park/status/2028044161199579557?s=46

### 1. **GameNGen (Google Research)**

- “Can a neural network _replace a game engine_?”
- Answer: **Yes (for a single world like DOOM)**

### 2. **Genie (DeepMind)**

- “Can a model _learn how the world works_ just from videos?”
- Answer: **Yes (but not real-time, not stable)**

Genie Paper: https://arxiv.org/html/2402.15391v1 , [drivelink](https://drive.google.com/file/d/1kE_T3VjfL74mPf3PTCkQOJM5H5Og2E8p/view?usp=sharing)
GameNGen: https://arxiv.org/pdf/2408.14837, [drivelink](https://drive.google.com/file/d/1DTBk8g5e4V8Mq1FRb02n0RrOTP_NvF9e/view?usp=sharing)

List of sources for the implementations of the world models:

| Repo                                                                                          | Type           | Focus                           | Compute    |
| --------------------------------------------------------------------------------------------- | -------------- | ------------------------------- | ---------- |
| `insait-institute/GenieRedux` [github](https://github.com/insait-institute/genieredux)​       | Research-grade | Multi-env world model + dataset | Multi-GPU  |
| `myscience/open-genie` [github](https://github.com/myscience/open-genie)​                     | Clean reimpl   | Genie v1 architecture           | Single GPU |
| `lukehollis/genie-bottle` [github](https://github.com/lukehollis/genie-bottle)​               | Educational    | Single-machine CoinRun          | Single GPU |
| `kimbring2/Genie_Implementation` [github](https://github.com/kimbring2/Genie_Implementation)​ | Case study     | Component-by-component          | Single GPU |
| `AlmondGod/tinyworlds` [github](https://github.com/AlmondGod/tinyworlds)​                     | Minimal        | Autoregressive baseline         | Minimal    |
| `arnaudstiegler/gameNgen-repro` [github](https://github.com/arnaudstiegler/gameNgen-repro)​   | Reproduction   | DOOM + diffusion                | TPU/GPU    |
| `Masao-Taketani/GameNGen` [github](https://github.com/Masao-Taketani/GameNGen)​               | Unofficial     | GameNGen full pipeline          | GPU        |

| Resource                                                                                                                                                                                    | What You Get                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [arXiv 2402.15391](https://arxiv.org/abs/2402.15391) [arxiv](https://arxiv.org/abs/2402.15391)​                                                                                             | Original Genie 1 paper (ICML 2024 Best Paper)                         |
| [ICML proceedings](https://proceedings.mlr.press/v235/bruce24a.html) [proceedings.mlr](https://proceedings.mlr.press/v235/bruce24a.html)​                                                   | Full paper with scaling experiment details                            |
| [Genie 2 DeepMind blog](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) [deepmind](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)​ | Official Genie 2 architectural description                            |
| [SSRN: Mathematics of Genie 2](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5045438) [papers.ssrn](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5045438)​                     | Deep math — causal attention, training objectives, scalability proofs |
| [GenieRedux paper (arXiv 2409.06445)](https://arxiv.org/html/2409.06445v2) [arxiv](https://arxiv.org/html/2409.06445v2)​                                                                    | Open reimplementation with training details you can run               |
| [gist summary](https://gist.github.com/thehunmonkgroup/fb64746d70b2dcebd6b67c18e6507119) [gist.github](https://gist.github.com/thehunmonkgroup/fb64746d70b2dcebd6b67c18e6507119)​           | Readable mechanics walkthrough                                        |

Looking at the survey on a large scale:
https://arxiv.org/pdf/2411.14499

what does world model mean?
1. does it knows about the current knowledge?
2. should it be able to predict the next event what happens?

March 24:

Notes on GameNGen : [[GameNGen]]
Notes on Genie: [[Genie]]
