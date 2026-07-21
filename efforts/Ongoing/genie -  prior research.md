

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

# March 24:
Notes on GameNGen : [[GameNGen]]
Notes on Genie: [[Genie]]

# March 2:
Medium Blogs: gamegenx, matrixgame1,3


# July 15:
Working on the world model dataset generation pipeline of things.

# July 20:

Priority 1:
https://github.com/ZiYang-xie/WorldGen
https://github.com/SkyworkAI/Matrix-3D
https://github.com/Tencent-Hunyuan/HY-World-2.0

Generating static 3d worlds:
Priority 2:
https://github.com/YujiaHu1109/Flash-Sculptor




Proceeding to go with hy 2.0 since this is latest and has support for exporting to unreal engine unlike others where they generate only videos.

The problem with the videos it can hallucinate and also requires heavy rendering process as well without a memory, so u cant revist the place twice.

Pivoiting to static meshes because in the end of the day, the meshes are to be used by the computer grpahics team, which is the end product goal

from the CG Team perspecitive, the main concerns are
1. getting the meshes that are low poly but has good apperance.
2. getting meshes that can be imported into the 3d unity or unreal engine.
3. Humans style cope up for the avatars in the movie.


Current existing world models has 2 things
1. video based generation used for the gaming purpose.
2. static mesh and then wandering in the meshes

One is build on top of the video based models and the other one is to build a panaromic image and edit the 3d mesh.

The later one is much reliable for the computer graphic engineer so that they can edit things in the 3d space. Which can be exported later to the unity or unreal engine.

The goal further is to do build things looking at multi reference images somehow from a movie set of set of songs, bring in the avatar and the world out of it.


## Future Directions

### 1. Image + Text to 3D World

- **Video Generation Models**
    - Sana World
    - WanX World
    - Lingbot World Model
        
- **Static 3D Asset Generation**
    - Hunyuan World 2.0
    - Matrix3D
- **Video Game World Generation**
    - Matrix Game 3.0
    
We are currently focusing on **Static 3D Asset Generation** because the outputs from video world models and game world generation models cannot be directly exported to **Unity** or **Unreal Engine**.

Video world models often generate hallucinations and inconsistent geometry, making them difficult for the CG team to use in a production pipeline. Similarly, game world generation models prioritize faster inference over visual quality, resulting in lower-quality outputs. Moreover, both approaches are fundamentally powered by video generation models, meaning that achieving production-quality results would require training a dedicated in-house video world model, which is time-consuming and resource-intensive.

To overcome these limitations, we are focusing on generating **high-quality 3D meshes and assets** instead of complete 3D worlds. These meshes can be directly imported into Unity and Unreal Engine, allowing the CG team to assemble, edit, and build production-ready environments with greater flexibility and quality.


Installing the hypano was such pain


anyways the path located is at
gsplat_env is the conda environment that you were using.

and then installing the vllm is another pain and set of things for the running up the infra.


```

source /devwork/MiniConda/miniconda3/etc/profile.d/conda.sh
conda activate gsplat_env




pip uninstall vllm -y
pip install vllm==0.25.1 \
  --extra-index-url https://wheels.vllm.ai/0.25.1/cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128



CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --port 8000 --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --trust-remote-code \
  --gpu-memory-utilization 0.30



mkdir -p /devwork/teja/HY-World-2.0/scenes/bajirao
cp /devwork/teja/HY-World-2.0/hyworld2/panogen/output_panorama.png \
   /devwork/teja/HY-World-2.0/scenes/bajirao/panorama.png
TARGET_PATH=/devwork/teja/HY-World-2.0/scenes/bajirao
RESULT_DIR=/devwork/teja/HY-World-2.0/scenes/bajirao/gs_output
LLM_ADDR=0.0.0.0
LLM_PORT=8000
LLM_NAME=Qwen/Qwen3-VL-8B-Instruct
cd /devwork/teja/HY-World-2.0/hyworld2/worldgen




this is to get rid of the libglib error
sudo apt-get install -y libgl1 libglib2.0-0

the next error is from torch 3d
pip uninstall -y pytorch3d
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"



I was getting segmentation dumped error becasue of the numpy which i got sorted later
pip install "numpy==1.26.4"

cd /devwork/teja/HY-World-2.0/hyworld2/worldgen
CUDA_VISIBLE_DEVICES=0 python traj_generate.py --target_path $TARGET_PATH \
  --llm_addr $LLM_ADDR --llm_port $LLM_PORT --llm_name $LLM_NAME \
  --apply_nav_traj --apply_up_route --apply_recon_iteration --force_vlm



# nccl error has beein popped up
# this is becasue we are running on the single gpu and the error has been fixed by the later by fixing the forward pass things.

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 traj_render.py \
  --target_path $TARGET_PATH \
  --llm_addr $LLM_ADDR --llm_port $LLM_PORT --llm_name $LLM_NAME



# some clip vision error now
# Transformers 5.x flattened `CLIPVisionModel` — it no longer has a nested `.vision_model`. Updating the wrapper to patch the model directly.

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 video_gen.py \
  --target_path $TARGET_PATH --fsdp


export LD_LIBRARY_PATH="/home/parth_h200/.conda/envs/gsplat_env/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH" && export TARGET_PATH=/devwork/teja/HY-World-2.0/scenes/bajirao && cd /devwork/teja/HY-World-2.0/hyworld2/worldgen && CUDA_VISIBLE_DEVICES=0 /home/parth_h200/.conda/envs/gsplat_env/bin/torchrun --nproc_per_node 1 video_gen.py --target_path $TARGET_PATH --fsdp 2>&1 | tail -80

export RESULT_DIR=/devwork/teja/HY-World-2.0/scenes/bajirao/gs_output

CUDA_VISIBLE_DEVICES=0 python -m world_gs_trainer default \
  --data_dir $TARGET_PATH/gs_data --result_dir $RESULT_DIR \
  --max_steps 8000 --save_steps 8000 --eval_steps 8000 --ply_steps 8000 \
  --save_ply --convert_to_spz --disable_video \
  --use_scale_regularization --antialiased \
  --depth_loss --normal_loss --sky_depth_from_pcd \
  --use_mask_gaussian --mask_export_stochastic \
  --no-mask-export-anchor-protection --use_anchor_protection --export_mesh \
  --strategy.refine-start-iter 800 --strategy.refine-stop-iter 4000 \
  --strategy.refine-every 533 --strategy.refine-scale2d-stop-iter 4000 \
  --strategy.reset-every 99990 --strategy.grow-grad2d 0.0001 --strategy.prune-scale3d 0.1


ls $TARGET_PATH/render_results/generation_bank_worldstereo-memory-dmd/
# should contain global_pcd.ply, aligned_pcd.ply, etc.



export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
cd /devwork/teja/HY-World-2.0/hyworld2/worldgen



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 gen_gs_data.py \
  --root_path $TARGET_PATH --save_normal --split_sky




CUDA_VISIBLE_DEVICES=0 python -m world_gs_trainer default \
  --data_dir /devwork/teja/HY-World-2.0/scenes/bajirao/gs_data \
  --result_dir /devwork/teja/HY-World-2.0/scenes/bajirao/gs_output \
  --max_steps 8000 --port 8081 \
  --save_ply --depth_loss --normal_loss --sky_depth_from_pcd





#View the mesh
conda activate gsplat_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
cd /devwork/teja/HY-World-2.0/hyworld2/worldgen

python gs/extract_mesh.py \
  --ckpt /devwork/teja/HY-World-2.0/scenes/bajirao/gs_output/ckpts/ckpt_7999_rank0.pt \
  --data_dir /devwork/teja/HY-World-2.0/scenes/bajirao/gs_data
  
  
  
  
  to run the vllm
  
   CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-VL-8B-Instruct   --served-model-name Qwen/Qwen3-VL-8B-Instruct   --port 8000 --host 0.0.0.0   --tensor-parallel-size 1   --max-model-len 32768   --trust-remote-code   --gpu-memory-utilization 0.80
   
   
   
   
python show_gs.py --port 8081 --gpu_id 0 \
  --ckpt "/devwork/teja/HY-World-2.0/scenes/bajirao/gs_output/ply/point_cloud_7999.ply"
   

conda activate gsplat_env

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"

cd /devwork/teja/HY-World-2.0/hyworld2/worldgen
```


alright, i was able to setup the entire process and was able to run it in the viser in the right environment.

I need to dig the deeper into each of the modules on how its working and setup somehow things that it can inpaint things in the right way so that it can generate movie set songs on movie set 3d meshes, for the entire set reconstruction in the better way.


The including steps include building the set from the multi reference images rather than a single image and single prompt.

The VLM and LLM should decide the trajectory and then start inpainting the modules accordingly so that the entire scene can be looking like its comming from a movie.

The modules in the hunyuan world must be doing that rather than continous usage of the video model in the first place and the envs shuld be loading dynamically.

google docs links for the paper:
https://docs.google.com/document/d/1UmfQaXa3UHY8lPvf--25B995sacazzn7rXaQMLQOSFo/edit?tab=t.ewt1bhhhhf7l


Further things to look out for

Given multi reference images and objects can we generate this?
If this can be generate, can we edit it basically based on the video perspective and then made this as a software?
How to control and edit the meshes on the fly if in the end we dont like it.

Datasets for training purpose:
datasets for the panaroma generation
datasets for the camera controlled video generation
