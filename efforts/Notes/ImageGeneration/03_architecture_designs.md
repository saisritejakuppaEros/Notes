
Building the transformer from Ground up

The notes to be taken from the 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html


Breaking things into patches is the first thing to do

<img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/vit.gif" width="700">


```
def img_to_patch(x, patch_size, flatten_channels=True):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, C, H, W]
        patch_size - Number of pixels per dimension of the patches (integer)
        flatten_channels - If True, the patches will be returned in a flattened format
                           as a feature vector instead of a image grid.
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x
```



Now once the patches are present, you are able to run the things. 

<img src="https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial15_Vision_Transformer_11_0.svg" width="700">

```
img_patches = img_to_patch(CIFAR_images, patch_size=4, flatten_channels=False)

fig, ax = plt.subplots(CIFAR_images.shape[0], 1, figsize=(14,3))
fig.suptitle("Images as input sequences of patches")
for i in range(CIFAR_images.shape[0]):
    img_grid = torchvision.utils.make_grid(img_patches[i], nrow=64, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    ax[i].imshow(img_grid)
    ax[i].axis('off')
plt.show()
plt.close()
```



# Attention Block


```
class AttentionBlock(nn.Module):

	def __init__(self, embed_dim, hidden_dim, num_heads, dropout =0.0):
	
		super().__init__()
		self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout =dropout)
		
		self.layernorm_2 = nn.layernorm(embed_dim)
		self.linear = nn.sequential(
		nn.Linear(embed_dim, hidden_dim),
		nn.GELU(),
		nn.Dropout(dropout),
		nn.Linear(hidden_dim, embed_dim),
		nn.Dropout(dropout)
		)
	
	def forward(self, x):
		inp_x = self.layernorm(x)
		x = x + self.attn(inp_x, inp_x, inp_x)[0]
		x = x + self.linear(slef.layernorm2(x))
		return x
```


Submodules to look out for:
Linear Projection
Classification Token
Position l Encodings
MLP Head


```
class VisionTransformer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_channels, num_heads, num_layers, num_classes, patch_size, num_patches, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        self.input_layer = nn.Linear(num_channels*(patch_size**2), embed_dim)
        self.transformer = nn.Sequential(*[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)

        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))

    def forward(self, x):
        # Preprocess input
        x = img_to_patch(x, self.patch_size)
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out
```



This is the fundemental of a transformer architecture, lets do some shape analysis to see whats inside of it to understand the dims of what goes in and what is the output shape. 
	
|#|Module / Layer|Operation|Input Shape|Output Shape|Notes|
|---|---|---|---|---|---|
|1|Input|RGB Image|—|**[128, 3, 32, 32]**|Batch of CIFAR-10 images|
|2|`reshape()`|Split into patch grid|[128, 3, 32, 32]|**[128, 3, 8, 4, 8, 4]**|8×8 patches, each 4×4|
|3|`permute()`|Move patch dimensions together|[128, 3, 8, 4, 8, 4]|**[128, 8, 8, 3, 4, 4]**|Patch row, patch col, channels|
|4|`flatten(1,2)`|Flatten patch grid|[128, 8, 8, 3, 4, 4]|**[128, 64, 3, 4, 4]**|64 patches/image|
|5|`flatten(2,4)`|Flatten pixels inside each patch|[128, 64, 3, 4, 4]|**[128, 64, 48]**|3×4×4 = 48 features/patch|
|6|Linear Projection|`Linear(48 → 256)`|[128, 64, 48]|**[128, 64, 256]**|Patch Embedding|
|7|CLS Token|Concatenate learnable token|[128, 64, 256]|**[128, 65, 256]**|Adds one extra token|
|8|Position Embedding|Add positional encoding|[128, 65, 256]|**[128, 65, 256]**|Shape unchanged|
|9|Dropout|Regularization|[128, 65, 256]|**[128, 65, 256]**|Shape unchanged|
|10|Transpose|Batch-first → Sequence-first|[128, 65, 256]|**[65, 128, 256]**|Required by `nn.MultiheadAttention`|



| #   | Module               | Operation            | Input Shape    | Output Shape       | Notes                              |
| --- | -------------------- | -------------------- | -------------- | ------------------ | ---------------------------------- |
| 11  | LayerNorm            | Normalize embeddings | [65, 128, 256] | **[65, 128, 256]** | No shape change                    |
| 12  | Multi-Head Attention | Self Attention       | [65, 128, 256] | **[65, 128, 256]** | Output dimension = embed dimension |
| 13  | Residual Add         | `x + attention`      | [65, 128, 256] | **[65, 128, 256]** | Skip connection                    |
| 14  | LayerNorm            | Normalize again      | [65, 128, 256] | **[65, 128, 256]** | No shape change                    |
| 15  | Linear               | `256 → 512`          | [65, 128, 256] | **[65, 128, 512]** | Expansion                          |
| 16  | GELU                 | Activation           | [65, 128, 512] | **[65, 128, 512]** | Shape unchanged                    |
| 17  | Dropout              | Regularization       | [65, 128, 512] | **[65, 128, 512]** | Shape unchanged                    |
| 18  | Linear               | `512 → 256`          | [65, 128, 512] | **[65, 128, 256]** | Projection back                    |
| 19  | Dropout              | Regularization       | [65, 128, 256] | **[65, 128, 256]** | Shape unchanged                    |
| 20  | Residual Add         | `x + MLP`            | [65, 128, 256] | **[65, 128, 256]** | End of one encoder block           |
|     |                      |                      |                |                    |                                    |

|#|Module|Input Shape|Output Shape|Notes|
|---|---|---|---|---|
|21|Encoder Block × 6|[65, 128, 256]|**[65, 128, 256]**|Shape remains constant through all blocks|

|#|Module|Operation|Input Shape|Output Shape|Notes|
|---|---|---|---|---|---|
|22|CLS Extraction|`x[0]`|[65, 128, 256]|**[128, 256]**|Select first token only|
|23|LayerNorm|Normalize|[128, 256]|**[128, 256]**|No shape change|
|24|Linear|`256 → 10`|[128, 256]|**[128, 10]**|Final class logits|



Alright, lets dive much deeper into the variants of the transformer models to do deal with generative AI papers:

| Model / Project    | Link                                                                                  | Category              | Notes                                                              |
| ------------------ | ------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------ |
| PixArt-sigma       | https://github.com/PixArt-alpha/PixArt-sigma                                          | Text-to-image         | Open model/research project from PixArt-alpha.                     |
| Lumina-T2X         | https://github.com/Alpha-VLLM/Lumina-T2X/tree/main                                    | Text-to-image         | Open text-to-image project from Alpha-VLLM.                        |
| MaskDiT            | https://github.com/Anima-Lab/MaskDiT                                                  | Text-to-image         | Diffusion Transformer-based image generation project.              |
| Stable Diffusion 3 | https://arxiv.org/pdf/2403.03206                                                      | Text-to-image         | Stable Diffusion 3 paper/research reference.                       |
| Flux 2.0           | https://bfl.ai/research/representation-comparison                                     | Text-to-image         | Black Forest Labs research page related to Flux.                   |
| Qwen-Image         | https://github.com/QwenLM/Qwen-Image                                                  | Chinese text-to-image | Chinese open-source image generation model.                        |
| Kandinsky Image    | [GitHub - ai-forever/Kandinsky-3 · GitHub](https://github.com/ai-forever/Kandinsky-3) | Chinese text-to-image | Note: this is **not Chinese**; it is a Russian image model family. |
| Z-Image            | https://huggingface.co/collections/Ji-Xiang/image-generator                           | Chinese text-to-image | Chinese image-generation model family/collection.                  |





















## Best Repositories for Fine-Tuning Diffusion Models

| Repository | GitHub | Type | Supported Training |
|------------|--------|------|--------------------|
| **SimpleTuner** | https://github.com/bghira/SimpleTuner | CLI Training Toolkit | General fine-tuning for diffusion models, LoRA, full fine-tuning, DreamBooth, SDXL, Flux, and other modern diffusion models. |
| **kohya-ss / sd-scripts** | https://github.com/kohya-ss/sd-scripts | Training Scripts | DreamBooth, LoRA, LyCORIS, Textual Inversion, Hypernetworks, Native Fine-tuning, Image Generation, Model Conversion, SD/SDXL/Flux support. |
| **kohya_ss GUI** | https://github.com/bmaltais/kohya_ss | GUI Frontend | Gradio-based interface for Kohya's training scripts with support for LoRA, DreamBooth, SDXL, Flux, LyCORIS, and Textual Inversion. |
| **ModelScope Stable Diffusion Examples** | https://github.com/modelscope/modelscope | Training Framework | LoRA fine-tuning examples, Stable Diffusion training, DreamBooth, ControlNet, and multimodal model training using the ModelScope ecosystem. |