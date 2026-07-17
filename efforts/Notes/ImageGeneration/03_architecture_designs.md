
Building the transformer from Ground up

The notes to be taken from the 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html


Breaking things into patches is the first thing to do
https://uvadlc-notebooks.readthedocs.io/en/latest/_images/vit.gif


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

https://uvadlc-notebooks.readthedocs.io/en/latest/_images/tutorial_notebooks_tutorial15_Vision_Transformer_11_0.svg

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


Linear Projection
Classification Token
Position l Encodings
MLP Head

