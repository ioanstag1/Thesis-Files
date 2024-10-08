import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x, relative_pos=None):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if relative_pos is not None:
            attn += relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

#SuperToken Selection Module - from SPFormer , full code on : https://github.com/hhb072/STViT/blob/main/models/stvit.py#L206
class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size**2)
        weights = weights.reshape(kernel_size**2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x
        
#Visualisation 
import matplotlib.pyplot as plt
import os
def visualize_attention_maps(attention_maps, title="Attention Map"):
    """
    Visualize attention maps.

    Parameters:
    attention_maps (torch.Tensor): The attention maps to visualize. Expected shape: (B, num_heads, H, W)
    title (str): The title for the plot.
    """
    attention_maps = attention_maps.detach().cpu()
    B, num_heads, H, W = attention_maps.shape
    num_columns = 4
    num_rows = (num_heads + num_columns - 1) // num_columns  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attention_maps[0, i].numpy(), cmap='viridis')
        ax.set_title(f"Head {i + 1}")
        fig.colorbar(im, ax=ax)
        ax.axis('off')

    for i in range(num_heads, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
#Position Embedding
class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
                
        # self.conv_constant = nn.Parameter(torch.eye(kernel_size).reshape(dim, 1, kernel_size, kernel_size))
        # self.conv_constant.requires_grad = False
        
    def forward(self, x):
        # return F.conv2d(x, self.conv.weight+self.conv_constant, self.conv.bias, stride=1, padding=self.kernel_size//2, groups=self.dim) # equal to x + conv(x)
        return x + self.conv(x)

class AttentionS(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3, dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)
        # print("supertoken attention")
        # print("Input shape:", x.shape)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        # print("output shape:", x.shape)
        return x
        
class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()

class MlpS(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True, downsample=False, kernel_size=5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features
        #print(in_features,hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()         
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        
        self.conv = ResDWC(hidden_features, 3)
        
    def forward(self, x): 
        #print(x.shape)      
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)        
        x = self.conv(x)        
        x = self.fc2(x)               
        x = self.drop(x)
        return x
 
class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, n_iter=1, refine=True, refine_attention=True, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.n_iter = n_iter
        self.stoken_size = stoken_size
        self.refine = refine
        self.refine_attention = refine_attention

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        if refine:

            if refine_attention:
                self.stoken_refine = AttentionS(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w

        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)
        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)
        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(stoken_features)
            else:
                stoken_features = self.stoken_refine(stoken_features)
            #print(f"stoken_features after refinement: {stoken_features.shape}")

        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)
        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)
        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        # Visualize the supertokens
        #visualize_attention_maps(pixel_features, title="Supertokens")
        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                stoken_features = self.stoken_refine(stoken_features)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)



class Merge(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size,dim, stoken_size, num_heads, qkv_bias, attn_drop, proj_drop,mlp_ratio,act_layer=nn.GELU, drop = 0.,drop_path=0., in_chans=3,n_iter=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim * 4, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.stoken_size = stoken_size
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_out = nn.LayerNorm(out_dim)
        
        #SuperToken Cross-Attention Layer
        self.stoken_attention = StokenAttention(dim=in_dim, stoken_size=stoken_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop,n_iter=3)

        self.pos_embed = ResDWC(in_dim, 3)
                                        
        self.norm1 = LayerNorm2d(in_dim)
                    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(in_dim)
        self.mlp2 = MlpS(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed1, pixel_embed2):
        H_p = W_p = self.patch_size
        W_column = pixel_embed1.size(1)
        BW_column, H_row, _ = pixel_embed2.size()
        B = BW_column // W_column
        assert H_row == W_column

        img1 = pixel_embed1.reshape(B, H_row, W_column, H_p, W_p, self.in_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_dim, H_row * H_p, W_column * W_p)
        img2 = pixel_embed2.reshape(B, H_row, W_column, H_p, W_p, self.in_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_dim, H_row * H_p, W_column * W_p)
        
        # Apply StokenAttention and MLP to img1 and img2 separately
        if self.stoken_size > (1, 1):
            # visualize_attention_maps(img1, title="Image 1 before Superpixel Attention")
            # visualize_attention_maps(img2, title="Image 2 before Superpixel Attention")
        
            img1 = self.pos_embed(img1)
            img1 = img1 + self.drop_path(self.stoken_attention(self.norm1(img1)))
            img1 = img1 + self.drop_path(self.mlp2(self.norm2(img1)))
            
            img2 = self.pos_embed(img2)
            img2 = img2 + self.drop_path(self.stoken_attention(self.norm1(img2)))
            img2 = img2 + self.drop_path(self.mlp2(self.norm2(img2)))
            
            # Visualize attention maps after applying StokenAttention to img1 and img2
            # visualize_attention_maps(img1, title="Image 1 after Superpixel Attention")
            # visualize_attention_maps(img2, title="Image 2 after Superpixel Attention")
        
        # Concatenate img1 and img2
        img_reshaped = torch.cat([img1, img2], dim=1)
        
        # Apply convolutional layers to merged images
        img_merge = self.conv(img_reshaped)
        
        # Visualize the final merged image
        #visualize_attention_maps(img_merge, title="Final Merged Image")
          
        return img_merge
        
class Block(nn.Module):
    def __init__(self, dim, words_in_sentence, patch_size, sentences, in_chans=3, num_heads=2, num_inner_heads=4, mlp_ratio=4.,
            qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.patch_size = patch_size
        words = patch_size * patch_size
        self.norm_in = norm_layer(dim*words)
        self.attn_in1 = Attention(
            dim*words, dim*words, num_heads=num_inner_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn_in2 = Attention(
            dim*words, dim*words, num_heads=num_inner_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm_mlp_in = norm_layer(dim*words)
        self.mlp_in1 = Mlp(in_features=dim*words, hidden_features=int(dim*words * 4),
            out_features=dim*words, act_layer=act_layer, drop=drop)
        self.mlp_in2 = Mlp(in_features=dim*words, hidden_features=int(dim*words * 4),
            out_features=dim*words, act_layer=act_layer, drop=drop)
        
        self.norm_proj = norm_layer(dim*words)
        self.proj1 = nn.Linear(dim*words, dim*words, bias=True)
        self.proj2 = nn.Linear(dim*words, dim*words, bias=True)

        # Outer transformer
        self.norm_out = norm_layer(dim * words_in_sentence)
        self.attn_out1 = Attention(
            dim * words_in_sentence, dim * words_in_sentence, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn_out2 = Attention(
            dim * words_in_sentence, dim * words_in_sentence, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm_mlp = norm_layer(dim * words_in_sentence)
        self.mlp1 = Mlp(in_features=dim * words_in_sentence, hidden_features=int(dim * words_in_sentence * mlp_ratio),
            out_features=dim * words_in_sentence, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim * words_in_sentence, hidden_features=int(dim * words_in_sentence * mlp_ratio),
            out_features=dim * words_in_sentence, act_layer=act_layer, drop=drop)
        # self.relative_pos1 = nn.Parameter(torch.randn(1, num_heads, sentences, sentences))
        # self.relative_pos2 = nn.Parameter(torch.randn(1, num_heads, sentences, sentences))

    def forward(self, pixel_embed1, pixel_embed2, row_embed, column_embed, relative_pos=None):
        _, W_grid, _ = pixel_embed1.size()
        H_grid = W_grid
        H_p = W_p = self.patch_size
        B, N, C = row_embed.size()
        
        # outer
        assert N == H_grid
        row_embed = row_embed + self.drop_path(self.attn_out1(self.norm_out(row_embed)))
        row_embed = row_embed + self.drop_path(self.mlp1(self.norm_mlp(row_embed)))

        assert N == W_grid
        column_embed = column_embed + self.drop_path(self.attn_out2(self.norm_out(column_embed)))
        column_embed = column_embed + self.drop_path(self.mlp2(self.norm_mlp(column_embed)))

        # inner
        pixel_embed1 = pixel_embed1 + self.proj1(self.norm_proj(row_embed.reshape(B*H_grid, H_p, W_grid, W_p, -1).transpose(1, 2).reshape(B*H_grid, W_grid, -1)))
        attn_patch1 = self.attn_in1(self.norm_in(pixel_embed1.reshape(B, H_grid*W_grid, -1)))
        pixel_embed1 = pixel_embed1 + self.drop_path(attn_patch1.reshape(B*H_grid, W_grid, -1))
        pixel_embed1 = pixel_embed1 + self.proj2(self.norm_proj(column_embed.reshape(B, W_grid, H_grid, -1).transpose(1, 2).reshape(B*H_grid, W_grid, -1)))
        attn_patch2 = self.attn_in2(self.norm_in(pixel_embed1.reshape(B, H_grid*W_grid, -1)))
        pixel_embed1 = pixel_embed1 + self.drop_path(attn_patch2.reshape(B*H_grid, W_grid, -1))
        pixel_embed1 = pixel_embed1 + self.drop_path(self.mlp_in1(self.norm_mlp_in(pixel_embed1)))

        pixel_embed2 = pixel_embed2 + self.proj2(self.norm_proj(column_embed.reshape(B, W_grid, H_grid, -1).transpose(1, 2).reshape(B*H_grid, W_grid, -1)))
        attn_patch3 = self.attn_in2(self.norm_in(pixel_embed2.reshape(B, H_grid*W_grid, -1)))
        pixel_embed2 = pixel_embed2 + self.drop_path(attn_patch3.reshape(B*H_grid, W_grid, -1))
        pixel_embed2 = pixel_embed2 + self.proj1(self.norm_proj(row_embed.reshape(B*H_grid, H_p, W_grid, W_p, -1).transpose(1, 2).reshape(B*H_grid, W_grid, -1)))
        attn_patch4 = self.attn_in1(self.norm_in(pixel_embed2.reshape(B, H_grid*W_grid, -1)))
        pixel_embed2 = pixel_embed2 + self.drop_path(attn_patch4.reshape(B*H_grid, W_grid, -1))
        pixel_embed2 = pixel_embed2 + self.drop_path(self.mlp_in2(self.norm_mlp_in(pixel_embed2)))

        return pixel_embed1, pixel_embed2, row_embed, column_embed

class ToEmbed(nn.Module):
    def __init__(self, img_size=256, in_chans=3, patch_size=2, dim=8):
        super().__init__()
        img_size_tuple = (img_size, img_size)
        row_patch_size = (patch_size, img_size)
        self.grid_size = (img_size_tuple[0] // row_patch_size[0], img_size_tuple[1] // row_patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.img_size = img_size_tuple
        self.num_patches = num_patches
        self.row_patch_size = row_patch_size
        self.dim = dim
        row_pixel = row_patch_size[0] * row_patch_size[1]

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.norm_proj = nn.LayerNorm(row_pixel * dim)
        self.proj1 = nn.Linear(row_pixel * dim, row_pixel * dim)
        self.proj2 = nn.Linear(row_pixel * dim, row_pixel * dim)

    def forward(self, x, pixel_pos=None):
        B, C, H, W = x.shape
        assert H == self.img_size[0]
        assert W == self.img_size[1]

        x = self.unfold(x)
        if pixel_pos is not None:
            x = x + pixel_pos
        x = x.transpose(1, 2).reshape(B, self.num_patches, self.num_patches, self.dim, self.patch_size, self.patch_size)

        pixel_embed = x.permute(0, 1, 2, 4, 5, 3).reshape(B * self.num_patches, -1, self.patch_size * self.patch_size * self.dim)
        row_embed = self.norm_proj(x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.num_patches, -1))
        column_embed = self.norm_proj(x.permute(0, 2, 1, 4, 5, 3).reshape(B, self.num_patches, -1))
        return pixel_embed, pixel_embed, row_embed, column_embed


class Stage(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, dim, out_dim, num_heads=2, num_inner_head=2, depth=1,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, stoken_size=(8, 8)):
        super().__init__()

        self.pixel_embed = ToEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, dim=dim)
        row_patch_size = self.pixel_embed.row_patch_size
        self.row_pixel = row_patch_size[0] * row_patch_size[1]
        self.patch_pixel = patch_size * patch_size
        self.num_patches = self.pixel_embed.num_patches

        blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        for i in range(depth):
            blocks.append(Block(
                dim=dim, words_in_sentence=self.row_pixel, num_heads=num_heads, num_inner_heads=num_inner_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, in_chans=in_chans,
                drop_path=dpr[i], norm_layer=norm_layer, patch_size=patch_size, sentences=self.num_patches))
        self.blocks = nn.ModuleList(blocks)
        self.merge = Merge(in_dim=dim, out_dim=out_dim, patch_size=patch_size, in_chans=in_chans,dim=dim, stoken_size=stoken_size, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate, proj_drop=drop_rate,mlp_ratio=mlp_ratio, n_iter=1 )

        self.pos_drop = nn.Dropout(p=drop_rate)

    def forward(self, x, pixel_pos=None, row_pos=None, column_pos=None):
        pixel_embed1, pixel_embed2, row_embed, column_embed = self.pixel_embed(x, pixel_pos)
        if row_pos is not None:
            row_embed = row_embed + row_pos
        row_embed = self.pos_drop(row_embed)
        if column_pos is not None:
            column_embed = column_embed + column_pos
        column_embed = self.pos_drop(column_embed)
        for blk in self.blocks:
            pixel_embed1, pixel_embed2, row_embed, column_embed = blk(pixel_embed1, pixel_embed2, row_embed, column_embed)
        img_merge = self.merge(pixel_embed1, pixel_embed2)

        return img_merge

from torchvision import models 
class HoverTrans(nn.Module):
    def __init__(self, img_size=224, patch_size=[32, 16, 8, 4], in_chans=3, num_classes=2, embed_dim=768, dim=[48, 96, 192, 384], depth=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], num_inner_head=[2, 4, 8, 16], mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, stoken_size=[(4,4), (2, 2), (1,1), (1,1)]):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Χρήση του Densenet ως pre-trained stem
        densenet = models.densenet121(pretrained = True)
        stride = [4, 2, 2, 2]
        self.stage = nn.ModuleList([])
        self.downsample = nn.ModuleList([])
        for i in range(4):
            if i == 0:
                self.stage.append(Stage(img_size=img_size // stride[i], patch_size=patch_size[i], in_chans=in_chans, dim=dim[i], out_dim=dim[i] * 2,
                                        depth=depth[i], num_heads=num_heads[i], num_inner_head=num_inner_head[i], mlp_ratio=mlp_ratio, qkv_bias=True,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, stoken_size=stoken_size[i]))
                num_patches = self.stage[i].num_patches
                row_pixel = self.stage[i].row_pixel
                patch_pixel = self.stage[i].patch_pixel
                self.row_pos = nn.Parameter(torch.zeros(1, num_patches, row_pixel * dim[i]))
                self.column_pos = nn.Parameter(torch.zeros(1, num_patches, row_pixel * dim[i]))
                self.pixel_pos = nn.Parameter(torch.zeros(1, dim[i] * patch_pixel, num_patches * num_patches))

                #DenseNet
                self.downsample.append(nn.Sequential(
                            densenet.features[:4],  # Χρησιμοποιούμε τα πρώτα 4 layers
                            nn.Conv2d(64, dim[0], kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(dim[0]),
                            nn.ReLU(inplace=True),
                        ))
                #self.downsample.append(nn.Sequential(
                            #nn.Conv2d(in_chans, in_chans*2, 3, stride=2, padding=1),
                            #nn.BatchNorm2d(in_chans*2),
                            #nn.ReLU(inplace=True),
                            #nn.Conv2d(in_chans*2, in_chans*4, 3, stride=2, padding=1),
                            #nn.BatchNorm2d(in_chans*4),
                            #nn.ReLU(inplace=True),
                            #nn.Conv2d(in_chans*4, dim[i], 3, stride=1, padding=1),
                            #nn.BatchNorm2d(dim[i]),
                            #nn.ReLU(inplace=True),
                        #))
            else:
                self.stage.append(Stage(img_size=img_size // (2 ** (i + 2)), patch_size=patch_size[i], in_chans=dim[i - 1] * 2, dim=dim[i], out_dim=dim[i] * 2,
                                        depth=depth[i], num_heads=num_heads[i], num_inner_head=num_inner_head[i], mlp_ratio=mlp_ratio, qkv_bias=True,
                                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm, stoken_size=stoken_size[i]))
                self.downsample.append(nn.AvgPool2d(kernel_size=stride[i]))

        self.norm = norm_layer(dim[3] * 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim[3] * 2, num_classes)

        trunc_normal_(self.row_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
        trunc_normal_(self.column_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                trunc_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        img_ds = self.downsample[0](x)
        img_merge = self.stage[0](img_ds, self.pixel_pos, self.row_pos, self.column_pos)
        #visualize_image(img_merge, title=f"Image Merge Stage {0}")
        for i in range(3):
            img_ds = self.downsample[i + 1](img_merge)
            img_merge = self.stage[i + 1](img_ds)
            #visualize_image(img_merge, title=f"Image Merge Stage {i + 1}")
        return img_merge

    def forward(self, x):
        output = self.forward_features(x)
        output_flat = self.avgpool(output).flatten(1)
        output_flat = self.norm(output_flat)
        output_flat = self.head(output_flat)

        return output_flat


def create_model(embed_dim=640, **kwargs):
    model = HoverTrans(embed_dim=embed_dim, **kwargs)
    #print(model)
    return model

