import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import numpy as np 

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
        # Combine Q and K and reshape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk.unbind(0)  # Split into q and k
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # Process v separately
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # QK^T
        if relative_pos is not None:
            attn += relative_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #print(q.shape)
        #print("attn",attn)
        #print("attn:",attn.shape)
        # x = (A_reduced @ V).transpose(1, 2).reshape(B, T, -1)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return q, k, v, attn


class TokenSelector(nn.Module):
    def __init__(self, H, H_prime, embed_dim, kernel_size=3, attn_drop=0.):
        super(TokenSelector, self).__init__()
        self.H = H  # Original number of attention heads
        self.H_prime = H_prime  # Expanded number of attention heads
        self.embed_dim = embed_dim  # Embedding dimension
        self.head_dim = embed_dim // H
        self.scale = self.head_dim ** -0.5  # Scale factor for attention

        # 1x1 convolution for expanding attention maps from H to H_prime
        self.expand_conv = nn.Conv2d(H, H_prime, kernel_size=1, bias=False)

        # Convolution for local token selection applied separately to each head with parameter groups=H_prime
        self.local_conv = nn.Conv2d(H_prime, H_prime, kernel_size=kernel_size, padding=kernel_size // 2, groups=H_prime)

        # Linear projection to reduce the number of attention maps from H_prime to H
        self.reduce_conv = nn.Conv2d(H_prime, H, kernel_size=1, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    def forward(self, A, Q, K, V):
        B, H, T, _ = A.shape  # Batch size (B), number of heads (H), number of tokens (T), attention map size

        # Expand the attention maps A from H to H_prime using a 1x1 convolution
        A_expanded = self.expand_conv(A)  # [B, H_prime, T, T]

        # Apply grouped convolution and ReLU on the expanded attention maps
        A_local = self.local_conv(A_expanded)
        A_local = F.relu(A_local)  # Zero out negative values to enhance selectivity

        # Apply linear projection to reduce the number of attention maps back to the original H
        A_reduced = self.reduce_conv(A_local)

        # Scale the attention maps
        A_reduced = A_reduced * self.scale

        # Apply softmax to get the attention scores
        attn = A_reduced.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute attention output 
        output = attn@V

        # Project the output back to the original embedding dimension
        output = output.transpose(1, 2).reshape(B, T, -1)
        output = self.proj(output)


        return output, A_expanded


"""
class TokenSelector(nn.Module):
    def __init__(self, H, H_prime, embed_dim, kernel_size=3, attn_drop=0.):
        super(TokenSelector, self).__init__()
        self.H = H  # Original number of attention heads
        self.H_prime = H_prime  # Expanded number of attention heads
        self.embed_dim = embed_dim  # Embedding dimension
        self.head_dim = embed_dim // H
        self.scale = self.head_dim ** -0.5  # Scale factor for attention

        # 1x1 convolution for expanding attention maps from H to H_prime
        self.expand_conv = nn.Conv2d(H, H_prime, kernel_size=1, bias=False)

        # Convolution for local token selection applied separately to each head with parameter groups=H_prime
        self.local_conv = nn.Conv2d(H_prime, H_prime, kernel_size=kernel_size, padding=kernel_size // 2, groups=H_prime)

        # Linear projection to reduce the number of attention maps from H_prime to H
        self.reduce_conv = nn.Conv2d(H_prime, H, kernel_size=1, bias=False)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    def forward(self, A, Q, K, V):
        B, H, T, _ = A.shape  # Batch size (B), number of heads (H), number of tokens (T), attention map size

        # Expand the attention maps A from H to H_prime using a 1x1 convolution
        A_expanded = self.expand_conv(A)  # [B, H_prime, T, T]

        # Apply grouped convolution and ReLU on the expanded attention maps
        A_local = self.local_conv(A_expanded)
        A_local = F.relu(A_local)  # Zero out negative values to enhance selectivity

        # Compute attention output using einsum
        A_p = A_local.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        output = torch.einsum('bthc,btkd->bthd', A_p, V)
        output = output.permute(0, 2, 1, 3)
        #outputback to original heads 
        output = self.reduce_conv(output)
        # output back to the original embedding dimension
        output = output.transpose(1, 2).reshape(B, T, -1)
        output = self.proj(output)
        return output, A_expanded
"""
class TSViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, H_prime):
        super(TSViTBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.H_prime = max(H_prime, num_heads)
        self.attention = Attention(embed_dim, hidden_dim=embed_dim * 4, num_heads=num_heads)
        self.token_selector = TokenSelector(num_heads, H_prime, embed_dim)
    def forward(self, x):
        q, k, v, attn_output_weights = self.attention(x)
        B, N, C = x.shape
        attn_local_selected,attn_mixed = self.token_selector(attn_output_weights, q, k, v)
        return attn_local_selected ,attn_mixed


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

class Merge(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size, in_chans=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim*4, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim*4, in_dim*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim*4, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            )
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, pixel_embed1, pixel_embed2):
        H_p = W_p = self.patch_size
        W_column = pixel_embed1.size(1)
        BW_column, H_row, _ = pixel_embed2.size()
        B = BW_column // W_column
        assert H_row == W_column

        img1 = pixel_embed1.reshape(B, H_row, W_column, H_p, W_p, self.in_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_dim, H_row*H_p, W_column*W_p)
        img2 = pixel_embed2.reshape(B, H_row, W_column, H_p, W_p, self.in_dim).permute(0, 5, 1, 3, 2, 4).reshape(B, self.in_dim, H_row*H_p, W_column*W_p)
        img_reshaped = torch.cat([img1, img2], dim=1)
        img_merge = self.conv(img_reshaped)

        return img_merge

class Block(nn.Module):
    def __init__(self, dim, words_in_sentence, patch_size, sentences, in_chans=3, num_heads=4, num_inner_heads=4,
                 mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
		
        # Inner transformer
        self.patch_size = patch_size
        words = patch_size * patch_size
        self.norm_in = norm_layer(dim * words)
        self.attn_in1 = TSViTBlock(embed_dim=dim * words, num_heads=num_inner_heads, H_prime=num_inner_heads * 2)
        self.attn_in2 = TSViTBlock(embed_dim=dim * words, num_heads=num_inner_heads, H_prime=num_inner_heads * 2)
        self.norm_mlp_in = norm_layer(dim * words)
        self.mlp_in1 = Mlp(in_features=dim * words, hidden_features=int(dim * words * 4),
                           out_features=dim * words, act_layer=act_layer, drop=drop)
        self.mlp_in2 = Mlp(in_features=dim * words, hidden_features=int(dim * words * 4),
                           out_features=dim * words, act_layer=act_layer, drop=drop)

        self.norm_proj = norm_layer(dim * words)
        self.proj1 = nn.Linear(dim * words, dim * words, bias=True)
        self.proj2 = nn.Linear(dim * words, dim * words, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim * words_in_sentence)
        self.attn_out1 = TSViTBlock(embed_dim=dim * words_in_sentence, num_heads=num_heads, H_prime=num_heads * 2)
        self.attn_out2 = TSViTBlock(embed_dim=dim * words_in_sentence, num_heads=num_heads, H_prime=num_heads * 2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim * words_in_sentence)
        self.mlp1 = Mlp(in_features=dim * words_in_sentence,
                        hidden_features=int(dim * words_in_sentence * mlp_ratio),
                        out_features=dim * words_in_sentence, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim * words_in_sentence,
                        hidden_features=int(dim * words_in_sentence * mlp_ratio),
                        out_features=dim * words_in_sentence, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed1, pixel_embed2, row_embed, column_embed, relative_pos=None):
        _, W_grid, _ = pixel_embed1.size()
        H_grid = W_grid
        H_p = W_p = self.patch_size
        B, N, C = row_embed.size()

        # outer
        assert N == H_grid
        # x+attention
        row_embed_attn_out1, attn_map1 = self.attn_out1(self.norm_out(row_embed))  # Extract first element
        #print("mixed attn1:",attn_map1.shape)
        row_embed = row_embed + self.drop_path(row_embed_attn_out1)
        # x+ffn
        row_embed = row_embed + self.drop_path(self.mlp1(self.norm_mlp(row_embed)))
        assert N == W_grid
        column_embed_attn_out2, attn_map2 = self.attn_out2(self.norm_out(column_embed))  # Extract first element
        #print("mixed attn2:",attn_map2.shape)
        column_embed = column_embed + self.drop_path(column_embed_attn_out2)
        column_embed = column_embed + self.drop_path(self.mlp2(self.norm_mlp(column_embed)))

        # inner
        pixel_embed1 = pixel_embed1 + self.proj1(
            self.norm_proj(row_embed.reshape(B * H_grid, H_p, W_grid, W_p, -1).transpose(1, 2).reshape(B * H_grid,
                                                                                                       W_grid, -1)))
        attn_patch1, _ = self.attn_in1(self.norm_in(pixel_embed1.reshape(B, H_grid * W_grid, -1)))  # Extract first element
        pixel_embed1 = pixel_embed1 + self.drop_path(attn_patch1.reshape(B * H_grid, W_grid, -1))
        pixel_embed1 = pixel_embed1 + self.proj2(
            self.norm_proj(column_embed.reshape(B, W_grid, H_grid, -1).transpose(1, 2).reshape(B * H_grid, W_grid, -1)))
        attn_patch2, attn_map3 = self.attn_in2(self.norm_in(
            pixel_embed1.reshape(B, H_grid * W_grid, -1)))  # Extract first element
        #print("mixed attn3:",attn_map3.shape)
        pixel_embed1 = pixel_embed1 + self.drop_path(attn_patch2.reshape(B * H_grid, W_grid, -1))
        pixel_embed1 = pixel_embed1 + self.drop_path(self.mlp_in1(self.norm_mlp_in(pixel_embed1)))

        pixel_embed2 = pixel_embed2 + self.proj2(
            self.norm_proj(column_embed.reshape(B, W_grid, H_grid, -1).transpose(1, 2).reshape(B * H_grid, W_grid, -1)))
        attn_patch3, _ = self.attn_in2(self.norm_in(
            pixel_embed2.reshape(B, H_grid * W_grid, -1)))  # Extract first element
        pixel_embed2 = pixel_embed2 + self.drop_path(attn_patch3.reshape(B * H_grid, W_grid, -1))
        pixel_embed2 = pixel_embed2 + self.proj1(
            self.norm_proj(row_embed.reshape(B * H_grid, H_p, W_grid, W_p, -1).transpose(1, 2).reshape(B * H_grid,
                                                                                                       W_grid, -1)))
        attn_patch4, attn_map4 = self.attn_in1(self.norm_in(
            pixel_embed2.reshape(B, H_grid * W_grid, -1)))  # Extract first element
        #print("mixed attn4:",attn_map4.shape)
        pixel_embed2 = pixel_embed2 + self.drop_path(attn_patch4.reshape(B * H_grid, W_grid, -1))
        pixel_embed2 = pixel_embed2 + self.drop_path(self.mlp_in2(self.norm_mlp_in(pixel_embed2)))

        return pixel_embed1, pixel_embed2, row_embed, column_embed, [attn_map1, attn_map2, attn_map3, attn_map2 ]



class ToEmbed(nn.Module):
    def __init__(self, img_size, in_chans=3, patch_size=2, dim=8):
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
        x = x.transpose(1, 2).reshape(B , self.num_patches, self.num_patches, self.dim, self.patch_size, self.patch_size)

        pixel_embed = x.permute(0, 1, 2, 4, 5, 3).reshape(B * self.num_patches, -1, self.patch_size*self.patch_size*self.dim)
        row_embed = self.norm_proj(x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.num_patches, -1))
        column_embed =  self.norm_proj(x.permute(0, 2, 1, 4, 5, 3).reshape(B, self.num_patches, -1))

        return pixel_embed, pixel_embed, row_embed, column_embed

class Stage(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, dim, out_dim, num_heads=2, num_inner_head=2, depth=1,
                 mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
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
        self.merge = Merge(in_dim=dim, out_dim=out_dim, patch_size=patch_size, in_chans=in_chans,)

        self.pos_drop = nn.Dropout(p=drop_rate)


    def forward(self, x, pixel_pos=None, row_pos=None, column_pos=None):
        pixel_embed1, pixel_embed2, row_embed, column_embed = self.pixel_embed(x, pixel_pos)
        if row_pos is not None:
            row_embed = row_embed + row_pos
        row_embed = self.pos_drop(row_embed)
        if column_pos is not None:
            column_embed = column_embed + column_pos
        column_embed = self.pos_drop(column_embed)

        attn_maps = []  # To collect attention maps for CLAS loss calculation

        for blk in self.blocks:
            pixel_embed1, pixel_embed2, row_embed, column_embed, attn_layers = blk(pixel_embed1, pixel_embed2, row_embed, column_embed)
            attn_maps.extend(attn_layers)  # Collect attention maps from each block

        img_merge = self.merge(pixel_embed1, pixel_embed2)

        return img_merge, attn_maps

from torchvision import models

class HoverTrans(nn.Module):
    def __init__(self, img_size, patch_size=32, in_chans=3, num_classes=2, embed_dim=768, dim=48, depth=12,
                 num_heads=12, num_inner_head=4, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
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
                self.stage.append(Stage(img_size=img_size//stride[i], patch_size=patch_size[i], in_chans=in_chans, dim=dim[i], out_dim=dim[i]*2,
                            depth=depth[i], num_heads=num_heads[i], num_inner_head=num_inner_head[i], mlp_ratio=mlp_ratio, qkv_bias=False,
                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm))
                num_patches = self.stage[i].num_patches
                row_pixel = self.stage[i].row_pixel
                patch_pixel = self.stage[i].patch_pixel
                self.row_pos = nn.Parameter(torch.zeros(1, num_patches, row_pixel * dim[i]))
                self.column_pos = nn.Parameter(torch.zeros(1, num_patches, row_pixel * dim[i]))
                self.pixel_pos = nn.Parameter(torch.zeros(1, dim[i]*patch_pixel, num_patches * num_patches))
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
                self.stage.append(Stage(img_size=img_size//(2**(i+2)), patch_size=patch_size[i], in_chans=dim[i], dim=dim[i], out_dim=dim[i]*2,
                            depth=depth[i], num_heads=num_heads[i], num_inner_head=num_inner_head[i], mlp_ratio=mlp_ratio, qkv_bias=False,
                            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=nn.LayerNorm))
                self.downsample.append(nn.AvgPool2d(kernel_size=stride[i]))

        self.norm = norm_layer(dim[3]*2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim[3]*2, num_classes)

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
        img_merge, attn_maps_stage = self.stage[0](img_ds, self.pixel_pos, self.row_pos, self.column_pos)


        for i in range(3):
            img_ds = self.downsample[i+1](img_merge)
            img_merge, attn_maps_stage = self.stage[i+1](img_ds)

        return img_merge, attn_maps_stage

    def forward(self, x):
        output, attn_maps = self.forward_features(x)
        output_flat = self.avgpool(output).flatten(1)
        output_flat = self.norm(output_flat)
        output_flat = self.head(output_flat)

        return output_flat, attn_maps


def create_model(embed_dim=640, **kwargs):
    model = HoverTrans(embed_dim=embed_dim, **kwargs)
    return model


