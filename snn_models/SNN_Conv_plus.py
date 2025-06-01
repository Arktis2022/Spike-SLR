import json
import sys

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from timm.models.layers import DropPath
from timm.models.layers import to_2tuple
from torch.nn import functional as F

sys.path.append('../')

class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)

def nonzero_ratio(tensor):
    """
    Calculate the ratio of non-zero elements in a tensor.

    Arguments:
    tensor -- a PyTorch tensor

    Returns:
    ratio -- the ratio of non-zero elements to the total number of elements in the tensor
    """
    # Number of non-zero elements
    nonzero_count = torch.count_nonzero(tensor)
    
    # Total number of elements in the tensor
    total_count = tensor.numel()
    
    # Calculate the ratio
    ratio = nonzero_count.item() / total_count
    
    return ratio

# data pro-processing
class MS_SPS_Conv(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=4,
            in_channels=2,
            embed_dims=128,
            pooling_stat="11",
            spike_mode="lif",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        
        self.pooling_stat = pooling_stat

        self.C = in_channels
       
        self.H, self.W = (
            self.image_size[0] // patch_size[0],
            self.image_size[1] // patch_size[1],
        )
        
        self.num_patches = self.H * self.W
        
        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 4)
        if spike_mode == "lif":
            self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.proj_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.proj_conv1 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 2)
        if spike_mode == "lif":
            self.proj_lif1 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
    
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.proj_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.proj_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.last_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.last_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.last_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.last_bn = nn.BatchNorm2d(embed_dims)
        self.last_maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

    def forward(self, x, hook=None):
        # (3*16*2*108*108)
        T, B, _, H, W = x.shape
       
        ratio = 1
        
        x = self.proj_conv(x.flatten(0, 1))     
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  
        x = self.proj_lif(x)
        if hook is not None:
            hook[self._get_name() + "_lif"] = x.detach()
        x = self.proj_conv1(x.flatten(0, 1).contiguous())
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        if self.pooling_stat[0] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x_feat = x
        x = self.last_conv(x)
        x = self.last_bn(x)

        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W), hook
    
class Appearance_average(nn.Module):
    def __init__(
            self,
            in_features=128,
            hidden_features=512,
            embed_dims=128,
            spike_mode="lif",
    ):
        super().__init__()
        
        self.attention_conv1 = nn.Conv2d(
            in_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.attention_bn1 = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.attention_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.attention_lif1 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.attention_maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.attention_conv2 = nn.Conv2d(hidden_features, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.attention_bn2 = nn.BatchNorm2d(embed_dims)
        if spike_mode == "lif":
            self.attention_lif2 = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.attention_lif2 = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.attention_maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

    def forward(self, x, hook=None):
        # xsum_list = []
        xsum = torch.sum(x, dim=0, keepdim=True)
        xsum = xsum / x.size(0)
        xsum = xsum.repeat(x.size(0), 1, 1, 1, 1)
        T, B, _, H, W = xsum.shape
        x = self.attention_lif1(x)
        if hook is not None:
            hook[self._get_name() + "_lif1"] = x.detach()

        x = self.attention_conv1(xsum.flatten(0, 1))
        x = self.attention_bn1(x).reshape(T, B, -1, H, W).contiguous()
        
        x = self.attention_lif2(x)
        if hook is not None:
            hook[self._get_name() + "_lif2"] = x.detach()
        x = x.flatten(0, 1).contiguous()
        # x = self.attention_maxpool1(x)
        # print(nonzero_ratio(x))
        x = self.attention_conv2(x)
        x = self.attention_bn2(x).reshape(T, B, -1, H, W).contiguous()
        xsum_1 = torch.sum(x, dim=0, keepdim=True)
        xsum_1 = xsum_1 / x.size(0)
        xsum_1 = xsum_1.repeat(x.size(0), 1, 1, 1, 1)

        return xsum_1, hook

class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        # self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="torch"
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="torch"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        # identity = x
        N = H * W

        x = self.shortcut_lif(x)    # size (3*16*128*27*27)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)     

        q_conv_out = self.q_conv(x_for_qkv)     
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()   
        q_conv_out = self.q_lif(q_conv_out)    

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )   

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(x_for_qkv).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        
        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        # x = x + identity
        return x, v, hook

class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        
        hidden_features = hidden_features or in_features
        
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x, x_attention, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
        #     identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x_attention = x_attention.flatten(0, 1)
        x = x * x_attention
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        # x = x + identity
        return x, hook

class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )
        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )

    def forward(self, x, x_attention, hook=None):
        identify = x
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x, x_attention, hook=hook)
        x = x + x_attn
        x = x + identify
        return x, attn, hook

class EvNetBackbone(nn.Module):
    def __init__(self,
                 img_size_h=128,            
                 img_size_w=128,            
                 patch_size=4,            
                 in_channels=2,             
                 embed_dims=128,            
                 num_heads=8,               
                 mlp_ratios=2,             
                 qkv_bias=False,           
                 qk_scale=None,            
                 drop_rate=0.0,            
                 attn_drop_rate=0.0,       
                 drop_path_rate=0.0,        
                 norm_layer=nn.LayerNorm,   
                 depths=2,                  
                 pooling_stat="0011",       
                 attn_mode="direct_xor",
                 spike_mode="lif",
                 get_embed=False,           
                 dvs_mode=True,
                 TET=False,
                 cml=False,
                 pretrained=False,
                 pretrained_cfg=None,
                 ):
        super(EvNetBackbone, self).__init__()

        self.depths = depths

        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]

        patch_embed = MS_SPS_Conv(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        attentions = Appearance_average(
            embed_dims=embed_dims,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depths)
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"attentions", attentions)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        # self.dropout = nn.Dropout(p=0.2)
        # self.head = nn.Linear(embed_dims, embed_dims)
        # self.head1 = (
        #     nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        # )

    def forward_features(self, x, hook=None):
        patch_embed = getattr(self, f"patch_embed")
        block = getattr(self, f"block")
        attentions = getattr(self, f"attentions")

        x, _, hook = patch_embed(x, hook=hook)
        x_attention, _ = attentions(x, hook=hook)

        for blk in block:
            x, _, hook = blk(x, x_attention, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None):
        x = x.transpose(0, 1).contiguous()
        x, hook = self.forward_features(x, hook=hook)
        # 10*16*128
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()
        # x = self.dropout(x)
        #
        # x = F.relu(self.head(x))
        # x = self.head1(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.mean(axis=0)
        # # x.shape:torch.Tensor([8,19])
        return x


class CLFBlock(nn.Module):
    def __init__(self, ipt_dim, opt_classes, **args):
        super(CLFBlock, self).__init__()
        self.clf_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.linear_1 = nn.Linear(ipt_dim, ipt_dim)
        self.linear_2 = nn.Linear(ipt_dim, opt_classes)

    def forward(self, x):
        # (3,16,128)
        x = self.clf_lif(self.linear_1(x))
        x = self.linear_2(x)
        x = x.mean(axis=0)
        clf = F.log_softmax(x, dim=1)
        return clf

