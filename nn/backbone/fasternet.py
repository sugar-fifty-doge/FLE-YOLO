# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch, yaml
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import numpy as np

__all__ = ['fasternet_t0', 'fasternet_t1', 'fasternet_t2', 'fasternet_s', 'fasternet_m', 'fasternet_l']

class Partial_conv3(nn.Module):   #在部分卷积上进行操作，实现不同的前向传播方式，，，，看代码主要实现了什么一般都是看前向传播的

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)   #部分卷积操作

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):   #MLP模块

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__() #调用父类nn.Module的一些构造方法
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(     #部分卷积操作
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):    #将多个多层感知机模块串联到一起，形成一个阶段

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:  #前向传播
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):   #将图像的大小分割为path_size的块，将图像转换成序列数据

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)  #创建了一个卷积层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):   #降维操作，相当于池化层啦，主要就是减少计算量并提取更高级别的特征表示

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)  #创建了一个名为reduction的卷积层（主要用来降维），有步长就是降维了
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()    #从父类nn.Module中继承一些方法

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split images into non-overlapping patches  把图像划分成不重叠的图像块（调用前面定义好的PatchEmbed类）
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule  随机深度衰减规则  在训练过程中以逐渐增加的方式丢弃一部分层，防止过拟合的发生
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers    （调用前面定义好的BasicStage类）
        stages_list = []
        for i_stage in range(self.num_stages):  #使用一个for循环遍历模型中的每一个阶段
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),   #维度大小和深度大小、丢弃率、都需要根据i计算，剩下的的都是固定值
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:   #判断是否为最后一个阶段，如果不是，则添加一个PatchMerging层，也是调用前面的类，
                                                # 用于图像块到的合并，主要作用是减少数据维度，提取更高级别的特征
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)   #将构建好的阶段按顺序组合成一个整体的神经网络

        # add a norm layer for each output   为每一个层的输出添加归一化操作
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                raise NotImplementedError
            else:
                layer = norm_layer(int(embed_dim * 2 ** i_emb))
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        
        self.channel = [i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
    def forward(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.patch_embed(x)       #将张量x输入模型的patch_embed层，得到处理后的张量x
        outs = []     #空列表，存储每一个阶段输出的特征张量
        for idx, stage in enumerate(self.stages):    #遍历模型的每一个阶段，对输入张量x进行
            x = stage(x)
            if idx in self.out_indices:    #如果当前索引在out_indices中，就需要进行标准化处理
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)     #将输出结果添加到输出列表中
        return outs

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}     #初始化idx和临时空字典，用于记录参数数量和临时需要保存的参数
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict     #更新模型权重

def fasternet_t0(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_t0.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t1(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_t1.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_t2(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_t2.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_s(weights=None, cfg='ultralytics/nn/backbone/faster_cfgg/fasternet_s.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_m(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_m.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

def fasternet_l(weights=None, cfg='ultralytics/nn/backbone/faster_cfg/fasternet_l.yaml'):
    with open(cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    model = FasterNet(**cfg)
    if weights is not None:
        pretrain_weight = torch.load(weights, map_location='cpu')
        model.load_state_dict(update_weight(model.state_dict(), pretrain_weight))
    return model

if __name__ == '__main__':     #主程序部分 （当一个python代码中没有一个明确的在主程序时，往往代表该部分代码被作为一个模块使用，用于被其他代码导入和调用）
    import yaml                #所以在这个界面需要执行这段包含函数、类、变量等定义的代码时，就要写入主程序，只想作为一个模块被其它模块调用时，就不需要主程序了
    model = fasternet_t0(weights='fasternet_t0-epoch.281-val_acc1.71.9180.pth', cfg='D:\\code\\hook\\yolov8-main（hook4）\\ultralytics\\nn\\backbone\\faster_cfg\\fasternet_t0.yaml')
    print(model.channel)    #打印输出了模型的通道数
    inputs = torch.randn((1, 3, 640, 640))   #创建随机输入张量
    for i in model(inputs):    #遍历模型，打印每个输出张量到的大小
        print(i.size())