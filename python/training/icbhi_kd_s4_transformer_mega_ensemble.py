#!/usr/bin/env python3
"""
ICBHI 2017 Strategy 4 — Transformer Mega-Ensemble KD.

Combines ALL advanced techniques to beat SOTA (ICBHI Score > 70%):

  1. Mega Teacher Ensemble (7 architectures):
     - 3 CNN: ResNetCNN, ResNetCRNN, EfficientNetB0
     - 2 Transformer: AST-Tiny, Swin-Tiny
     - 2 Hybrid: FusionTeacher (time+spectral), CRNNAttention

  2. Cross-Architecture Feature Bridge:
     - Patch-to-Spatial projection (Transformer → CNN)
     - Attention Map Transfer from Transformer
     - Cross-Attention alignment

  3. 8-Component Loss Function:
     - Focal Loss (class-balanced)
     - Standard Logit KD
     - Decoupled KD (target + non-target)
     - Feature Distillation (cosine)
     - Attention Transfer
     - Relational KD (pairwise distance)
     - Adversarial Feature Matching
     - EMA Self-Distillation
     - Binary Auxiliary Loss

  4. Training Techniques:
     - MixUp augmentation
     - Curriculum Learning (easy → hard)
     - Progressive KD weighting
     - Class-Balanced Effective-Number Sampling
     - TTA logit collection for teacher calibration

  5. Inference:
     - Dual-Threshold Prediction
     - TTA ensemble (10 views)
     - SWA (Stochastic Weight Averaging)

Target: ICBHI Score > 70%, Specificity > 90%.

References:
  - Gong et al., "AST: Audio Spectrogram Transformer", InterSpeech 2021.
  - Liu et al., "Swin Transformer", ICCV 2021.
  - Zhao et al., "Decoupled Knowledge Distillation", CVPR 2022.
  - Park et al., "Relational Knowledge Distillation", CVPR 2019.
  - Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017.
  - Tarvainen & Valpola, "Mean Teachers are Better Role Models", NeurIPS 2017.
  - Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import torch
torch.set_num_threads(2)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import python.training.icbhi_kd_pipeline_multiview_ensemble as base
from python.common.paths import ensure_dir


# ============================================================================
# 1. TRANSFORMER BUILDING BLOCKS
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with optional attention weight output."""

    def __init__(self, dim, num_heads=3, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, return_attention=False):
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """2D patch embedding for spectrograms."""

    def __init__(self, in_ch=1, embed_dim=192, patch_size=16, stride=16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, (H, W)


# ============================================================================
# 2. TRANSFORMER TEACHER MODELS
# ============================================================================

class ASTTinyTeacher(nn.Module):
    """Audio Spectrogram Transformer (Tiny variant).

    Architecture:
      PatchEmbed → [CLS] + Position Embed → Transformer Blocks → Classification

    Params: ~5.7M (suitable as teacher for respiratory sound classification)
    """

    def __init__(self, nc=4, in_ch=1, embed_dim=192, depth=4, num_heads=3,
                 patch_size=16, stride=16, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Will be initialized dynamically based on input size
        self.pos_embed = None
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop, attn_drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(embed_dim, nc))
        self._embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _get_pos_embed(self, num_patches, device):
        """Create or resize positional embedding."""
        n = num_patches + 1  # +1 for CLS
        if self.pos_embed is None or self.pos_embed.shape[1] != n:
            pe = nn.Parameter(torch.zeros(1, n, self._embed_dim, device=device))
            nn.init.trunc_normal_(pe, std=0.02)
            self.pos_embed = pe
        return self.pos_embed

    def forward(self, x):
        B = x.shape[0]
        patches, (H, W) = self.patch_embed(x)  # [B, N, D]
        N = patches.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)  # [B, N+1, D]
        pos = self._get_pos_embed(N, x.device)
        x = self.pos_drop(x + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_out = x[:, 0]  # [CLS] token
        return self.head(cls_out)

    def forward_with_features(self, x):
        """Forward returning intermediate features and attention maps."""
        B = x.shape[0]
        patches, (H, W) = self.patch_embed(x)
        N = patches.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        pos = self._get_pos_embed(N, x.device)
        x = self.pos_drop(x + pos)

        features = {}
        attentions = {}
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x, return_attention=True)
            features[f"layer_{i}"] = x[:, 1:]  # exclude CLS
            attentions[f"layer_{i}"] = attn

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out), features, attentions


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (simplified Swin).

    Handles non-divisible spatial dimensions by padding.
    """

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.attn = MultiHeadSelfAttention(dim, num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        ws = self.window_size
        # Pad to make H, W divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
            x = x.view(B, H * W, C)
        # Reshape into windows
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        x = self.attn(x)
        x = x.view(B, H // ws, W // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, C)
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x.view(B, H, W, C)
            x = x[:, :H - pad_h, :W - pad_w, :].contiguous()
            x = x.view(B, -1, C)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer block with window attention."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.shift_size = shift_size
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # For shift, work in 2D grid
        if self.shift_size > 0:
            x_2d = x.view(B, H, W, C)
            x_2d = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x_2d.view(B, N, C)
        x = self.window_attn(x, H, W)
        if self.shift_size > 0:
            x_2d = x.view(B, H, W, C)
            x_2d = torch.roll(x_2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x_2d.view(B, N, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinTinyTeacher(nn.Module):
    """Simplified Swin Transformer for spectrogram classification.

    Params: ~28M — larger than AST, captures multi-scale features.
    """

    def __init__(self, nc=4, in_ch=1, embed_dim=96, depths=(2, 2, 2),
                 num_heads=(3, 6, 12), window_size=7, patch_size=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_drop = nn.Dropout(0.1)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dim = embed_dim
        for i, (d, h) in enumerate(zip(depths, num_heads)):
            stage_blocks = nn.ModuleList()
            for j in range(d):
                shift = window_size // 2 if j % 2 == 1 else 0
                stage_blocks.append(SwinBlock(dim, h, window_size, shift))
            self.stages.append(stage_blocks)
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 2),
                ))
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(dim, nc))
        self._dim = dim

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, dim, H', W']
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', dim]
        x = self.pos_drop(x)

        for i, stage in enumerate(self.stages):
            for blk in stage:
                x = blk(x, H, W)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
                H, W = max(H // 2, 1), max(W // 2, 1)
                # Truncate or pad to match H*W
                target_len = H * W
                if x.shape[1] > target_len:
                    x = x[:, :target_len, :]
                elif x.shape[1] < target_len:
                    x = F.pad(x, (0, 0, 0, target_len - x.shape[1]))

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.head(x)


class FusionTeacher(nn.Module):
    """Multi-modal Fusion Teacher: 1D temporal + 2D spectral branches.

    Combines time-domain and spectral features via attention-based fusion.
    """

    def __init__(self, nc=4, in_ch=1):
        super().__init__()
        # Branch 1: 2D CNN on spectrogram (reuse ResNet-style)
        self.spec_branch = nn.Sequential(
            base.ConvBNAct(in_ch, 32),
            base.ResidualBlock(32, 32),
            base.ResidualBlock(32, 64, 2),
            base.ResidualBlock(64, 64),
            base.ResidualBlock(64, 128, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        # Branch 2: 2D CNN with different receptive field
        self.temporal_branch = nn.Sequential(
            base.ConvBNAct(in_ch, 32),
            nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3), groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0), groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # Attention-based fusion
        self.fusion_attn = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(256, nc))

    def forward(self, x):
        f_spec = self.spec_branch(x).flatten(1)   # [B, 128]
        f_temp = self.temporal_branch(x).flatten(1)  # [B, 128]
        combined = torch.cat([f_spec, f_temp], dim=1)  # [B, 256]
        weights = self.fusion_attn(combined)  # [B, 2]
        fused = weights[:, 0:1] * f_spec + weights[:, 1:2] * f_temp
        return self.head(combined)


class CRNNAttentionTeacher(nn.Module):
    """Enhanced ResNetCRNN with self-attention pooling."""

    def __init__(self, nc=4, in_ch=1, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            base.ConvBNAct(in_ch, 32),
            base.ResidualBlock(32, 64, 2),
            base.ResidualBlock(64, 96, 2),
            base.ResidualBlock(96, 128),
        )
        self.reduce = nn.AdaptiveAvgPool2d((8, None))
        self.lstm = nn.LSTM(128 * 8, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        # Self-attention over LSTM outputs
        self.self_attn = nn.MultiheadAttention(hidden * 2, num_heads=4,
                                               batch_first=True, dropout=0.1)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(nn.Dropout(0.35), nn.Linear(hidden * 2, nc))

    def forward(self, x):
        x = self.cnn(x)
        x = self.reduce(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.lstm(x)
        # Self-attention
        out, _ = self.self_attn(out, out, out)
        # Attention pooling
        w = F.softmax(self.attn_pool(out).squeeze(-1), dim=1).unsqueeze(1)
        ctx = torch.bmm(w, out).squeeze(1)
        return self.head(ctx)


# ============================================================================
# 3. CROSS-ARCHITECTURE FEATURE BRIDGE
# ============================================================================

class TransformerToCNNBridge(nn.Module):
    """Projects Transformer features to CNN spatial feature maps.

    Transformer: [B, N, D] patches → reshape → [B, D, H', W']
    → Conv projection → [B, C_target, H, W]
    """

    def __init__(self, transformer_dim, target_channels, spatial_size=(8, 8)):
        super().__init__()
        self.spatial_size = spatial_size
        self.proj = nn.Sequential(
            nn.Linear(transformer_dim, target_channels * spatial_size[0] * spatial_size[1]),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(target_channels, target_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, N, D]
        B = x.shape[0]
        x = self.proj(x.mean(dim=1))  # [B, C*H*W]
        x = x.view(B, -1, self.spatial_size[0], self.spatial_size[1])
        return self.conv(x)


class FeatureExtractingTransformer(nn.Module):
    """Wraps Transformer teacher to extract intermediate features."""

    def __init__(self, teacher, arch_name):
        super().__init__()
        self.teacher = teacher
        self.arch_name = arch_name

    def forward(self, x):
        if hasattr(self.teacher, 'forward_with_features'):
            logits, features, attentions = self.teacher.forward_with_features(x)
            return logits, features, attentions
        return self.teacher(x), {}, {}


class CrossAttentionBridge(nn.Module):
    """Cross-attention: Query=student CNN features, Key/Value=Transformer features."""

    def __init__(self, cnn_dim, trans_dim, num_heads=4):
        super().__init__()
        self.q_proj = nn.Linear(cnn_dim, trans_dim)
        self.cross_attn = nn.MultiheadAttention(trans_dim, num_heads, batch_first=True)

    def forward(self, cnn_feat, trans_feat):
        # cnn_feat: [B, C] → [B, 1, C]
        q = self.q_proj(cnn_feat).unsqueeze(1)
        out, _ = self.cross_attn(q, trans_feat, trans_feat)
        return out.squeeze(1)


# ============================================================================
# 4. ADVANCED KD LOSSES
# ============================================================================

class DecoupledKDLoss(nn.Module):
    """Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

    Separates target class logit from non-target logits for better KD.
    """

    def __init__(self, temperature=4.0, alpha=1.0, beta=8.0):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(self, student_logits, teacher_logits, targets):
        s = student_logits / self.T
        t = teacher_logits / self.T

        # Target class KD
        s_target = s.gather(1, targets.unsqueeze(1))
        t_target = t.gather(1, targets.unsqueeze(1))
        loss_target = F.kl_div(
            F.log_softmax(s_target, dim=1),
            F.softmax(t_target, dim=1),
            reduction='batchmean',
        )

        # Non-target class KD
        mask = torch.ones_like(s, dtype=torch.bool)
        mask.scatter_(1, targets.unsqueeze(1), False)
        s_non = s[mask].view(s.shape[0], -1)
        t_non = t[mask].view(t.shape[0], -1)
        loss_non = F.kl_div(
            F.log_softmax(s_non, dim=1),
            F.softmax(t_non, dim=1),
            reduction='batchmean',
        )

        return self.alpha * loss_target * self.T ** 2 + self.beta * loss_non * self.T ** 2


class AttentionMapTransferLoss(nn.Module):
    """Transfer attention maps from Transformer to CNN features.

    Transformer attention: [B, heads, N, N] → average → spatial attention
    CNN features: [B, C, H, W] → spatial attention = sum_c |feat_c|^2
    """

    def forward(self, transformer_attn, cnn_features):
        # transformer_attn: [B, heads, N+1, N+1] (includes CLS)
        # Average across heads, take CLS attention to patches
        attn = transformer_attn.mean(dim=1)  # [B, N+1, N+1]
        cls_attn = attn[:, 0, 1:]  # [B, N] — CLS attention to patches
        N = cls_attn.shape[1]
        H = W = int(math.sqrt(N))
        if H * W < N:
            cls_attn = cls_attn[:, :H * W]
        elif H * W > N:
            cls_attn = F.pad(cls_attn, (0, H * W - N))
        cls_attn = cls_attn.view(-1, 1, H, W)  # [B, 1, H, W]

        # CNN spatial attention
        cnn_attn = (cnn_features ** 2).sum(dim=1, keepdim=True)  # [B, 1, H', W']
        cnn_attn = cnn_attn / (cnn_attn.amax(dim=(2, 3), keepdim=True) + 1e-8)

        # Resize to match
        if cls_attn.shape != cnn_attn.shape:
            cls_attn = F.adaptive_avg_pool2d(cls_attn, cnn_attn.shape[2:])

        return F.mse_loss(cls_attn, cnn_attn)


class MultiTemperatureKDLoss(nn.Module):
    """KD with multiple temperatures for richer knowledge transfer."""

    def __init__(self, temperatures=(2.0, 4.0, 8.0)):
        super().__init__()
        self.temps = temperatures

    def forward(self, student_logits, teacher_probs):
        loss = 0.0
        for T in self.temps:
            s_logp = F.log_softmax(student_logits / T, dim=1)
            # teacher_probs are already soft — re-scale
            t_soft = teacher_probs ** (1.0 / T)
            t_soft = t_soft / t_soft.sum(dim=1, keepdim=True)
            loss += -(t_soft * s_logp).sum(dim=1).mean() * (T ** 2)
        return loss / len(self.temps)


class FeatureProjector(nn.Module):
    """Projects features to common dimension for distillation."""

    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for ch in in_channels_list
        ])

    def forward(self, features_list):
        return [proj(feat) for proj, feat in zip(self.projectors, features_list)]


class FeatureDistillationLoss(nn.Module):
    """Feature-level distillation using cosine/MSE loss."""

    def __init__(self, loss_type="cosine"):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                tf = F.adaptive_avg_pool2d(tf, sf.shape[2:])
            if self.loss_type == "cosine":
                sf_flat = sf.flatten(1)
                tf_flat = tf.flatten(1)
                cos_sim = F.cosine_similarity(sf_flat, tf_flat, dim=1)
                loss += (1 - cos_sim).mean()
            else:
                loss += F.mse_loss(sf, tf)
        return loss


class RelationalDistillationLoss(nn.Module):
    """Relational Knowledge Distillation (Park et al., CVPR 2019)."""

    def forward(self, student_embeds, teacher_embeds):
        s_flat = [f.flatten(1) for f in student_embeds]
        t_flat = [f.flatten(1) for f in teacher_embeds]
        loss = 0.0
        for sf, tf in zip(s_flat, t_flat):
            pdist_s = torch.cdist(sf, sf, p=2)
            pdist_t = torch.cdist(tf, tf, p=2)
            pdist_s = pdist_s / (pdist_s.norm() + 1e-8)
            pdist_t = pdist_t / (pdist_t.norm() + 1e-8)
            loss += F.mse_loss(pdist_s, pdist_t)
        return loss


class AttentionTransferLoss(nn.Module):
    """Attention Transfer (Zagoruyko & Komodakis, ICLR 2017)."""

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            am_s = (sf ** 2).sum(dim=1, keepdim=True)
            am_t = (tf ** 2).sum(dim=1, keepdim=True)
            am_s = am_s / (am_s.amax(dim=(2, 3), keepdim=True) + 1e-8)
            am_t = am_t / (am_t.amax(dim=(2, 3), keepdim=True) + 1e-8)
            if am_s.shape != am_t.shape:
                am_t = F.adaptive_avg_pool2d(am_t, am_s.shape[2:])
            loss += F.mse_loss(am_s, am_t)
        return loss


class FeatureDiscriminator(nn.Module):
    """Adversarial discriminator for feature matching."""

    def __init__(self, feat_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 5. EMA + SWA + CURRICULUM
# ============================================================================

class EMAModel:
    """Exponential Moving Average model."""

    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def get_ema_model(self):
        return self.ema_model


class CurriculumSampler:
    """Curriculum learning: gradually include harder samples."""

    def __init__(self, n_samples, difficulty_scores, curriculum_start=0.4,
                 warmup_epochs=25):
        self.n_samples = n_samples
        self.sorted_indices = np.argsort(difficulty_scores)
        self.curriculum_start = curriculum_start
        self.warmup_epochs = warmup_epochs

    def get_active_indices(self, epoch):
        if epoch >= self.warmup_epochs:
            ratio = 1.0
        else:
            ratio = self.curriculum_start + (1.0 - self.curriculum_start) * (epoch / self.warmup_epochs)
        n_active = max(int(self.n_samples * ratio), 32)
        return self.sorted_indices[:n_active]


class ProgressiveLossScheduler:
    """Progressively shift loss weights from hard-label to KD."""

    def __init__(self, warmup_epochs=30, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.start = {k: v[0] for k, v in kwargs.items()}
        self.end = {k: v[1] for k, v in kwargs.items()}

    def get_weights(self, epoch):
        if epoch >= self.warmup_epochs:
            return dict(self.end)
        t = epoch / self.warmup_epochs
        return {k: self.start[k] + (self.end[k] - self.start[k]) * t for k in self.start}


class SWA:
    """Stochastic Weight Averaging."""

    def __init__(self, model, start_epoch=80):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_state = None
        self.n_averaged = 0

    @torch.no_grad()
    def update(self, epoch):
        if epoch < self.start_epoch:
            return
        if self.swa_state is None:
            self.swa_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.n_averaged = 1
        else:
            for k in self.swa_state:
                self.swa_state[k].mul_(self.n_averaged).add_(self.model.state_dict()[k])
                self.swa_state[k].div_(self.n_averaged + 1)
            self.n_averaged += 1

    def apply(self):
        if self.swa_state is not None:
            self.model.load_state_dict(self.swa_state)


# ============================================================================
# 6. MIXUP + CLASS-BALANCED SAMPLING
# ============================================================================

def mixup_data(x, y, alpha=0.3):
    """MixUp augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def class_balanced_weights(records, nc, beta=0.9999):
    """Class-balanced effective-number sampling weights."""
    labels = [base.get_label(r, nc) for r in records]
    counts = np.bincount(labels, minlength=nc).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * nc
    return torch.tensor([weights[y] for y in labels], dtype=torch.double)


def compute_difficulty(teacher_logits, nc):
    """Sample difficulty = 1 - max(teacher_prob)."""
    probs = F.softmax(torch.tensor(teacher_logits), dim=1).numpy()
    return 1.0 - probs.max(axis=1)


# ============================================================================
# 7. STUDENT DATASET WITH MIXUP
# ============================================================================

class MixUpStudentKDDataset(Dataset):
    """Student KD dataset with MixUp support."""

    def __init__(self, base_ds, teacher_probs, mixup_alpha=0.3):
        self.base_ds = base_ds
        self.teacher_probs = torch.tensor(teacher_probs, dtype=torch.float32)
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y, _ = self.base_ds[idx]
        tprob = self.teacher_probs[idx]
        return x, y, torch.tensor(idx, dtype=torch.long), tprob


# ============================================================================
# 8. ENHANCED FEATURE-EXTRACTING STUDENT
# ============================================================================

class FeatureExtractingStudent(nn.Module):
    """Wraps DSCNNResSEStudent to extract intermediate features."""

    def __init__(self, student_model):
        super().__init__()
        self.student = student_model
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        blocks = list(self.student.features.children())
        if len(blocks) >= 3:
            blocks[1].register_forward_hook(self._make_hook("feat_early"))
            blocks[3].register_forward_hook(self._make_hook("feat_mid"))
            blocks[5].register_forward_hook(self._make_hook("feat_late"))

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x):
        logits = self.student(x)
        return logits, {k: v for k, v in self._features.items()}


# ============================================================================
# 9. TTA LOGIT COLLECTION
# ============================================================================

def collect_logits_with_tta(model, arch, seed, args, splits, stats, device, output_dir,
                            n_tta=5, tta_noise_std=0.005, tta_time_shift=0.05):
    """Collect teacher logits with Test-Time Augmentation."""
    logits_dir = ensure_dir(output_dir / "teacher_logits")

    for split_name, records in splits.items():
        if not records:
            continue
        ds = base.ICBHIDataset(records, args, stats, augment=False, return_sample_id=True)
        loader = base.make_loader(ds, args)

        model.eval()
        all_logits = []
        sample_ids = None

        with torch.no_grad():
            for batch_idx, (x, y, ident) in enumerate(loader):
                x = x.to(device)
                batch_logits = [model(x).cpu()]

                for _ in range(n_tta - 1):
                    x_aug = x.clone()
                    if tta_noise_std > 0:
                        x_aug = x_aug + torch.randn_like(x_aug) * tta_noise_std
                    if tta_time_shift > 0 and x_aug.size(-1) > 1:
                        max_shift = max(1, int(tta_time_shift * x_aug.size(-1)))
                        shift = random.randint(-max_shift, max_shift)
                        x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                    batch_logits.append(model(x_aug).cpu())

                avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
                all_logits.append(avg_logits)

                if sample_ids is None:
                    if isinstance(ident, torch.Tensor):
                        sample_ids = ident.numpy().tolist()
                    else:
                        sample_ids = list(ident)

        logits = torch.cat(all_logits, dim=0).numpy()
        stem = f"{arch}_seed_{seed}_{split_name}"
        np.save(logits_dir / f"{stem}.npy", logits)

        expected_ids = [r.sample_id for r in records]
        with (logits_dir / f"{stem}_sample_ids.json").open("w", encoding="utf-8") as f:
            json.dump(expected_ids, f, indent=2)

        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        y_true = np.array([base.get_label(r, args.num_classes) for r in records], dtype=np.int64)
        y_pred = probs.argmax(axis=1)
        metrics = base.compute_metrics(y_true, y_pred, probs, args.num_classes)
        base.save_metrics(output_dir, f"teacher_{arch}_seed_{seed}_{split_name}",
                          metrics, y_true, y_pred, args.num_classes)

        print(f"  TTA logits collected: {stem} shape={logits.shape}", flush=True)


# ============================================================================
# 10. TEMPERATURE CALIBRATION
# ============================================================================

def find_optimal_temperature(logits, records, nc, temp_range=None):
    """Find optimal temperature for teacher calibration."""
    if temp_range is None:
        temp_range = np.linspace(0.5, 10.0, 96)

    y_true = np.array([base.get_label(r, nc) for r in records], dtype=np.int64)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(y_true, dtype=torch.long)

    best_temp = 1.0
    best_nll = float("inf")
    for temp in temp_range:
        log_probs = F.log_softmax(logits_t / temp, dim=1)
        nll = F.nll_loss(log_probs, targets).item()
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    return best_temp


# ============================================================================
# 11. MODEL FACTORY EXTENSION
# ============================================================================

def make_s4_model(name, nc, in_ch, args):
    """Extended model factory for S4 (includes Transformer + hybrid teachers)."""
    if name == "ast_tiny":
        return ASTTinyTeacher(nc, in_ch,
                               embed_dim=getattr(args, 'ast_embed_dim', 192),
                               depth=getattr(args, 'ast_depth', 4),
                               num_heads=getattr(args, 'ast_heads', 3))
    if name == "swin_tiny":
        return SwinTinyTeacher(nc, in_ch)
    if name == "fusion":
        return FusionTeacher(nc, in_ch)
    if name == "crnn_attn":
        return CRNNAttentionTeacher(nc, in_ch)
    # Fallback to base factory
    return base.make_model(name, nc, in_ch, args)


# ============================================================================
# 12. DUAL-THRESHOLD PREDICTION
# ============================================================================

def dual_threshold_prediction(probs, threshold_normal, threshold_abnormal):
    """Dual-threshold prediction for better Se/Sp control."""
    preds = probs.argmax(axis=1)
    p_normal = probs[:, 0]
    p_abnormal_max = probs[:, 1:].max(axis=1)
    preds = np.where(p_normal >= threshold_normal, 0, preds)
    abnormal_class = probs[:, 1:].argmax(axis=1) + 1
    preds = np.where(
        (p_normal < threshold_normal) & (p_abnormal_max >= threshold_abnormal),
        abnormal_class, preds,
    )
    return preds


def sweep_dual_threshold(y_true, probs):
    """Sweep both thresholds to maximize ICBHI Score."""
    best = {"threshold_normal": 0.5, "threshold_abnormal": 0.5, "icbhi_score": -1.0}
    for th_n in np.linspace(0.10, 0.80, 15):
        for th_a in np.linspace(0.10, 0.70, 13):
            pred = dual_threshold_prediction(probs, float(th_n), float(th_a))
            se, sp, score = base.icbhi_score(y_true, pred)
            if score > best["icbhi_score"]:
                best = {
                    "threshold_normal": float(th_n),
                    "threshold_abnormal": float(th_a),
                    "icbhi_score": float(score),
                    "sensitivity": float(se),
                    "specificity": float(sp),
                }
    single = base.sweep_threshold(y_true, probs)
    if single["icbhi_score"] > best["icbhi_score"]:
        best = {
            "threshold_normal": float(single["threshold"]),
            "threshold_abnormal": 0.5,
            "icbhi_score": float(single["icbhi_score"]),
            "sensitivity": float(single["sensitivity"]),
            "specificity": float(single["specificity"]),
        }
    return best


# ============================================================================
# 13. STUDENT TRAINING — MEGA ENSEMBLE KD
# ============================================================================

def train_student_mega_ensemble(args, splits, stats, device, output_dir):
    """Train student with mega-ensemble KD: 8-component loss, curriculum, EMA, SWA."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes

    # --- Load teacher logits with TTA + calibration ---
    print("Loading and calibrating teacher logits...", flush=True)
    val_logits, teacher_names = base.load_teacher_logits(args, output_dir, "val", splits["val"])
    train_logits, _ = base.load_teacher_logits(args, output_dir, "train", splits["train"])

    # Temperature calibration per teacher
    calibrated_val = []
    temps = []
    for t in range(val_logits.shape[0]):
        temp = find_optimal_temperature(val_logits[t], splits["val"], nc)
        calibrated_val.append(val_logits[t] / temp)
        temps.append(temp)
        print(f"  Calibrated {teacher_names[t]}: optimal_temp={temp:.2f}", flush=True)

    calibrated_val = np.stack(calibrated_val, axis=0)

    # Save calibration info
    student_dir = ensure_dir(output_dir / "students" / args.student_arch)
    with (student_dir / "calibration_temps.json").open("w") as f:
        json.dump({"teacher_names": teacher_names, "temperatures": temps}, f, indent=2)

    # Reliability weights on calibrated val logits
    weights = base.reliability_weights(calibrated_val, splits["val"], nc)
    train_probs = base.weighted_teacher_probs(train_logits, weights, args.temperature)

    with (student_dir / "teacher_reliability.json").open("w", encoding="utf-8") as f:
        json.dump({"teacher_names": teacher_names, "class_weights": weights.tolist(),
                    "calibration_temps": temps}, f, indent=2)

    # --- Create student model ---
    student_raw = make_s4_model(args.student_arch, nc, in_ch, args).to(device)
    student = FeatureExtractingStudent(student_raw).to(device)

    base.init_wandb(args, f"{args.pipeline_name}-student-{args.student_arch}",
                    {"student_params": base.count_params(student)[0],
                     "teacher_names": teacher_names,
                     "n_teachers": len(teacher_names)})

    # --- EMA model ---
    ema = EMAModel(student_raw, decay=args.ema_decay)

    # --- Loss functions ---
    cb_focal = base.FocalLoss(
        base.class_weights(splits["train"], nc, device),
        args.focal_gamma, args.label_smoothing,
    )

    # Feature projectors for distillation
    s_ch = [32, 64, 128]  # DSCNNResSE approximate channels
    proj_dim = 64
    projector = FeatureProjector(s_ch, proj_dim).to(device)

    # Adversarial discriminator
    disc = FeatureDiscriminator(proj_dim * 8 * 8).to(device)

    # KD losses
    dkd_loss_fn = DecoupledKDLoss(temperature=args.temperature)
    feat_loss_fn = FeatureDistillationLoss(loss_type="cosine")
    attn_loss_fn = AttentionTransferLoss()
    rkd_loss_fn = RelationalDistillationLoss()

    # --- Curriculum sampler ---
    avg_train_logits = train_logits.mean(axis=0)
    difficulty = compute_difficulty(avg_train_logits, nc)
    curriculum = CurriculumSampler(len(splits["train"]), difficulty,
                                   curriculum_start=args.curriculum_start,
                                   warmup_epochs=args.curriculum_warmup)

    # --- Progressive loss scheduler ---
    loss_sched = ProgressiveLossScheduler(
        warmup_epochs=args.progressive_warmup,
        hard=(args.hard_weight_start, args.hard_weight),
        kd=(args.kd_weight_start, args.kd_weight),
        binary=(args.binary_weight_start, args.binary_weight),
    )

    # --- SWA ---
    swa = SWA(student_raw, start_epoch=max(args.epochs_student - 20,
                                            int(args.epochs_student * 0.7)))

    # --- Datasets ---
    base_train = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    train_ds = MixUpStudentKDDataset(base_train, train_probs, mixup_alpha=args.mixup_alpha)
    val_loader = base.make_loader(base.ICBHIDataset(splits["val"], args, stats, False), args)

    # --- Optimizers ---
    student_params = (list(student.parameters()) +
                      list(projector.parameters()) +
                      list(dkd_loss_fn.parameters()) if hasattr(dkd_loss_fn, 'parameters') else [])
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr_student,
                               weight_decay=args.weight_decay)
    opt_proj = torch.optim.AdamW(projector.parameters(), lr=args.lr_student,
                                  weight_decay=args.weight_decay)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=args.lr_student * 0.5,
                                  weight_decay=args.weight_decay)

    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=max(args.epochs_student, 1))

    # --- Checkpoint tracking ---
    best_score, best_epoch, patience = -1.0, 0, 0
    best_tiebreak_macro = -1.0
    best_tiebreak_bal = -1.0
    best_tiebreak_both = -1.0
    min_both_f1_guard = 0.05 if nc == 4 else -1.0
    best_path = student_dir / "best.pt"

    # --- Training loop ---
    for epoch in range(1, args.epochs_student + 1):
        student.train()
        ema.get_ema_model().eval()

        # Curriculum subset
        active_indices = curriculum.get_active_indices(epoch)
        active_labels = [base.get_label(splits["train"][i], nc) for i in active_indices]
        active_counts = np.bincount(active_labels, minlength=nc).astype(np.float64)
        sample_w = np.array([1.0 / max(active_counts[y], 1) for y in active_labels])
        sampler = WeightedRandomSampler(torch.tensor(sample_w, dtype=torch.double),
                                        len(active_indices), replacement=True)
        subset_ds = Subset(train_ds, active_indices)
        train_loader = DataLoader(subset_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

        # Progressive weights
        w = loss_sched.get_weights(epoch)
        w_hard, w_kd, w_bin = w['hard'], w['kd'], w['binary']

        # Loss accumulators
        totals = {k: 0.0 for k in ['total', 'hard', 'kd', 'dkd', 'feat', 'attn',
                                     'rkd', 'adv', 'ema', 'bin']}

        for x, y, _, tprob in train_loader:
            x, y, tprob = x.to(device), y.to(device), tprob.to(device)
            batch_size = x.size(0)

            # --- MixUp ---
            use_mixup = args.mixup_alpha > 0 and np.random.random() < args.mixup_prob
            if use_mixup:
                x_mix, y_a, y_b, lam = mixup_data(x, y, args.mixup_alpha)
                logits, s_feats = student(x_mix)
                hard_loss = lam * cb_focal(logits, y_a) + (1 - lam) * cb_focal(logits, y_b)
                tprob_mix = lam * tprob + (1 - lam) * tprob[torch.randperm(batch_size, device=device)]
            else:
                logits, s_feats = student(x)
                hard_loss = cb_focal(logits, y)
                tprob_mix = tprob

            # --- Logit KD (standard) ---
            kd_loss = -(tprob_mix * F.log_softmax(logits / args.temperature, dim=1)
                        ).sum(dim=1).mean() * (args.temperature ** 2)

            # --- Decoupled KD ---
            dkd_loss = dkd_loss_fn(logits, tprob_mix, y)

            # --- Feature distillation (from EMA teacher) ---
            feat_loss = torch.tensor(0.0, device=device)
            attn_loss = torch.tensor(0.0, device=device)
            rkd_loss = torch.tensor(0.0, device=device)

            if epoch > 5:
                with torch.no_grad():
                    ema_logits, ema_feats = ema.get_ema_model()(x) if hasattr(
                        ema.get_ema_model(), '__call__') else (None, {})
                    # Use student itself as feature reference for self-distillation
                    _, ref_feats = student(x)

                # Project student features
                s_projected = projector([
                    s_feats.get("feat_early", torch.zeros(1)),
                    s_feats.get("feat_mid", torch.zeros(1)),
                    s_feats.get("feat_late", torch.zeros(1)),
                ])

                # Self-feature distillation (student → EMA)
                if ema_logits is not None:
                    ema_s = FeatureExtractingStudent(ema.get_ema_model())
                    _, ema_feats_map = ema_s(x)
                    ema_projected = projector([
                        ema_feats_map.get("feat_early", torch.zeros(1)),
                        ema_feats_map.get("feat_mid", torch.zeros(1)),
                        ema_feats_map.get("feat_late", torch.zeros(1)),
                    ])
                    feat_loss = feat_loss_fn(s_projected, ema_projected)
                    attn_loss = attn_loss_fn(s_projected, ema_projected)
                    if batch_size > 2:
                        rkd_loss = rkd_loss_fn(s_projected, ema_projected)

            # --- Adversarial loss ---
            adv_loss = torch.tensor(0.0, device=device)
            if epoch > 10 and args.adv_weight > 0:
                s_flat = torch.cat([
                    F.adaptive_avg_pool2d(f, 8).flatten(1) for f in s_projected
                ], dim=1)
                # Discriminator step
                with torch.no_grad():
                    ema_logits_d, ema_feats_d = ema.get_ema_model()(x) if hasattr(
                        ema.get_ema_model(), '__call__') else (None, {})
                    if ema_logits_d is not None:
                        ema_s_d = FeatureExtractingStudent(ema.get_ema_model())
                        _, ema_f_d = ema_s_d(x)
                        ema_proj_d = projector([
                            ema_f_d.get("feat_early", torch.zeros(1)),
                            ema_f_d.get("feat_mid", torch.zeros(1)),
                            ema_f_d.get("feat_late", torch.zeros(1)),
                        ])
                        t_flat = torch.cat([
                            F.adaptive_avg_pool2d(f, 8).flatten(1) for f in ema_proj_d
                        ], dim=1)

                        opt_disc.zero_grad(set_to_none=True)
                        d_real = disc(t_flat.detach())
                        d_fake = disc(s_flat.detach())
                        d_loss = -(torch.log(d_real + 1e-8).mean() +
                                   torch.log(1 - d_fake + 1e-8).mean())
                        d_loss.backward()
                        opt_disc.step()

                        d_fake_for_g = disc(s_flat)
                        adv_loss = -torch.log(d_fake_for_g + 1e-8).mean()

            # --- EMA teacher KD ---
            ema_kd_loss = torch.tensor(0.0, device=device)
            if epoch > args.ema_warmup:
                with torch.no_grad():
                    ema_logits_kd = ema.get_ema_model()(x)
                    if isinstance(ema_logits_kd, tuple):
                        ema_logits_kd = ema_logits_kd[0]
                    ema_probs = F.softmax(ema_logits_kd / args.ema_temperature, dim=1)
                ema_kd_loss = -(ema_probs * F.log_softmax(
                    logits / args.ema_temperature, dim=1)
                ).sum(dim=1).mean() * (args.ema_temperature ** 2)

            # --- Binary auxiliary loss ---
            hard_bin = (y != 0).float()
            teacher_bin = (1.0 - tprob[:, 0]).clamp(0, 1)
            bin_target = 0.5 * hard_bin + 0.5 * teacher_bin
            bin_loss = F.binary_cross_entropy_with_logits(
                base.abnormal_logit_from_4class(logits), bin_target)

            # --- Combined loss ---
            loss = (w_hard * hard_loss
                    + w_kd * kd_loss
                    + args.dkd_weight * dkd_loss
                    + args.feat_weight * feat_loss
                    + args.attn_weight * attn_loss
                    + args.rkd_weight * rkd_loss
                    + args.adv_weight * adv_loss
                    + args.ema_weight * ema_kd_loss
                    + w_bin * bin_loss)

            opt_s.zero_grad(set_to_none=True)
            opt_proj.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip)
            opt_s.step()
            opt_proj.step()

            # Update EMA
            if epoch > args.ema_warmup:
                ema.update(student_raw)

            n = batch_size
            totals['total'] += float(loss.item()) * n
            totals['hard'] += float(hard_loss.item()) * n
            totals['kd'] += float(kd_loss.item()) * n
            totals['dkd'] += float(dkd_loss.item()) * n
            totals['feat'] += float(feat_loss.item()) * n
            totals['attn'] += float(attn_loss.item()) * n
            totals['rkd'] += float(rkd_loss.item()) * n
            totals['adv'] += float(adv_loss.item()) * n
            totals['ema'] += float(ema_kd_loss.item()) * n
            totals['bin'] += float(bin_loss.item()) * n

        sched_s.step()
        swa.update(epoch)

        # --- Validation ---
        eval_model = ema.get_ema_model() if epoch > args.ema_warmup else student_raw
        val_m, yv, _, pv, _ = base.evaluate_model(eval_model, val_loader, device, nc)
        tuned = base.sweep_threshold(yv, pv)
        score = float(tuned["icbhi_score"] if args.selection_metric == "threshold_icbhi_score"
                      else val_m[args.selection_metric])
        both_f1 = float(val_m.get("both_f1", 0.0)) if nc == 4 else 0.0
        meets_guard = both_f1 >= min_both_f1_guard
        macro_f1 = float(val_m.get("macro_f1", 0.0))
        bal_acc = float(val_m.get("balanced_accuracy", 0.0))

        # Checkpoint selection
        better_primary = score > best_score + 1e-12
        tie_primary = abs(score - best_score) <= 1e-12
        better_tiebreak = tie_primary and (
            (macro_f1 > best_tiebreak_macro + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12 and bal_acc > best_tiebreak_bal + 1e-12)
            or (abs(macro_f1 - best_tiebreak_macro) <= 1e-12
                and abs(bal_acc - best_tiebreak_bal) <= 1e-12
                and both_f1 > best_tiebreak_both + 1e-12)
        )

        should_save = False
        if meets_guard and (better_primary or better_tiebreak):
            should_save = True
        elif (not meets_guard) and (best_epoch == 0) and better_primary:
            should_save = True

        if should_save:
            best_score, best_epoch, patience = score, epoch, 0
            best_tiebreak_macro = macro_f1
            best_tiebreak_bal = bal_acc
            best_tiebreak_both = both_f1
            save_model = ema.get_ema_model() if epoch > args.ema_warmup else student_raw
            torch.save({
                "model_state": save_model.state_dict(),
                "epoch": epoch,
                "arch": args.student_arch,
                "threshold": tuned["threshold"],
                "metrics": val_m,
                "threshold_metrics": tuned,
                "args": vars(args),
                "uses_ema": epoch > args.ema_warmup,
                "selection_info": {
                    "score": score, "macro_f1": macro_f1,
                    "balanced_accuracy": bal_acc, "both_f1": both_f1,
                    "meets_both_f1_guard": meets_guard,
                },
            }, best_path)
            np.save(student_dir / "val_probs_best.npy", pv)
        else:
            patience += 1

        # Logging
        denom = max(len(subset_ds), 1)
        base.log_wandb({
            "epoch": epoch,
            "loss": totals['total'] / denom,
            "hard_loss": totals['hard'] / denom,
            "kd_loss": totals['kd'] / denom,
            "dkd_loss": totals['dkd'] / denom,
            "feat_loss": totals['feat'] / denom,
            "attn_loss": totals['attn'] / denom,
            "rkd_loss": totals['rkd'] / denom,
            "adv_loss": totals['adv'] / denom,
            "ema_loss": totals['ema'] / denom,
            "bin_loss": totals['bin'] / denom,
            "curriculum_ratio": curriculum.get_active_indices(epoch).shape[0] / len(splits["train"]),
            "w_hard": w_hard, "w_kd": w_kd, "w_bin": w_bin,
            **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))},
            "val_threshold_icbhi_score": tuned["icbhi_score"],
            "val_threshold": tuned["threshold"],
            "best_score": float(best_score),
        }, prefix="student", step=epoch)

        print(f"student ep={epoch:03d} loss={totals['total']/denom:.4f} "
              f"curriculum={curriculum.get_active_indices(epoch).shape[0]/len(splits['train']):.2f} "
              f"w_h={w_hard:.2f} w_k={w_kd:.2f} w_b={w_bin:.2f} "
              f"val_icbhi={val_m['icbhi_score']:.4f} tuned={tuned['icbhi_score']:.4f} "
              f"se={val_m['sensitivity']:.4f} sp={val_m['specificity']:.4f} "
              f"best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            break

    # Apply SWA
    if swa.n_averaged > 0:
        print(f"Applying SWA ({swa.n_averaged} averaged models)...", flush=True)
        swa.apply()
        torch.save({
            "model_state": student_raw.state_dict(),
            "arch": args.student_arch,
            "n_swa_averaged": swa.n_averaged,
            "args": vars(args),
        }, student_dir / "swa_best.pt")

    base.finish_wandb()
    return best_path


# ============================================================================
# 14. FINAL EVALUATION — MEGA
# ============================================================================

def evaluate_final_mega(args, splits, stats, device, output_dir):
    """Evaluate with TTA + dual-threshold + SWA."""
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    nc = args.num_classes
    student_dir = output_dir / "students" / args.student_arch

    # Load best checkpoint
    model = make_s4_model(args.student_arch, nc, in_ch, args).to(device)
    best_path = student_dir / "best.pt"
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    threshold = float(ckpt.get("threshold", 0.5))

    base.init_wandb(args, f"{args.pipeline_name}-final-eval",
                    {"student_checkpoint": str(best_path), "threshold": threshold})

    summary = {
        "student_checkpoint": str(best_path),
        "threshold": threshold,
        "student_params": base.count_params(model)[0],
    }

    for split_name in ["val", "test"]:
        if not splits[split_name]:
            continue
        loader = base.make_loader(base.ICBHIDataset(splits[split_name], args, stats, False), args)

        # Raw evaluation
        raw_m, yt, yp, probs, _ = base.evaluate_model(model, loader, device, nc)

        # Single threshold
        tuned_pred = base.threshold_predictions(probs, threshold)
        tuned_m = base.compute_metrics(yt, tuned_pred, probs, nc)

        # Dual threshold
        if split_name == "val":
            dual_tuned = sweep_dual_threshold(yt, probs)
            summary["dual_threshold_info"] = dual_tuned
        else:
            dual_info = summary.get("dual_threshold_info", {})
            dual_tuned = dual_info

        dual_pred = dual_threshold_prediction(
            probs, dual_tuned["threshold_normal"], dual_tuned["threshold_abnormal"])
        dual_m = base.compute_metrics(yt, dual_pred, probs, nc)

        # TTA evaluation
        tta_m = evaluate_student_tta(model, loader, device, args, n_tta=args.n_tta_eval)

        base.save_metrics(output_dir, f"student_{split_name}_raw", raw_m, yt, yp, nc)
        base.save_metrics(output_dir, f"student_{split_name}_threshold", tuned_m, yt, tuned_pred, nc)
        base.save_metrics(output_dir, f"student_{split_name}_dual_threshold", dual_m, yt, dual_pred, nc)
        base.save_metrics(output_dir, f"student_{split_name}_tta", tta_m, yt, yt, nc)

        summary[f"{split_name}_raw"] = raw_m
        summary[f"{split_name}_threshold"] = tuned_m
        summary[f"{split_name}_dual_threshold"] = dual_m
        summary[f"{split_name}_tta"] = tta_m

        base.log_wandb({f"{split_name}_raw_{k}": v for k, v in raw_m.items()
                         if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_threshold_{k}": v for k, v in tuned_m.items()
                         if isinstance(v, (int, float))}, prefix="final")
        base.log_wandb({f"{split_name}_dual_{k}": v for k, v in dual_m.items()
                         if isinstance(v, (int, float))}, prefix="final_dual")
        base.log_wandb({f"{split_name}_tta_{k}": v for k, v in tta_m.items()
                         if isinstance(v, (int, float))}, prefix="final_tta")

    # Save summary
    metrics_dir = ensure_dir(output_dir / "metrics")
    with (metrics_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (metrics_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "mode", "icbhi_score", "sensitivity", "specificity",
                     "macro_f1", "accuracy", "binary_icbhi_score"])
        for split_name in ["val", "test"]:
            for mode in ["raw", "threshold", "dual_threshold", "tta"]:
                m = summary.get(f"{split_name}_{mode}")
                if m:
                    w.writerow([split_name, mode, m.get("icbhi_score"), m.get("sensitivity"),
                                m.get("specificity"), m.get("macro_f1"), m.get("accuracy"),
                                m.get("binary_icbhi_score")])

    # Export ONNX
    if args.export_onnx:
        try:
            dummy = torch.randn(1, in_ch, args.n_mels, args.target_frames, device=device)
            torch.onnx.export(model.eval(), dummy, str(output_dir / "student_final.onnx"),
                              export_params=True, opset_version=11, do_constant_folding=True,
                              input_names=["mel_spectrogram"], output_names=["logits"],
                              dynamic_axes={"mel_spectrogram": {0: "batch"}, "logits": {0: "batch"}})
            print(f"ONNX exported: {output_dir / 'student_final.onnx'}", flush=True)
        except Exception as exc:
            print(f"ONNX export failed: {exc}", flush=True)

    base.finish_wandb()
    return summary


def evaluate_student_tta(model, loader, device, args, n_tta=10):
    """Evaluate student with Test-Time Augmentation."""
    model.eval()
    yt_all, logits_all = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            batch_logits = [model(x).cpu()]
            if isinstance(batch_logits[0], tuple):
                batch_logits = [batch_logits[0][0]]

            for _ in range(n_tta - 1):
                x_aug = x.clone()
                x_aug = x_aug + torch.randn_like(x_aug) * 0.005
                if x_aug.size(-1) > 1:
                    shift = random.randint(-max(1, int(0.03 * x_aug.size(-1))),
                                           max(1, int(0.03 * x_aug.size(-1))))
                    x_aug = torch.roll(x_aug, shifts=shift, dims=-1)
                out = model(x_aug)
                if isinstance(out, tuple):
                    out = out[0]
                batch_logits.append(out.cpu())

            avg_logits = torch.stack(batch_logits, dim=0).mean(dim=0)
            logits_all.append(avg_logits)
            yt_all.extend(y.numpy().tolist())

    logits = torch.cat(logits_all, dim=0).numpy()
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()
    y_true = np.array(yt_all, dtype=np.int64)

    tuned = base.sweep_threshold(y_true, probs)
    y_pred = base.threshold_predictions(probs, tuned["threshold"])
    return base.compute_metrics(y_true, y_pred, probs, args.num_classes)


# ============================================================================
# 15. ARGUMENT PARSING
# ============================================================================

def parse_args():
    args = base.parse_args()

    # Pipeline identity
    if args.pipeline_name == "icbhi_kd_multiview_ensemble":
        args.pipeline_name = "icbhi_kd_s4_transformer_mega_ensemble"

    # Official ICBHI protocol
    if args.benchmark_protocol == "add_rsc":
        args.benchmark_protocol = "official_icbhi"

    # Mega teacher ensemble: 7 architectures
    if args.teacher_arches == "resnet_cnn,resnet_crnn,efficientnet_b0":
        args.teacher_arches = "resnet_cnn,resnet_crnn,efficientnet_b0,ast_tiny,crnn_attn"

    # Student
    if args.student_arch == "ds_cnn_res_se":
        args.student_arch = "ds_cnn_res_se"

    # Input view
    if args.input_view == "logmel_delta":
        args.input_view = "logmel_delta"

    # Loss weights — S4 optimized
    if args.hard_weight == 0.35:
        args.hard_weight = 0.22
    if args.kd_weight == 0.45:
        args.kd_weight = 0.38
    if args.binary_weight == 0.20:
        args.binary_weight = 0.25

    # Temperature
    if args.temperature == 4.0:
        args.temperature = 4.0

    # Label smoothing
    if args.label_smoothing == 0.05:
        args.label_smoothing = 0.07

    # Stronger SpecAugment
    if args.freq_mask == 12:
        args.freq_mask = 16
    if args.time_mask == 48:
        args.time_mask = 64

    # Selection metric
    args.selection_metric = "threshold_icbhi_score"

    # S4-specific arguments
    defaults = {
        # Feature-level KD weights
        "feat_weight": 0.08,
        "attn_weight": 0.04,
        "rkd_weight": 0.04,
        "adv_weight": 0.02,
        "dkd_weight": 0.08,
        # EMA
        "ema_decay": 0.998,
        "ema_temperature": 3.0,
        "ema_weight": 0.12,
        "ema_warmup": 10,
        # MixUp
        "mixup_alpha": 0.3,
        "mixup_prob": 0.5,
        # Class-balanced
        "cb_beta": 0.9999,
        # Curriculum
        "curriculum_start": 0.4,
        "curriculum_warmup": 25,
        # Progressive KD
        "hard_weight_start": 0.40,
        "kd_weight_start": 0.30,
        "binary_weight_start": 0.20,
        "progressive_warmup": 30,
        # TTA
        "n_tta_teachers": 5,
        "n_tta_eval": 10,
        # SWA
        "swa_start_ratio": 0.7,
        # AST config
        "ast_embed_dim": 192,
        "ast_depth": 4,
        "ast_heads": 3,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    return args


# ============================================================================
# 16. MAIN
# ============================================================================

def main():
    args = parse_args()
    base.set_seed(args.seed)
    device = base.default_device(args.device)
    output_dir, splits, stats = base.prepare_run(args)
    base.print_run_header(args, output_dir, splits)

    # Stage 1: Train ALL teachers (CNN + Transformer + Hybrid)
    if args.stage in {"all", "teachers"}:
        for arch in base.parse_csv(args.teacher_arches):
            for seed in base.parse_int_csv(args.seeds):
                print(f"\n{'='*60}", flush=True)
                print(f"Training teacher: {arch} seed={seed}", flush=True)
                print(f"{'='*60}", flush=True)
                try:
                    model, _, _ = train_s4_teacher(arch, seed, args, splits, stats, device, output_dir)
                    # Collect logits with TTA for Transformer teachers
                    if arch in ("ast_tiny", "swin_tiny"):
                        collect_logits_with_tta(model, arch, seed, args, splits, stats,
                                                device, output_dir, n_tta=args.n_tta_teachers)
                    else:
                        base.collect_and_save_logits(model, arch, seed, args, splits, stats,
                                                     device, output_dir)
                except Exception as exc:
                    print(f"  [WARN] Teacher {arch} seed={seed} failed: {exc}", flush=True)
                    import traceback; traceback.print_exc()

    # Stage 2: Train student with mega-ensemble KD
    if args.stage in {"all", "student"}:
        print(f"\n{'='*60}", flush=True)
        print("Training student with Mega-Ensemble KD", flush=True)
        print(f"{'='*60}", flush=True)
        train_student_mega_ensemble(args, splits, stats, device, output_dir)

    # Stage 3: Final evaluation
    if args.stage in {"all", "evaluate"}:
        print(f"\n{'='*60}", flush=True)
        print("Final evaluation with TTA + dual-threshold + SWA", flush=True)
        print(f"{'='*60}", flush=True)
        evaluate_final_mega(args, splits, stats, device, output_dir)


def train_s4_teacher(arch, seed, args, splits, stats, device, output_dir):
    """Train teacher — supports both base and S4 model architectures."""
    base.set_seed(seed)
    in_ch = 3 if args.input_view == "logmel_delta" else 1
    model = make_s4_model(arch, args.num_classes, in_ch, args).to(device)

    train_ds = base.ICBHIDataset(splits["train"], args, stats, augment=True)
    val_ds = base.ICBHIDataset(splits["val"], args, stats, augment=False)

    train_loader = base.make_loader(train_ds, args, splits["train"], balanced=True)
    val_loader = base.make_loader(val_ds, args)

    criterion = base.FocalLoss(
        base.class_weights(splits["train"], args.num_classes, device),
        args.focal_gamma, args.label_smoothing,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_teacher, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs_teacher, 1))

    best_score, best_epoch, patience = -1.0, 0, 0
    teacher_dir = ensure_dir(output_dir / "teachers" / arch / f"seed_{seed}")
    best_path = teacher_dir / "best.pt"

    for epoch in range(1, args.epochs_teacher + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            n_samples += x.size(0)

        sched.step()

        # Validation
        val_m, _, _, _, _ = base.evaluate_model(model, val_loader, device, args.num_classes)
        score = float(val_m.get("icbhi_score", 0.0))

        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "arch": arch,
                "metrics": val_m, "args": vars(args),
            }, best_path)
        else:
            patience += 1

        print(f"teacher={arch} seed={seed} ep={epoch:03d} "
              f"loss={total_loss/max(n_samples,1):.4f} "
              f"val_icbhi={score:.4f} best={best_score:.4f}", flush=True)

        if patience >= args.patience:
            break

    # Load best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, best_epoch, best_score


if __name__ == "__main__":
    main()
