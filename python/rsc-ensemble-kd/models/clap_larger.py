"""
Wrapper for laion/larger_clap_general — same interface as models/clap.py
but loads the larger CLAP model trained on ~2.5M audio-text pairs.

This module is backward-compatible: it reuses the same cached data
(training.pt / test.pt) because the processor format is identical.

Usage (standalone, does NOT modify main.py):
    python main_larger_clap.py --tag BTS_larger_clap ...
"""
from transformers import ClapModel, ClapAudioModelWithProjection
import torch
import torch.nn as nn


class PretrainedCLAPWithProjection(nn.Module):
    """Audio-only CLAP from larger_clap_general (512-dim output)."""

    def __init__(self, pretrained_name, final_feat_dim):
        super().__init__()
        self.pretrained = pretrained_name
        self.audio_features = ClapAudioModelWithProjection.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim

    def forward(self, x, args=None, alpha=None, training=False):
        x = self.audio_features(x)
        return x.audio_embeds


class PretrainedCLAP(nn.Module):
    """Text+Audio CLAP from larger_clap_general or fused variants."""

    def __init__(self, pretrained_name, final_feat_dim):
        super().__init__()
        self.pretrained = pretrained_name
        self.audio_features = ClapModel.from_pretrained(pretrained_name)
        self.final_feat_dim = final_feat_dim
        # Fused models require is_longer parameter
        self._needs_is_longer = 'fused' in pretrained_name

    def forward(self, x, args=None, alpha=None, training=False):
        text_inputs, attention_mask, audio_inputs = x
        kwargs = dict(
            input_ids=text_inputs,
            attention_mask=attention_mask,
            input_features=audio_inputs,
        )
        if self._needs_is_longer:
            kwargs['is_longer'] = torch.zeros(audio_inputs.shape[0], dtype=torch.bool, device=audio_inputs.device)
        x = self.audio_features(**kwargs)
        text_embeds = x.text_embeds
        audio_embeds = x.audio_embeds
        if args.clap_final == 'concat':
            return torch.cat((text_embeds, audio_embeds), dim=-1)
        elif args.clap_final == 'add':
            return (text_embeds * args.te_alpha) + (audio_embeds * (1 - args.te_alpha))
