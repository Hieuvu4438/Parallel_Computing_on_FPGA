"""
BEATs: Audio Pre-Training with Acoustic Tokenizers (Microsoft, ICML 2023)

Uses the original BEATs TransformerEncoder for proper weight loading.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import logging

# Add util to path for BEATs backbone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'util'))
from beats_backbone import TransformerEncoder

logger = logging.getLogger(__name__)


def find_beats_checkpoint():
    """Find BEATs checkpoint in cache."""
    hf_cache = glob.glob(os.path.expanduser(
        '~/.cache/beats/models--WeiChihChen--*/snapshots/*/*.pt'
    ))
    if hf_cache:
        return hf_cache[0]
    return None


class BEATsConfig:
    """BEATs configuration from checkpoint."""
    def __init__(self, cfg):
        self.__dict__.update(cfg)


class BEATs(nn.Module):
    """
    BEATs model for ICBHI classification.
    Loads pretrained BEATs and adds classification head.
    """

    def __init__(self, checkpoint_name=None, label_dim=4, freeze_encoder=False):
        super().__init__()

        ckpt_path = find_beats_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError("BEATs checkpoint not found")

        print(f"Loading BEATs from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        cfg = checkpoint.get('cfg', {})
        self.cfg = BEATsConfig(cfg)

        # Build model using original BEATs architecture
        self.embed = cfg.get('embed_dim', 512)
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.get('encoder_embed_dim', 768))
            if self.embed != cfg.get('encoder_embed_dim', 768)
            else None
        )

        self.input_patch_size = cfg.get('input_patch_size', 16)
        self.patch_embedding = nn.Conv2d(
            1, self.embed,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=cfg.get('conv_bias', False)
        )

        self.dropout_input = nn.Dropout(cfg.get('dropout_input', 0.0))
        self.encoder = TransformerEncoder(self.cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        # Classification head
        self.final_feat_dim = cfg.get('encoder_embed_dim', 768)
        self.predictor_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.final_feat_dim, label_dim)

        # Load pretrained weights
        state_dict = checkpoint['model']
        # Remove predictor keys (we add our own classifier)
        model_state = self.state_dict()
        pretrained_state = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                pretrained_state[k] = v

        missing, unexpected = self.load_state_dict(pretrained_state, strict=False)
        print(f"  Loaded {len(pretrained_state)}/{len(state_dict)} keys")
        if missing:
            print(f"  Missing: {len(missing)} keys (classifier + some encoder)")
        if unexpected:
            print(f"  Unexpected: {len(unexpected)} keys")

        # Freeze encoder
        if freeze_encoder:
            for name, param in self.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"  Encoder frozen. Trainable: {trainable:,}/{total:,}")

        print(f"BEATs loaded. Feature dim: {self.final_feat_dim}")

    def forward(self, x, args=None, alpha=None, training=False):
        """
        Args:
            x: [B, 1, n_mels, n_frames] mel-spectrogram
        Returns:
            features: [B, final_feat_dim]
        """
        # x: [B, 1, n_mels, n_frames] -> [B, 1, T, F]
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, T, F]

        # BEATs expects raw waveform, but we have mel-spectrogram
        # We'll use the mel-spectrogram directly through patch embedding
        x = x.unsqueeze(1)  # [B, 1, T, F]

        # Patch embedding
        features = self.patch_embedding(x)  # [B, embed, T', F']
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)  # [B, T', embed]
        features = self.layer_norm(features)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        # Transformer encoding
        x, _ = self.encoder(x)  # [B, T', encoder_embed_dim]

        # Pool over time
        x = x.mean(dim=1)  # [B, encoder_embed_dim]

        return x

    def classify(self, features):
        """Classify features to logits."""
        x = self.predictor_dropout(features)
        return self.classifier(x)


def get_beats_class():
    return BEATs
