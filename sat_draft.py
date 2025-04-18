#!/usr/bin/env python3
"""Show, Attend and Tell – PyTorch implementation from scratch.

Core components:
    • EncoderCNN – extracts convolutional feature maps (14×14×D) from an image.
    • Attention     – additive (Bahdanau‑style) attention producing dynamic context vectors.
    • DecoderRNN   – LSTM language model that conditions on visual context each step.
    • ShowAttendTell – end‑to‑end captioning model (encoder + attention + decoder).

This file is self‑contained: it can be dropped into a project and imported.  A minimal
training script is sketched at the bottom.

Author: ChatGPT (o3)
Date: 2025‑04‑18
"""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models, transforms

################################################################################
# Encoder – fixed CNN backbone (VGG‑16 conv4_3 by default) that outputs a grid
# of visual features (L positions, D channels).  Gradients flow through the
# backbone (finetune=True) or only through a learned 1×1 projection (finetune=False).
################################################################################

class EncoderCNN(nn.Module):
    """Extract convolutional feature maps suitable for attention‑based decoding."""

    def __init__(self, backbone: str = "vgg16", finetune: bool = False) -> None:
        super().__init__()
        if backbone == "vgg16":
            cnn = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            # Keep layers up to conv4_3 feature map (before max‑pool).
            self.features = cnn.features[:33]  # conv4_3 relu
            feature_dim = 512
        elif backbone == "resnet50":
            cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            modules = list(cnn.children())[:-2]  # until C5 feature map (7×7×2048)
            self.features = nn.Sequential(*modules)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.project = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

        # Decide if we fine‑tune the backbone weights.
        for p in self.features.parameters():
            p.requires_grad = finetune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Images → annotation vectors (B, L=196, D)."""
        feats = self.adaptive_pool(self.features(images))  # (B, C, 14, 14)
        feats = self.project(feats)                       # (B, D, 14, 14)
        B, D, H, W = feats.shape
        return feats.view(B, D, H * W).permute(0, 2, 1)   # (B, L, D)

################################################################################
# Additive attention: fatt(a_i, h_{t‑1}) = v^T tanh(W_a a_i + W_h h_{t‑1})
################################################################################

class AdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, a: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights α_t and context vector ẑ_t.

        Args:
            a: (B, L, D) annotation vectors.
            h_prev: (B, H) previous LSTM hidden state.
        Returns:
            context (B, D), alphas (B, L)
        """
        wh = self.W_h(h_prev).unsqueeze(1)        # (B, 1, A)
        wa = self.W_a(a)                          # (B, L, A)
        e  = self.v(torch.tanh(wa + wh)).squeeze(-1)  # (B, L)
        alpha = F.softmax(e, dim=1)               # (B, L)
        context = torch.bmm(alpha.unsqueeze(1), a).squeeze(1)  # (B, D)
        return context, alpha

################################################################################
# Decoder – LSTM cell unrolled for T time‑steps with attention each step.
################################################################################

class DecoderRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = AdditiveAttention(feature_dim, hidden_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

    def init_hidden_state(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM (h_0, c_0) as learned projections of spatial mean features."""
        mean_ctx = a.mean(dim=1)  # (B, D)
        h = torch.tanh(self.init_h(mean_ctx))
        c = torch.tanh(self.init_c(mean_ctx))
        return h, c

    def forward(
        self,
        a: torch.Tensor,
        captions: torch.Tensor,
        lengths: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.

        Args:
            a        : (B, L, D) annotation vectors from encoder.
            captions : (B, T) int64 token IDs **with** <START> at position 0.
            lengths  : list of actual lengths (including <START> + words + <END>).  Used for packing.
        Returns:
            scores   : (sum(lengths‑1), V) vocabulary logits for each decoded step.
            alphas   : (B, T‑1, L) attention maps.
        """
        B, L, D = a.shape
        T = captions.size(1)
        embeddings = self.dropout(self.embed(captions))  # (B, T, E)

        h, c = self.init_hidden_state(a)
        alphas: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []

        for t in range(T - 1):  # decode up to last token (exclude <END>)
            context, alpha = self.attention(a, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            logits.append(output)
            alphas.append(alpha)

        logits = torch.stack(logits, dim=1)   # (B, T‑1, V)
        alphas = torch.stack(alphas, dim=1)   # (B, T‑1, L)

        # Pack logits for efficient cross‑entropy loss (mask padding).
        packed_logits = pack_padded_sequence(logits, [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data
        return packed_logits, alphas

    @torch.no_grad()
    def sample(self, a: torch.Tensor, max_len: int, start_idx: int, end_idx: int) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
        """Greedy caption generation (batch‑wise)."""
        B = a.size(0)
        h, c = self.init_hidden_state(a)
        inputs = torch.full((B,), start_idx, dtype=torch.long, device=a.device)

        captions: List[List[int]] = [[start_idx] for _ in range(B)]
        attention_maps: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for _ in range(max_len):
            context, alpha = self.attention(a, h)
            emb = self.embed(inputs)
            h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            scores = self.fc(h)
            _, next_word = scores.max(dim=1)
            inputs = next_word
            for i in range(B):
                captions[i].append(next_word[i].item())
                attention_maps[i].append(alpha[i].detach().cpu())
        # Cut off after first <END>
        clean_caps = []
        for cap in captions:
            if end_idx in cap:
                cap = cap[: cap.index(end_idx) + 1]
            clean_caps.append(cap)
        return clean_caps, attention_maps

################################################################################
# Full model wrapper (encoder + decoder) to simplify training/inference.
################################################################################

class ShowAttendTell(nn.Module):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        self.encoder = EncoderCNN(backbone=kwargs.get("backbone", "vgg16"), finetune=kwargs.get("finetune", False))
        feature_dim = 512 if kwargs.get("backbone", "vgg16") == "vgg16" else 2048
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=kwargs.get("embed_dim", 256),
            feature_dim=feature_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            attn_dim=kwargs.get("attn_dim", 512),
            dropout=kwargs.get("dropout", 0.3),
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: List[int]):
        a = self.encoder(images)
        scores, alphas = self.decoder(a, captions, lengths)
        return scores, alphas

################################################################################
# Minimal training loop sketch – you still need a Dataset that returns
# (image_tensor, caption_tensor, length).
################################################################################

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e‑4
    grad_clip: float = 5.0
    save_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train(model: ShowAttendTell, loader, criterion, cfg: TrainingConfig, vocab_size: int, save_dir: str | pathlib.Path = "checkpoints"):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, captions, lengths in loader:
            images = images.to(cfg.device)
            captions = captions.to(cfg.device)
            scores, _ = model(images, captions, lengths)
            targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch:02d}/{cfg.epochs} – loss {running_loss / len(loader):.3f}")

        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"sat_epoch{epoch}.pt")

################################################################################
# Example usage (pseudo‑code – fill Dataset/Tokenizer details before running).
################################################################################

if __name__ == "__main__":
    # 1. Build vocabulary → token ↔ index maps, START/END/PAD ids.
    vocab_size = 10000  # substitute with real size
    START, END = 1, 2   # indices in vocabulary

    # 2. Prepare DataLoader returning (image, caption, length).
    #    Each caption tensor starts with <START> and ends with <END>, padded to max length in batch.
    # from dataset import COCODataset
    # loader = DataLoader(COCODataset(...), batch_size=cfg.batch_size, ...)

    # 3. Instantiate model + loss.
    model = ShowAttendTell(vocab_size=vocab_size, backbone="vgg16", finetune=False)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD index = 0

    # cfg = TrainingConfig(epochs=25, lr=5e‑4)
    # train(model, loader, criterion, cfg, vocab_size)

    print("Model instantiated – fill dataset & training loop to run.")
