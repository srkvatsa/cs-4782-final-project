#!/usr/bin/env python3
"""Show, Attend and Tell – PyTorch implementation from scratch.

Core components:
    • EncoderCNN – extracts convolutional feature maps (14×14×D) from an image.
    • Attention     – additive (Bahdanau‑style) attention producing dynamic context vectors.
    • DecoderRNN   – LSTM language model that conditions on visual context each step.
    • ShowAttendTell – end‑to‑end captioning model (encoder + attention + decoder).

Implements key optimization techniques from the original paper:
    • Length-based sampling for efficient training
    • Early stopping based on BLEU score
    • Support for hyperparameter optimization
"""
from __future__ import annotations

import os
import math
import pathlib
import random
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import kagglehub
import nltk
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
nltk.download('punkt_tab')
nltk.download('wordnet')  # Required for METEOR

import types
import shutil

import json

################################################################################
# Encoder – fixed CNN backbone (VGG‑16 conv4_3 by default) that outputs a grid
# of visual features (L positions, D channels).  Gradients flow through the
# backbone (finetune=True) or only through a learned 1×1 projection (finetune=False).
################################################################################

class EncoderCNN(nn.Module):
    """Extract convolutional feature maps suitable for attention‑based decoding."""

    def __init__(self, backbone: str = "vgg19", finetune: bool = False) -> None:
        super().__init__()
        if backbone == "vgg19":
            cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            # Keep layers up to conv4_3 feature map (before max‑pool).
            self.features = cnn.features[:36]  # conv4_3 relu
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
        """Images → annotation vectors (B, L=196, D)."""
        feats = self.adaptive_pool(self.features(images))  # (B, C, 14, 14)
        feats = self.project(feats)                       # (B, D, 14, 14)
        B, D, H, W = feats.shape
        return feats.view(B, D, H * W).permute(0, 2, 1)   # (B, L, D)

################################################################################
# Additive attention: fatt(a_i, h_{t‑1}) = v^T tanh(W_a a_i + W_h h_{t‑1})
################################################################################

class DualAdditiveAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_a = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_c = nn.Linear(hidden_dim, attn_dim, bias=False)  # for context input
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.beta_fc = nn.Linear(hidden_dim, 1)  # for gating scalar
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, a: torch.Tensor, h_prev: torch.Tensor, context_input: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a: (B, L, D) – annotation vectors
            h_prev: (B, H) – previous LSTM state
            context_input: optional (B, H) – something else to attend with
        """
        wh = self.W_h(h_prev).unsqueeze(1)  # (B, 1, A)
        wa = self.W_a(a)                    # (B, L, A)
        score_input = wa + wh

        if context_input is not None:
            wc = self.W_c(context_input).unsqueeze(1)  # (B, 1, A)
            score_input = score_input + wc

        e = self.v(torch.tanh(score_input)).squeeze(-1)  # (B, L)
        alpha = F.softmax(e, dim=1)

        # Calculate gating scalar β
        beta = torch.sigmoid(self.beta_fc(h_prev))  # (B, 1)

        context = torch.bmm(alpha.unsqueeze(1), a).squeeze(1)  # (B, D)
        gated_context = beta * context  # Apply gating

        return gated_context, alpha, beta

def doubly_stochastic_regularization(attn_weights, lambda_reg=1.0):
    """
    attn_weights: list of T tensors, each (batch_size, num_locations)
    returns: regularization term
    """
    # Stack into shape (T, B, L)
    attn_tensor = torch.stack(attn_weights, dim=0)  # (T, B, L)
    attn_sum = attn_tensor.sum(dim=0)               # sum over times (B, L)
    reg = ((1.0 - attn_sum) ** 2).mean()            # average over batch and locations
    return lambda_reg * reg

################################################################################
# Decoder – LSTM cell unrolled for T time‑steps with attention each step.
################################################################################
def get_teacher_forcing_ratio(epoch, max_epochs, start_ratio=1.0, min_ratio=0.5, decay='linear'):
    """
    Calculate teacher forcing ratio based on training progress.

    Args:
        epoch: Current epoch number
        max_epochs: Total number of epochs
        start_ratio: Initial teacher forcing ratio
        min_ratio: Minimum teacher forcing ratio
        decay: Type of decay schedule ('linear', 'exp', or 'sigmoid')

    Returns:
        The teacher forcing ratio for the current epoch
    """
    if decay == 'linear':
        # Linear decay from start_ratio to min_ratio
        return max(min_ratio, start_ratio - (start_ratio - min_ratio) * (epoch / max_epochs))
    elif decay == 'exp':
        # Exponential decay
        return max(min_ratio, start_ratio * (0.95 ** epoch))
    elif decay == 'sigmoid':
        # Sigmoid decay centered at max_epochs/2
        x = 10 * (epoch - max_epochs/2) / max_epochs
        sigmoid = 1 / (1 + math.exp(x))
        return min_ratio + (start_ratio - min_ratio) * sigmoid
    else:
        return start_ratio  # No decay

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
        use_double_attention: bool = True,
        use_hard_attention: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_double_attention = use_double_attention
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = DualAdditiveAttention(feature_dim, hidden_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.use_hard_attention = use_hard_attention
        self.temperature = temperature

    def init_hidden_state(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_ctx = a.mean(dim=1)
        h = torch.tanh(self.init_h(mean_ctx))
        c = torch.tanh(self.init_c(mean_ctx))
        return h, c

    def forward(self, a: torch.Tensor, captions: torch.Tensor, lengths: List[int],
        teacher_forcing_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with scheduled sampling based on teacher_forcing_ratio.

        Args:
            a: Feature vectors (B, L, D)
            captions: Ground truth captions (B, T)
            lengths: List of caption lengths
            teacher_forcing_ratio: Probability of using teacher forcing (1.0 = always use ground truth)

        Returns:
            Tuple of (logits, attention_weights) or (logits, attention_weights, log_probs) with hard attention
        """
        B, L, D = a.shape
        T = captions.size(1)
        embeddings = self.dropout(self.embed(captions))

        h, c = self.init_hidden_state(a)
        alphas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        prev_h = h  # for double attention

        # First input is always <START> token
        inputs = captions[:, 0]

        for t in range(1, T):
            # Use previous word as input with probability (1 - teacher_forcing_ratio)
            if t > 1 and random.random() > teacher_forcing_ratio:
                inputs = predicted  # Use model's prediction
            else:
                inputs = captions[:, t-1]  # Use ground truth

            context_input = prev_h if self.use_double_attention else None
            if self.use_hard_attention:
                context, alpha, beta, log_prob, entropy = self.hard_attention_step(a, h, context_input, mode='train')
                log_probs.append(log_prob)
                entropies.append(entropy)
            else:
                context, alpha, beta = self.attention(a, h, context_input)

            # Get word embedding
            emb = self.embed(inputs)

            lstm_input = torch.cat([emb, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))

            # Get prediction for next step
            _, predicted = output.max(dim=1)

            logits.append(output)
            alphas.append(alpha)
            betas.append(beta)
            prev_h = h  # update for next time step

        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1)

        # Pack padded sequences
        packed_logits = pack_padded_sequence(logits, [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data

        if self.use_hard_attention:
            return packed_logits, alphas, torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)
        else:
            return packed_logits, alphas

    def greedy_search(self, features, max_len, start_idx, end_idx):
        """
        Args:
            features (Tensor): Encoder output features (1, L, D)
            max_len (int): Maximum caption length
            start_idx (int): Index of <START> token
            end_idx (int): Index of <END> token

        Returns:
            List[int]: Predicted caption as a list of word indices
        """
        device = features.device
        inputs = torch.tensor([start_idx], device=device)  # (1,)

        h, c = self.init_hidden_state(features)  # (1, hidden_dim)
        prev_h = h
        outputs = []

        for _ in range(max_len):
            context_input = prev_h if self.use_double_attention else None
            if self.use_hard_attention:
                context, alpha, beta, _, _ = self.hard_attention_step(features, h, context_input, mode='deterministic')
            else:
                context, alpha, beta = self.attention(features, h, context_input)

            emb = self.embed(inputs).squeeze(1)  # (1, embed_dim)
            lstm_input = torch.cat([emb, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            output = self.fc(self.dropout(h))  # (1, vocab_size)
            predicted = output.argmax(dim=1)  # (1,)
            predicted_idx = predicted.item()

            if predicted_idx == end_idx:
                break

            outputs.append(predicted_idx)
            inputs = predicted  # Use prediction as next input
            prev_h = h  # update for next step

        return outputs

    @torch.no_grad()
    def sample(self, a: torch.Tensor, max_len: int, start_idx: int, end_idx: int) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
        B = a.size(0)
        h, c = self.init_hidden_state(a)
        inputs = torch.full((B,), start_idx, dtype=torch.long, device=a.device)

        captions: List[List[int]] = [[start_idx] for _ in range(B)]
        attention_maps: List[List[torch.Tensor]] = [[] for _ in range(B)]

        prev_h = h  # for double attention

        for _ in range(max_len):
            context_input = prev_h if self.use_double_attention else None
            if self.use_hard_attention:
                context, alpha, beta, _, _ = self.hard_attention_step(a, h, context_input, mode='deterministic')
            else:
                context, alpha, beta = self.attention(a, h, context_input)
            emb = self.embed(inputs)
            h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            scores = self.fc(h)
            _, next_word = scores.max(dim=1)
            inputs = next_word
            for i in range(B):
                captions[i].append(next_word[i].item())
                attention_maps[i].append(alpha[i].detach().cpu())
            prev_h = h  # update for next step
        # Cut off after first <END>
        clean_caps = []
        for cap in captions:
            if end_idx in cap:
                cap = cap[:cap.index(end_idx) + 1]
            clean_caps.append(cap)
        return clean_caps, attention_maps

    def hard_attention_step(self, a, h_prev, context_input, mode='train'):
        """
        Args:
            mode: 'train' (default) | 'sample' | 'deterministic'
        """
        e = self.attention.v(torch.tanh(
            self.attention.W_a(a) + self.attention.W_h(h_prev).unsqueeze(1)
            + (self.attention.W_c(context_input).unsqueeze(1) if context_input is not None else 0)
        )).squeeze(-1)

        e = e / self.temperature
        alpha = F.softmax(e, dim=1)
        dist = torch.distributions.Categorical(alpha)
        entropy = dist.entropy()

        if mode == 'train':
            if torch.rand(1).item() < 0.5:
                idx = dist.sample()
                log_prob = dist.log_prob(idx)
                context = a[torch.arange(a.size(0), device=a.device), idx]
                alpha = F.one_hot(idx, num_classes=a.size(1)).float()
            else:
                context = torch.sum(alpha.unsqueeze(-1) * a, dim=1)
                log_prob = torch.zeros(a.size(0), device=a.device)
        elif mode == 'sample':
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            context = a[torch.arange(a.size(0), device=a.device), idx]
            alpha = F.one_hot(idx, num_classes=a.size(1)).float()
        elif mode == 'deterministic':
            idx = torch.argmax(alpha, dim=1)
            log_prob = torch.zeros(a.size(0), device=a.device)
            context = a[torch.arange(a.size(0), device=a.device), idx]
            alpha = F.one_hot(idx, num_classes=a.size(1)).float()

        beta = torch.ones(a.size(0), 1, device=a.device)
        return context, alpha, beta, log_prob, entropy


################################################################################
# Full model wrapper (encoder + decoder) to simplify training/inference.
################################################################################

class ShowAttendTell(nn.Module):
    def __init__(self, vocab_size: int, **kwargs):
        super().__init__()
        backbone = kwargs.get("backbone", "vgg19")
        self.encoder = EncoderCNN(backbone=backbone, finetune=kwargs.get("finetune", False))
        feature_dim = 512 if backbone == "vgg19" else 2048
        self.decoder = DecoderRNN(
            vocab_size=vocab_size,
            embed_dim=kwargs.get("embed_dim", 256),
            feature_dim=feature_dim,
            hidden_dim=kwargs.get("hidden_dim", 512),
            attn_dim=kwargs.get("attn_dim", 512),
            dropout=kwargs.get("dropout", 0.3),
            use_double_attention=kwargs.get("use_double_attention", True),
            use_hard_attention=kwargs.get("use_hard_attention", False),
        )
        self.decoder.beam_search = types.MethodType(beam_search, self.decoder)

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: List[int], teacher_forcing_ratio: float = 1.0):
      a = self.encoder(images)
      out = self.decoder(a, captions, lengths, teacher_forcing_ratio)

      if self.decoder.use_hard_attention:
          scores, alphas, log_probs, entropies = out
          return scores, alphas, log_probs, entropies
      else:
          scores, alphas = out
          return scores, alphas

################################################################################
# Vocabulary & Dataset Classes
################################################################################

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.freq_threshold = freq_threshold

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        # First pass: count frequencies
        for sentence in sentence_list:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

        # Second pass: add words that meet threshold
        # Sort by frequency (descending) and take the top 9996 (excluding 4 special tokens)
        most_common = frequencies.most_common(10000 - 4)
        for word, _ in most_common:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        return self

    def numericalize(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def __len__(self):
        return len(self.word2idx)

def collate_fn(batch):
    """Custom collate function for padding sequences in a batch."""
    images, captions, lengths = zip(*batch)

    # Stack all images in batch
    images = torch.stack(images, 0)

    # Store original lengths before padding
    lengths = torch.tensor(lengths)

    # Pad sequences to max length in batch
    padded_captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)

    return images, padded_captions, lengths.tolist()

class Flickr8kDataset(Dataset):
    def __init__(self, img_folder, captions_file, vocab=None, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.vocab = vocab

        self.captions = defaultdict(list)  # <-- NEW
        self.image_names = []
        self.texts = []
        skipped = 0

        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_file = parts[0].split('#')[0].strip()
                    caption = parts[1].strip()
                    img_path = os.path.join(self.img_folder, img_file)
                    if os.path.exists(img_path):
                        self.image_names.append(img_file)
                        self.texts.append(caption)
                        self.captions[img_file].append(caption)  # <-- store in dict
                    else:
                        skipped += 1


        if skipped:
            print(f"[Info] Skipped {skipped} captions with missing images.")

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = Vocabulary().build_vocab(self.texts)
        else:
            self.vocab = vocab

        # Group captions by length for length-based batching
        self.length_to_indices = defaultdict(list)
        for idx, text in enumerate(self.texts):
            caption = self.vocab.numericalize(text)
            caption_length = len(caption) + 2  # +2 for <START> and <END>
            self.length_to_indices[caption_length].append(idx)

        self.available_lengths = sorted(self.length_to_indices.keys())

        if not self.available_lengths:
            raise ValueError("No valid captions found in dataset")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = [self.vocab.word2idx["<START>"]] + \
                  self.vocab.numericalize(self.texts[idx]) + \
                  [self.vocab.word2idx["<END>"]]
        caption = torch.tensor(caption)
        length = len(caption)

        return image, caption, length

    def get_length_based_batch_indices(self, batch_size):
        length = random.choice(self.available_lengths)
        indices = self.length_to_indices[length]
        if len(indices) < batch_size:
            return random.choices(indices, k=batch_size)
        else:
            return random.sample(indices, batch_size)

def get_flickr8k_splits(data_dir, transform):
    img_folder = os.path.join(data_dir, "Flickr8k_Dataset", "Flicker8k_Dataset")
    captions_file = os.path.join(data_dir, "Flickr8k.token.txt")

    dataset = Flickr8kDataset(img_folder=img_folder, captions_file=captions_file, transform=transform)
    vocab = dataset.vocab

    # Load split files
    def load_split(file_name):
        path = os.path.join(data_dir, file_name)
        with open(path, 'r') as f:
            return set(line.strip() for line in f)

    train_images = load_split('Flickr_8k.trainImages.txt')
    val_images = load_split('Flickr_8k.devImages.txt')
    test_images = load_split('Flickr_8k.testImages.txt')

    # Filter indices based on image_names
    train_indices = [i for i, name in enumerate(dataset.image_names) if name in train_images]
    val_indices = [i for i, name in enumerate(dataset.image_names) if name in val_images]
    test_indices = [i for i, name in enumerate(dataset.image_names) if name in test_images]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, vocab

################################################################################
# Length-Based Batch Sampler
################################################################################

class LengthBasedBatchSampler:
    """
    Batch sampler that samples batches of uniform caption length, as described
    in the Show, Attend and Tell paper.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Handle both Dataset and Subset objects
        if hasattr(dataset, 'length_to_indices'):
            # Original dataset
            self.length_to_indices = dataset.length_to_indices
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'length_to_indices'):
            # Subset object - rebuild length_to_indices using only the indices in the subset
            self.length_to_indices = defaultdict(list)
            original_dataset = dataset.dataset
            subset_indices = dataset.indices

            # Map subset indices to original indices
            for i, idx in enumerate(subset_indices):
                # Get the caption length from the original dataset
                text = original_dataset.texts[idx]
                caption = original_dataset.vocab.numericalize(text)
                caption_length = len(caption) + 2  # +2 for <START> and <END>
                # Store the subset index (i) in our new length_to_indices
                self.length_to_indices[caption_length].append(i)
        else:
            raise TypeError("Dataset must have length_to_indices attribute or be a Subset of such a dataset")

        # Store available lengths for length-based sampling
        self.available_lengths = sorted(list(self.length_to_indices.keys()))

        # Pre-calculate total batches
        self.total_batches = self._calculate_total_batches()

    def _calculate_total_batches(self):
        total = 0
        for indices in self.length_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        lengths = self.available_lengths
        all_batches = []

        for _ in range(self.total_batches):
            # Randomly sample a length
            length = random.choice(lengths)
            indices = self.length_to_indices[length]

            # Sample a batch of that length
            batch = random.sample(indices, self.batch_size) if len(indices) >= self.batch_size else \
                    random.choices(indices, k=self.batch_size)

            all_batches.append(batch)

        # Shuffle final batch order
        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.total_batches

################################################################################
# Beam Search
################################################################################

# Define the BeamNode class for tracking partial sequences
class BeamNode:
    def __init__(self, prev_node, word_id, logprob, length, hidden_state, cell_state, alpha):
        self.prev_node = prev_node  # Previous node in the beam
        self.word_id = word_id      # Current word index
        self.logprob = logprob      # Cumulative log-probability
        self.length = length        # Current length of the sequence
        self.hidden_state = hidden_state  # LSTM hidden state
        self.cell_state = cell_state      # LSTM cell state
        self.alpha = alpha  # Attention map at this step

    def sequence(self):
        seq = []
        node = self
        while node.prev_node is not None:
            seq.append(node.word_id)
            node = node.prev_node
        return seq[::-1]  # reverse

    def score(self):
      alpha = 0.5
      lp = ((5 + self.length) / 6) ** alpha
      return self.logprob / lp

# Add beam_search method to DecoderRNN
@torch.no_grad()
def beam_search(self, a, beam_size, max_len, start_idx, end_idx):
    B = a.size(0)
    assert B == 1, "Beam search currently supports batch size 1 only"

    device = a.device
    h, c = self.init_hidden_state(a)
    inputs = torch.full((1,), start_idx, dtype=torch.long, device=device)

    root = BeamNode(prev_node=None, word_id=start_idx, logprob=0.0, length=1,
                    hidden_state=h, cell_state=c, alpha=None)
    nodes = [root]
    end_nodes = []

    for _ in range(max_len):
        next_nodes = []
        for node in nodes:
            if node.word_id == end_idx:
                end_nodes.append(node)
                continue
            h, c = node.hidden_state, node.cell_state
            context_input = h if self.use_double_attention else None
            context, alpha, _ = self.attention(a, h, context_input)
            emb = self.embed(torch.tensor([node.word_id], device=device))
            h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            temperature = 1.5
            scores = F.log_softmax(self.fc(h) / temperature, dim=-1)
            logprobs, top_ids = scores.topk(beam_size)

            for i in range(beam_size):
                next_node = BeamNode(prev_node=node,
                                     word_id=top_ids[0][i].item(),
                                     logprob=node.logprob + logprobs[0][i].item(),
                                     length=node.length + 1,
                                     hidden_state=h,
                                     cell_state=c,
                                     alpha=alpha)
                next_nodes.append(next_node)

        nodes = sorted(next_nodes, key=lambda n: n.score(), reverse=True)[:beam_size]

        if len(end_nodes) >= beam_size:
            break

    if len(end_nodes) == 0:
        end_nodes = nodes

    best_node = sorted(end_nodes, key=lambda n: n.score(), reverse=True)[0]
    return best_node.sequence(), best_node

################################################################################
# Training & Evaluation Functions
################################################################################

@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    grad_clip: float = 5.0
    save_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 5  # Patience for early stopping
    lambda_reg: float = 1.0  # Weight for doubly stochastic attention regularization
    length_based_sampling: bool = True  # Use length-based sampling as in paper
    early_stopping: bool = True  # Use early stopping based on BLEU
    teacher_forcing: bool = True  # Enable scheduled sampling (teacher forcing reduction)
    tf_start_ratio: float = 1.0  # Initial teacher forcing ratio
    tf_min_ratio: float = 0.5  # Minimum teacher forcing ratio
    tf_decay: str = 'sigmoid'  # Teacher forcing decay type ('linear', 'exp', 'sigmoid')

def save_checkpoint(model, optimizer, epoch, best_bleu, best_meteor, patience_counter, save_path, bleu_log=None):
    """Save a complete checkpoint that can be used to resume training."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_bleu': best_bleu,
        'best_meteor': best_meteor,  # Add METEOR score to checkpoint
        'patience_counter': patience_counter
    }
    torch.save(checkpoint_data, save_path)
    print(f"Checkpoint saved to {save_path}")

    # Save BLEU and METEOR log if provided, to a per-checkpoint log file
    if bleu_log is not None:
        log_name = f"eval_scores_epoch{epoch}.json"
        log_path = os.path.join(os.path.dirname(save_path), log_name)
        with open(log_path, 'w') as f:
            json.dump(bleu_log, f, indent=2)
        print(f"Saved evaluation scores log to {log_path}")

def load_checkpoint(model, optimizer=None, checkpoint_path=None, device=None):
    """Load a checkpoint and return the epoch to start from and other training state."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 1, 0.0, 0.0, 0, []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_bleu = checkpoint.get('best_bleu', 0.0)
    best_meteor = checkpoint.get('best_meteor', 0.0)  # Load METEOR score
    patience_counter = checkpoint.get('patience_counter', 0)

    # Load corresponding BLEU and METEOR log if it exists
    log_path = os.path.join(os.path.dirname(checkpoint_path), f"eval_scores_epoch{checkpoint['epoch']}.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            bleu_log = json.load(f)
        print(f"Loaded evaluation score log with {len(bleu_log)} entries from {log_path}")
    else:
        bleu_log = []

    print(f"Loaded checkpoint from epoch {start_epoch-1}")
    print(f"Best BLEU score so far: {best_bleu:.4f}")
    print(f"Best METEOR score so far: {best_meteor:.4f}")
    print(f"Patience counter: {patience_counter}")

    return start_epoch, best_bleu, best_meteor, patience_counter, bleu_log

def train_and_evaluate(
    model: ShowAttendTell,
    train_dataset,
    val_dataset,
    criterion,
    cfg: TrainingConfig,
    vocab_size: int,
    save_dir: str | pathlib.Path = "checkpoints",
    resume_from: str = None,
    start_epoch: str = 1
):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.8,
        patience=3,
        verbose=True,
        min_lr=1e-6  # or 1e-5 for a slightly more aggressive floor
    )

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Optionally load checkpoint
    best_bleu = 0.0
    best_meteor = 0.0  # Initialize best METEOR score
    patience_counter = 0
    bleu_log = []
    if resume_from and model.decoder.use_hard_attention:
        # Load model weights but not optimizer state
        start_epoch, best_bleu, best_meteor, _, bleu_log = load_checkpoint(
            model, optimizer=None, checkpoint_path=resume_from, device=cfg.device)
        
        # Create fresh optimizer after loading model weights
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Use lower learning rate for hard attention
        
        print("Switched to hard attention with fresh optimizer state")
    else:
        # Regular checkpoint loading with optimizer state
        if resume_from:
            start_epoch, best_bleu, best_meteor, patience_counter, bleu_log = load_checkpoint(
                model, optimizer, resume_from, cfg.device)

    if cfg.length_based_sampling:
        train_sampler = LengthBasedBatchSampler(train_dataset, cfg.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    best_model_path = save_dir / "best_model.pt"
    best_model_meteor_path = save_dir / "best_model_meteor.pt"  # A separate path for best METEOR model

    for epoch in range(start_epoch, cfg.epochs + 1):
        start_time = time.time()

        # Calculate teacher forcing ratio for this epoch
        teacher_forcing_ratio = get_teacher_forcing_ratio(
            epoch=epoch-1,  # 0-based for the function
            max_epochs=cfg.epochs,
            start_ratio=1.00,
            min_ratio=0.5,
            decay='linear'  # options: 'linear', 'exp', 'sigmoid'
        )

        print(f"Epoch {epoch}/{cfg.epochs}, Teacher forcing ratio: {teacher_forcing_ratio:.3f}")

        # CNN fine-tuning warm-up logic
        if epoch == 3 and hasattr(model.encoder, 'features'):
            print("Unfreezing CNN layers for fine-tuning...")
            for param in model.encoder.features[28:].parameters():
                param.requires_grad = True

        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        baseline = None

        for images, captions, lengths in pbar:
            images = images.to(cfg.device)
            captions = captions.to(cfg.device)

            if model.decoder.use_hard_attention:
                # Forward pass
                model.decoder.temperature = max(0.5, 1.0 - 0.5 * ((epoch - 51) / 19))
                scores, alphas, log_probs, entropies = model(images, captions, lengths, teacher_forcing_ratio)
                # scores: (N, vocab_size); log_probs: (B, T); entropies: (B, T)

                # Pack targets
                targets = pack_padded_sequence(
                    captions[:, 1:], [l - 1 for l in lengths],
                    batch_first=True, enforce_sorted=False
                ).data

                # Compute unreduced CE loss
                ce_loss_all = criterion(scores, targets)  # (N,)
                batch_size = log_probs.size(0)

                # Mean CE per sample
                ce_loss_per_sample = ce_loss_all.view(batch_size, -1).mean(dim=1)  # (B,)
                ce_loss = ce_loss_per_sample.mean()  # scalar for logging

                # Compute REINFORCE reward = -CE loss (we want to maximize log-likelihood)
                with torch.no_grad():
                    # Simpler reward is more stable to begin with
                    reward = -ce_loss_per_sample

                    # Apply baseline subtraction for variance reduction
                    if baseline is None:
                        baseline = reward.mean().item()
                    else:
                        baseline = 0.95 * baseline + 0.05 * reward.mean().item()

                    advantage = reward - baseline

                    # Normalize advantage for stable training
                    if advantage.std() > 0:
                        advantage = advantage / (advantage.std() + 1e-8)

                # REINFORCE term: sum log_probs over time, multiply by advantage
                log_probs_total = log_probs.sum(dim=1)  # (B,)
                reinforce_loss = -(log_probs_total * advantage).mean()

                # Entropy regularization: encourage exploration
                entropy_bonus = 0.05 * entropies.sum(dim=1).mean()  # Mean over B

                # Total REINFORCE loss with entropy
                reinforce_loss = reinforce_loss - entropy_bonus

                # Combine with CE loss (hybrid objective)
                total_loss = ce_loss + reinforce_loss
            else:
                # Original soft attention code
                scores, alphas = model(images, captions, lengths, teacher_forcing_ratio)
                targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data
                ce_loss = criterion(scores, targets)

                # Add doubly stochastic regularization
                reg_loss = doubly_stochastic_regularization(
                    [alphas[:, t, :] for t in range(alphas.size(1))],
                    lambda_reg=cfg.lambda_reg
                )
                total_loss = ce_loss + reg_loss

            # Rest of training steps remain the same
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            running_loss += total_loss.item()

            # Update progress bar with appropriate loss components
            if model.decoder.use_hard_attention:
                pbar.set_postfix({"loss": total_loss.item(), "ce": ce_loss.item(),
                                 "reinforce": reinforce_loss.item(), "tf_ratio": teacher_forcing_ratio})
            else:
                pbar.set_postfix({"loss": total_loss.item(), "ce": ce_loss.item(),
                                 "reg": reg_loss.item(), "tf_ratio": teacher_forcing_ratio})

        train_loss = running_loss / len(train_loader)
        current_bleu1, current_bleu2, current_bleu3, current_bleu4, current_meteor = evaluate_bleu_and_meteor_with_greedy(model, val_loader, cfg.device, val_dataset.dataset.vocab)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:02d}/{cfg.epochs} – Train Loss: {train_loss:.4f}, "
              f"BLEU-1: {current_bleu1:.4f}, BLEU-2: {current_bleu2:.4f}, BLEU-3: {current_bleu3:.4f}, BLEU-4: {current_bleu4:.4f}, "
              f"METEOR: {current_meteor:.4f}, Time: {epoch_time:.1f}s")

        # Update evaluation log
        bleu_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "bleu_1": current_bleu1,
            "bleu_2": current_bleu2,
            "bleu_3": current_bleu3,
            "bleu_4": current_bleu4,
            "meteor": current_meteor,
            "time": epoch_time
        })

        # You might want to use BLEU-4 or METEOR as the scheduling metric
        # Let's use a combination of both for scheduling
        combined_score = (current_bleu4 + current_meteor) / 2
        if (epoch > 40):
          scheduler.step(combined_score)

        if (epoch % cfg.save_every == 0):
          full_ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
          save_checkpoint(model, optimizer, epoch, best_bleu, best_meteor, patience_counter, full_ckpt_path, bleu_log)

        # If neither improved, increment patience counter
        if current_bleu4 < best_bleu and current_meteor < best_meteor:
            patience_counter += 1
            print(f"No improvement in scores for {patience_counter} epochs")

            # Check for early stopping based on patience
            if cfg.early_stopping and patience_counter >= cfg.patience:
                print(f"Early stopping after {epoch} epochs without improvement")
                break

        # Save model if BLEU-4 improved
        if current_bleu4 > best_bleu:
            best_bleu = current_bleu4
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_bleu, best_meteor, patience_counter, best_model_path, bleu_log)
            print(f"New best BLEU-4 score: {best_bleu:.4f} - Saving model")

        # Save model if METEOR improved
        if current_meteor > best_meteor:
            best_meteor = current_meteor
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_bleu, best_meteor, patience_counter, best_model_meteor_path, bleu_log)
            print(f"New best METEOR score: {best_meteor:.4f} - Saving model")

    # Load the best model based on BLEU-4 for final evaluation
    start_epoch, best_bleu, best_meteor, _, _ = load_checkpoint(model, optimizer, best_model_path, cfg.device)
    final_bleu1, final_bleu2, final_bleu3, final_bleu4, final_meteor = evaluate_bleu_and_meteor_with_greedy(model, val_loader, cfg.device, val_dataset.dataset.vocab)
    print(f"Training completed. Best BLEU-4 score: {final_bleu4:.4f}, BLEU-1: {final_bleu1:.4f}, BLEU-2: {final_bleu2:.4f}, BLEU-3: {final_bleu3:.4f}, METEOR: {final_meteor:.4f}")

    return model, final_bleu1, final_bleu2, final_bleu3, final_bleu4, final_meteor

def bleu_scores_without_bp(references, hypotheses):
    import collections
    from nltk.util import ngrams
    import math

    max_n = 4
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref_list, hyp in zip(references, hypotheses):
        # Get n-gram counts for each n
        hyp_ngrams = [collections.Counter(ngrams(hyp, i + 1)) for i in range(max_n)]
        ref_ngram_counters = [[collections.Counter(ngrams(ref, i + 1)) for ref in ref_list] for i in range(max_n)]

        for i in range(max_n):
            # Get max reference counts across refs
            ref_max = collections.Counter()
            for ref_counts in ref_ngram_counters[i]:
                ref_max |= ref_counts
            overlap = hyp_ngrams[i] & ref_max
            clipped_counts[i] += sum(overlap.values())
            total_counts[i] += max(sum(hyp_ngrams[i].values()), 1)

    # Precision per n-gram level
    precisions = [clipped_counts[i] / total_counts[i] for i in range(max_n)]

    # Compute BLEU-n scores (geometric mean of first n precisions)
    def geo_mean(precisions, n):
        return math.exp(sum(math.log(p + 1e-9) for p in precisions[:n]) / n)

    bleu1 = precisions[0]
    bleu2 = geo_mean(precisions, 2)
    bleu3 = geo_mean(precisions, 3)
    bleu4 = geo_mean(precisions, 4)

    return bleu1, bleu2, bleu3, bleu4

@torch.no_grad()
def evaluate_bleu_and_meteor_with_greedy(model, loader, device, vocab, max_len=20):
    model.eval()
    references = []
    hypotheses = []
    meteor_scores = []

    for images, captions, lengths in tqdm(loader, desc="Evaluating with Greedy Search"):
        images = images.to(device)
        batch_size = images.size(0)
        a = model.encoder(images)

        for i in range(batch_size):
            features = a[i:i+1]
            sampled_ids = model.decoder.greedy_search(
                features,
                max_len=max_len,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"]
            )

            pred_tokens = [vocab.idx2word[idx] for idx in sampled_ids
                           if idx not in {vocab.word2idx["<PAD>"], vocab.word2idx["<START>"], vocab.word2idx["<END>"]}]
            hypothesis = ' '.join(pred_tokens)
            hypotheses.append(pred_tokens)

            img_idx = loader.dataset.indices[i] if hasattr(loader.dataset, "indices") else i
            img_name = loader.dataset.dataset.image_names[img_idx]
            ref_caps = loader.dataset.dataset.captions[img_name]

            ref_token_lists = []
            ref_strings = []
            for ref in ref_caps:
                tokens = nltk.word_tokenize(ref.lower())
                ref_strings.append(' '.join(tokens))
                ref_token_lists.append(tokens)

            references.append(ref_token_lists)

            hypothesis_tokens = pred_tokens
            meteor = max([single_meteor_score(nltk.word_tokenize(ref), hypothesis_tokens) for ref in ref_strings])
            meteor_scores.append(meteor)

    for i in range(5):
        print(f"\nSample {i+1}")
        print("Hypothesis:", ' '.join(hypotheses[i]))
        print("Reference:", [' '.join(ref) for ref in references[i]])
        print("METEOR:", f"{meteor_scores[i]:.4f}")

    bleu1, bleu2, bleu3, bleu4 = bleu_scores_without_bp(references, hypotheses)
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu3, bleu4, meteor_avg

@torch.no_grad()
def evaluate_bleu_and_meteor_with_beam(model, loader, device, vocab, beam_size=8, max_len=20):
    model.eval()
    references = []
    hypotheses = []
    meteor_scores = []

    for images, captions, lengths in tqdm(loader, desc="Evaluating with Beam Search"):
        images = images.to(device)
        batch_size = images.size(0)
        a = model.encoder(images)

        for i in range(batch_size):
            features = a[i:i+1]
            sampled_ids, _ = model.decoder.beam_search(
                features,
                beam_size=beam_size,
                max_len=max_len,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"]
            )

            pred_tokens = [vocab.idx2word[idx] for idx in sampled_ids
                           if idx not in {vocab.word2idx["<PAD>"], vocab.word2idx["<START>"], vocab.word2idx["<END>"]}]
            hypothesis = ' '.join(pred_tokens)
            hypotheses.append(pred_tokens)

            img_idx = loader.dataset.indices[i] if hasattr(loader.dataset, "indices") else i
            img_name = loader.dataset.dataset.image_names[img_idx]
            ref_caps = loader.dataset.dataset.captions[img_name]

            ref_token_lists = []
            ref_strings = []
            for ref in ref_caps:
                tokens = nltk.word_tokenize(ref.lower())
                ref_strings.append(' '.join(tokens))
                ref_token_lists.append(tokens)

            references.append(ref_token_lists)

            # FIX: Pass tokenized hypothesis (list of tokens) rather than a string
            # meteor = max([single_meteor_score(ref, hypothesis) for ref in ref_strings])

            # The hypothesis needs to be tokenized - we already have pred_tokens
            hypothesis_tokens = pred_tokens  # Already a list of tokens
            meteor = max([single_meteor_score(nltk.word_tokenize(ref), hypothesis_tokens) for ref in ref_strings])
            meteor_scores.append(meteor)

    for i in range(5):
        print(f"\nSample {i+1}")
        print("Hypothesis:", ' '.join(hypotheses[i]))
        print("Reference:", [' '.join(ref) for ref in references[i]])
        print("METEOR:", f"{meteor_scores[i]:.4f}")

    bleu1, bleu2, bleu3, bleu4 = bleu_scores_without_bp(references, hypotheses)
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu3, bleu4, meteor_avg

################################################################################
# Visualization Utilities
################################################################################

def find_idx(dataset, vocab, target_caption_text):
    """
    Search the dataset for an image that has the given reference caption text.
    Returns the dataset index if found, else None.
    """
    for idx in range(len(dataset)):
        _, caption_tensor, _ = dataset[idx]
        caption_words = [vocab.idx2word[token.item()] for token in caption_tensor if token.item() not in {0, 1, 2}]  # Exclude special tokens
        caption_text = ' '.join(caption_words)

        if target_caption_text.strip().lower() in caption_text.strip().lower():
            print("Found: ", target_caption_text)
            return idx  # Found it!

    print(target_caption_text, "Not Found")
    return None  # Not found

# Generate and save some sample captions
def generate_sample_captions(model, dataset, device, num_samples=5):
    model.eval()
    # [2143, 4372, 1022, None, 4897]
    # [2143, 4372, 1022, 497, 4897, None, 59]
    # indices = [find_idx(dataset, vocab, "a girl walks towards some sheep in a grassy valley"),
    #            find_idx(dataset, vocab, "a sheltie dog carries a white-colored toy in its mouth as it walks across the snow"),
    #            find_idx(dataset, vocab, "a brown dog shaking off water"),
    #            find_idx(dataset, vocab, "a young man climbs a mountain"),
    #            find_idx(dataset, vocab, "a woman dressed in green is playing with her tan dog")]
    indices = [2143,
               4372,
               1022,
               497,
               4897,
               find_idx(dataset, vocab, "little girl crouches"),
               59]
    # indices = random.sample(range(len(dataset)), num_samples)
    print(indices)

    samples = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, caption, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Get image features
            features = model.encoder(image)

            # Generate caption with attention
            pred_indices, attention_maps = model.decoder.sample(
                features,
                max_len=20,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"]
            )

            # Convert to text
            pred_text = ' '.join([vocab.idx2word[idx] for idx in pred_indices[0]
                                  if idx not in {0, 1, 2}])  # Exclude PAD, START, END

            # Get reference caption
            ref_text = ' '.join([vocab.idx2word[idx.item()] for idx in caption
                                if idx.item() not in {0, 1, 2}])

            samples.append({
                'prediction': pred_text,
                'prediction_indices': pred_indices[0],
                'reference': ref_text,
                'attention_maps': attention_maps[0],
                'dataset_idx': idx
            })

    return samples

def visualize_attention_paper_style(image_path, caption_indices, attention_maps, vocab, reference_caption=None, save_path=None):
    """
    Visualize attention weights in the style of the Show, Attend and Tell paper with
    soft or hard Gaussian-blurred attention regions over the image.
    Supports both soft and hard attention visualizations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from PIL import Image, ImageFilter
    import scipy.ndimage

    def generate_gaussian_blob(index, map_size=(14, 14), sigma=1.0):
        """Create a soft Gaussian blob centered at the given index (for hard attention)."""
        blob = np.zeros(map_size)
        h, w = map_size
        y, x = divmod(index, w)
        blob[y, x] = 1.0
        blob = scipy.ndimage.gaussian_filter(blob, sigma=sigma)
        blob /= (blob.max() + 1e-8)
        return blob

    # Open image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # Filter out special tokens
    words = [vocab.idx2word[idx] for idx in caption_indices if idx not in {0, 1, 2}]
    prediction = ' '.join(words)
    num_words = min(len(words), len(attention_maps))

    plt.figure(figsize=(12, 2 * (num_words + 1)))

    # Plot original image with full caption
    plt.subplot(num_words + 1, 1, 1)
    plt.imshow(img_np)
    if reference_caption:
        plt.title(f"Reference: {reference_caption}\nPrediction: {prediction}", fontsize=12)
    else:
        plt.title(f"Prediction: {prediction}", fontsize=12)
    plt.axis('off')

    # Plot attention maps
    for i in range(num_words):
        plt.subplot(num_words + 1, 1, i + 2)

        att_map = attention_maps[i].cpu().detach().numpy()

        # print(f"[{i}] alpha max: {attention_maps[i].max().item():.4f}, min: {attention_maps[i].min().item():.4f}, argmax: {attention_maps[i].argmax().item()}")
        # print(f"alpha[i]: {attention_maps[i].detach().cpu().numpy()}")

        # If attention is nearly one-hot, treat as hard attention and generate a blob
        if np.count_nonzero(att_map > 0.9 * att_map.max()) == 1:
            max_idx = att_map.argmax()
            att_map = generate_gaussian_blob(max_idx, map_size=(14, 14), sigma=1.0)
        else:
            # Soft attention: just reshape
            att_map = att_map.reshape(14, 14)
            att_map = att_map / (att_map.max() + 1e-8)

        # Resize and blur to image size
        att_map_img = Image.fromarray((att_map * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
        att_map_blurred = att_map_img.filter(ImageFilter.GaussianBlur(radius=3))
        att_map_np = np.array(att_map_blurred) / 255.0

        # Blend with image
        plt.imshow(img_np)
        plt.imshow(att_map_np, cmap='gray', alpha=0.6, vmin=0, vmax=1)
        plt.title(f"'{words[i]}'", fontsize=12)
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



################################################################################
# Main execution
################################################################################

if __name__ == "__main__":
    import nltk
    from torchvision import transforms

    # Ensure nltk tokenizer is downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Add color jitter
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),  # Add slight affine transforms
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download Flickr8k dataset from Kaggle
    print("Downloading Flickr8k dataset...")
    data_dir = kagglehub.dataset_download("ashish2001/original-flickr8k-dataset")
    img_folder = os.path.join(data_dir, "Flickr8k_Dataset", "Flicker8k_Dataset")

    # Get splits + vocab
    print("Creating dataset and building vocabulary...")
    train_dataset, val_dataset, test_dataset, vocab = get_flickr8k_splits(data_dir, transform)

    vocab_size = len(vocab)

    # Default parameters
    model_params = {
        'embed_dim': 512,
        'hidden_dim': 512,
        'attn_dim': 512,
        'dropout': 0.2,
        'lambda_reg': 0.2,
        'lr': 1e-5,
        'use_double_attention': False,
        'finetune': True,
        'use_hard_attention': True
    }

    # Create model with parameters
    print("Creating model with parameters:", model_params)
    model = ShowAttendTell(
        vocab_size=vocab_size,
        backbone="resnet50",
        **model_params
    )

    # Loss function and config
    if model_params['use_hard_attention']:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0, reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)  # Ignore padding
    config = TrainingConfig(
        epochs=100,
        batch_size=64,
        lr=model_params['lr'],
        grad_clip=5.0,
        save_every=2,
        lambda_reg=model_params['lambda_reg'],
        patience=30,
        length_based_sampling=True
    )

    # Setup for resuming training
    # These are the values you should modify to resume training
    resume_path = "./checkpoints/checkpoint_epoch70.pt"  # Path to the .pt file to load (e.g., "checkpoints/sat_epoch10.pt")
    start_epoch = 71     # The epoch number to start from (e.g., 11 if resuming after epoch 10)

    if resume_path: # For test
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        start_epoch, best_bleu, best_meteor, patience_counter, bleu_log = load_checkpoint(model, optimizer, resume_path, config.device)
        model = model.to(device)

    # if resume_path:
    #     print(f"Resuming training from {resume_path} at epoch {start_epoch}")

    # Train the model
    # print("Starting training...")
    # model, val_bleu1, val_bleu2, val_bleu3, val_bleu4, val_meteor = train_and_evaluate(
    #     model,
    #     train_dataset,
    #     val_dataset,
    #     criterion,
    #     config,
    #     vocab_size,
    #     resume_from=resume_path,
    #     start_epoch=start_epoch
    # )

    # Print all evaluation scores after training
    # print(f"Validation BLEU-1: {val_bleu1:.4f}")
    # print(f"Validation BLEU-2: {val_bleu2:.4f}")
    # print(f"Validation BLEU-3: {val_bleu3:.4f}")
    # print(f"Validation BLEU-4: {val_bleu4:.4f}")
    # print(f"Validation METEOR: {val_meteor:.4f}")

    # # Final evaluation on test set
    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    # test_bleu1, test_bleu2, test_bleu3, test_bleu4, test_meteor = evaluate_bleu_and_meteor_with_greedy(model, test_loader, config.device, vocab)

    # Print evaluation scores for test set
    # print(f"Test BLEU-1 score: {test_bleu1:.4f}")
    # print(f"Test BLEU-2 score: {test_bleu2:.4f}")
    # print(f"Test BLEU-3 score: {test_bleu3:.4f}")
    # print(f"Test BLEU-4 score: {test_bleu4:.4f}")
    # print(f"Test METEOR score: {test_meteor:.4f}")

    print("Generating sample captions with attention visualization...")
    samples = generate_sample_captions(model, test_dataset, device, num_samples=10)

    # Create a directory for saving visualizations
    os.makedirs("attention_visualizations", exist_ok=True)

    # Visualize attention maps for each sample
    for i, sample in enumerate(samples):
      print(f"\nSample {i+1}:")
      print(f"Reference: {sample['reference']}")
      print(f"Prediction: {sample['prediction']}")

      # Get the image path from the dataset
      img_idx = test_dataset.indices[sample['dataset_idx']]
      img_name = os.path.join(img_folder, test_dataset.dataset.image_names[img_idx])

      # Visualize attention
      save_path = f"attention_visualizations/sample_{i+1}.png"
      visualize_attention_paper_style(
          img_name,
          sample['prediction_indices'],
          sample['attention_maps'],
          vocab,
          reference_caption=sample['reference'],  # Pass the reference caption
          save_path=save_path
      )
      print(f"Attention visualization saved to {save_path}")

    print("\nTraining and evaluation complete!")