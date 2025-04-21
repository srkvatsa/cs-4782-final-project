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

import kagglehub
import nltk
from tqdm import tqdm

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from nltk.translate.bleu_score import corpus_bleu
nltk.download('punkt_tab')

import types
import shutil

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

    def init_hidden_state(self, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_ctx = a.mean(dim=1)
        h = torch.tanh(self.init_h(mean_ctx))
        c = torch.tanh(self.init_c(mean_ctx))
        return h, c

    def forward(self, a: torch.Tensor, captions: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = a.shape
        T = captions.size(1)
        embeddings = self.dropout(self.embed(captions))

        h, c = self.init_hidden_state(a)
        alphas: List[torch.Tensor] = []
        betas: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []

        prev_h = h  # for double attention

        for t in range(T - 1):
            context_input = prev_h if self.use_double_attention else None
            context, alpha, beta = self.attention(a, h, context_input)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            logits.append(output)
            alphas.append(alpha)
            betas.append(beta)
            prev_h = h  # update for next time step

        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1)
        packed_logits = pack_padded_sequence(logits, [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data
        return packed_logits, alphas

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
            context, alpha, _ = self.attention(a, h, context_input)
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
            use_double_attention=kwargs.get("use_double_attention", True)
        )
        self.decoder.beam_search = types.MethodType(beam_search, self.decoder)

    def forward(self, images: torch.Tensor, captions: torch.Tensor, lengths: List[int]):
        a = self.encoder(images)
        scores, alphas = self.decoder(a, captions, lengths)
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
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        return self

    def numericalize(self, text):
        tokens = nltk.tokenize.word_tokenize(text.lower())
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
    
    def __len__(self):
        return len(self.word2idx)

class Flickr8kDataset(Dataset):
    def __init__(self, img_folder, captions_file, vocab=None, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.vocab = vocab

        self.image_names = []
        self.texts = []
        skipped = 0

        # Parse captions file
        with open(captions_file, 'r') as f:
            for line in f.readlines():
                # Assuming format: image_name#id,caption
                parts = line.strip().split(',', 1)  # Split on first comma only
                if len(parts) == 2:
                    img_file = parts[0].split('#')[0]
                    img_path = os.path.join(self.img_folder, img_file)
                    if os.path.exists(img_path):
                        self.image_names.append(img_file)
                        self.texts.append(parts[1])
                    else:
                        skipped += 1
                        print(f"[Skipped] Missing image file: {img_path}")

        if skipped:
            print(f"[Info] Skipped {skipped} captions with missing images.")

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = Vocabulary().build_vocab(self.texts)
        else:
            self.vocab = vocab

        # Group captions by length
        self.length_to_indices = defaultdict(list)
        for idx, text in enumerate(self.texts):
            caption = self.vocab.numericalize(text)
            caption_length = len(caption) + 2  # +2 for <START> and <END>
            self.length_to_indices[caption_length].append(idx)

        # Store available lengths for length-based sampling
        self.available_lengths = sorted(list(self.length_to_indices.keys()))

        # Check if we have any valid captions
        if not self.available_lengths:
            raise ValueError("No valid captions found in dataset")

    def __len__(self):
        return len(self.texts)  # Fix: should return the length of texts (captions)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_folder, self.image_names[idx])
        image = Image.open(img_name).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Tokenize caption
        caption = [self.vocab.word2idx["<START>"]] + self.vocab.numericalize(self.texts[idx]) + [self.vocab.word2idx["<END>"]]
        length = len(caption)
        caption = torch.tensor(caption)
        
        return image, caption, length
    
    def get_length_based_batch_indices(self, batch_size):
        """Get indices for a batch of samples with the same caption length."""
        # Randomly choose a caption length
        length = random.choice(self.available_lengths)
        indices = self.length_to_indices[length]
        
        # If we don't have enough samples of this length, sample with replacement
        if len(indices) < batch_size:
            sampled_indices = random.choices(indices, k=batch_size)
        else:
            sampled_indices = random.sample(indices, batch_size)
            
        return sampled_indices

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
        return self.logprob / self.length  # length normalization (Xu et al.)

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
            scores = F.log_softmax(self.fc(h), dim=-1)
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
    save_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 5  # Patience for early stopping
    lambda_reg: float = 1.0  # Weight for doubly stochastic attention regularization
    length_based_sampling: bool = True  # Use length-based sampling as in paper
    early_stopping: bool = True  # Use early stopping based on BLEU


def train_and_evaluate(
    model: ShowAttendTell, 
    train_dataset,
    val_dataset,
    criterion, 
    cfg: TrainingConfig, 
    vocab_size: int, 
    save_dir: str | pathlib.Path = "checkpoints"
):
    model.to(cfg.device)
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)  # RMSprop for flickr8k
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Create dataloaders based on configuration
    if cfg.length_based_sampling:
        # Length-based sampling as described in the paper
        train_sampler = LengthBasedBatchSampler(train_dataset, cfg.batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=collate_fn,
            num_workers=4,  # Adjust based on your CPU
            pin_memory=True  # Speeds up host to GPU transfers
        )
    else:
        # Traditional random sampling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )

    # Validation loader (always random sampling for evaluation)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # Keep track of best BLEU score and patience counter
    best_bleu = 0.0
    patience_counter = 0
    best_model_path = save_dir / "best_model.pt"

    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for images, captions, lengths in pbar:
            images = images.to(cfg.device)
            captions = captions.to(cfg.device)

            # Forward pass
            scores, alphas = model(images, captions, lengths)
            targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False).data

            # Calculate loss components
            ce_loss = criterion(scores, targets)
            reg_loss = doubly_stochastic_regularization(
                [alphas[:, t, :] for t in range(alphas.size(1))], 
                lambda_reg=cfg.lambda_reg
            )
            total_loss = ce_loss + reg_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # Update tracking
            running_loss += total_loss.item()
            pbar.set_postfix({"loss": total_loss.item(), "ce": ce_loss.item(), "reg": reg_loss.item()})

        train_loss = running_loss / len(train_loader)

        # Evaluation phase
        current_bleu1, current_bleu2, current_bleu3, current_bleu4 = evaluate_bleu_with_beam(model, val_loader, cfg.device, val_dataset.dataset.vocab)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:02d}/{cfg.epochs} – Train Loss: {train_loss:.4f}, "
              f"BLEU-1: {current_bleu1:.4f}, BLEU-2: {current_bleu2:.4f}, BLEU-3: {current_bleu3:.4f}, BLEU-4: {current_bleu4:.4f}, Time: {epoch_time:.1f}s")

        # Save checkpoint if requested
        if epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), save_dir / f"sat_epoch{epoch}.pt")

        # Check if this is the best model so far
        if current_bleu4 > best_bleu:
            best_bleu = current_bleu4
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best BLEU-4 score: {best_bleu:.4f} - Saving model")
        else:
            patience_counter += 1
            print(f"No improvement in BLEU-4 score for {patience_counter} epochs")

        # Early stopping based on BLEU score
        if cfg.early_stopping and patience_counter >= cfg.patience:
            print(f"Early stopping after {epoch} epochs without BLEU improvement")
            break

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    final_bleu1, final_bleu2, final_bleu3, final_bleu4 = evaluate_bleu_with_beam(model, val_loader, cfg.device, val_dataset.vocab)
    print(f"Training completed. Best BLEU-4 score: {final_bleu4:.4f}, BLEU-1: {final_bleu1:.4f}, BLEU-2: {final_bleu2:.4f}, BLEU-3: {final_bleu3:.4f}")

    return model, final_bleu1, final_bleu2, final_bleu3, final_bleu4


# Updated BLEU evaluation with beam search
@torch.no_grad()
def evaluate_bleu_with_beam(model, loader, device, vocab, beam_size=5, max_len=20):
    model.eval()
    references = []
    hypotheses = []

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

            # Filter special tokens from prediction
            pred_tokens = [vocab.idx2word[idx] for idx in sampled_ids 
                           if idx not in {vocab.word2idx["<PAD>"], vocab.word2idx["<START>"], vocab.word2idx["<END>"]}]

            # Filter special tokens from reference
            ref_tokens = [vocab.idx2word[idx.item()] for idx in captions[i] 
                          if idx.item() not in {vocab.word2idx["<PAD>"], vocab.word2idx["<START>"], vocab.word2idx["<END>"]}]

            hypotheses.append(pred_tokens)
            references.append([ref_tokens])

    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4

def zip_and_download_checkpoints(checkpoint_dir="checkpoints"):
    if os.path.exists(checkpoint_dir):
        shutil.make_archive("checkpoints", 'zip', checkpoint_dir)
        try:
            from google.colab import files
            files.download("checkpoints.zip")
            print("✅ Checkpoints zipped and download started.")
        except ImportError:
            print("⚠️ Not in a Colab environment — zip created but not downloaded.")
    else:
        print(f"❌ Checkpoint directory '{checkpoint_dir}' does not exist.")

################################################################################
# Main execution
################################################################################

if __name__ == "__main__":
    # Make sure NLTK packages are downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Download Flickr8k dataset
    print("Downloading Flickr8k dataset...")
    data_dir = kagglehub.dataset_download("adityajn105/flickr8k")
    img_folder = os.path.join(data_dir, "Images")
    captions_file = os.path.join(data_dir, "captions.txt")
    
    # Create dataset and build vocabulary
    print("Creating dataset and building vocabulary...")
    dataset = Flickr8kDataset(
        img_folder=img_folder,
        captions_file=captions_file,
        transform=transform
    )
    vocab = dataset.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create train/val/test splits
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Make sure all splits have the same dataset object
    train_dataset.dataset = dataset
    val_dataset.dataset = dataset
    test_dataset.dataset = dataset
    
    # Default parameters
    model_params = {
        'embed_dim': 512,
        'hidden_dim': 512,
        'attn_dim': 512,
        'dropout': 0.3,
        'lambda_reg': 1.0,
        'lr': 3e-4,
        'use_double_attention': True,
        'finetune': False
    }
    
    # Create model with parameters
    print("Creating model with parameters:", model_params)
    model = ShowAttendTell(
        vocab_size=vocab_size,
        backbone="vgg19",
        **model_params
    )
    
    # Loss function and config
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    config = TrainingConfig(
        epochs=50,
        batch_size=64,
        lr=model_params['lr'],
        grad_clip=5.0,
        save_every=5,
        lambda_reg=model_params['lambda_reg'],
        patience=10,
        length_based_sampling=True
    )
    
    # Path to resume training from (None for fresh training)
    resume_path = None  # Change this to the path of your checkpoint
    
    # You can uncomment and modify this to specify a checkpoint path:
    # resume_path = "checkpoints/best_checkpoint.pt"  # or any other checkpoint
    
    # If a command line argument is provided for the resume path
    import sys
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
        print(f"Resuming training from: {resume_path}")
    
    # Train the model
    print("Starting training...")
    model, val_bleu1, val_bleu2, val_bleu3, val_bleu4 = train_and_evaluate(
        model, 
        train_dataset,
        val_dataset,
        criterion, 
        config, 
        vocab_size,
        resume_from=resume_path
    )
    
    # Print all BLEU scores after training
    print(f"Validation BLEU-1: {val_bleu1:.4f}")
    print(f"Validation BLEU-2: {val_bleu2:.4f}")
    print(f"Validation BLEU-3: {val_bleu3:.4f}")
    print(f"Validation BLEU-4: {val_bleu4:.4f}")

    zip_and_download_checkpoints("checkpoints")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_bleu1, test_bleu2, test_bleu3, test_bleu4 = evaluate_bleu_with_beam(model, test_loader, config.device, vocab)
    
    # Print BLEU scores for test set
    print(f"Test BLEU-1 score: {test_bleu1:.4f}")
    print(f"Test BLEU-2 score: {test_bleu2:.4f}")
    print(f"Test BLEU-3 score: {test_bleu3:.4f}")
    print(f"Test BLEU-4 score: {test_bleu4:.4f}")
    
    # Generate and save some sample captions
    def generate_sample_captions(model, dataset, device, num_samples=5):
        model.eval()
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=num_samples)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn)
        
        samples = []
        with torch.no_grad():
            for images, captions, lengths in loader:
                images = images.to(device)
                a = model.encoder(images)
                pred_captions, attention_maps = model.decoder.sample(
                    a, 
                    max_len=20, 
                    start_idx=vocab.word2idx["<START>"], 
                    end_idx=vocab.word2idx["<END>"]
                )
                
                # Convert to text
                pred_text = ' '.join([vocab.idx2word[idx] for idx in pred_captions[0] 
                                     if idx not in {0, 1, 2}])  # Exclude PAD, START, END
                                     
                # Get reference caption
                ref_text = ' '.join([vocab.idx2word[idx.item()] for idx in captions[0] 
                                   if idx.item() not in {0, 1, 2}])
                
                samples.append({
                    'prediction': pred_text,
                    'reference': ref_text,
                    'attention_maps': attention_maps[0]
                })
        
        return samples
    
    print("Generating sample captions...")
    samples = generate_sample_captions(model, test_dataset, device)
    
    # Print sample results
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Reference: {sample['reference']}")
        print(f"Prediction: {sample['prediction']}")
    
    print("\nTraining and evaluation complete!")

################################################################################
# Visualization Utilities
################################################################################

def visualize_attention(image_path, caption, attention_maps, vocab, save_path=None):
    """
    Visualize attention weights overlaid on the original image.
    
    Args:
        image_path: Path to the image
        caption: List of word indices
        attention_maps: List of attention weight tensors
        vocab: Vocabulary object for converting indices to words
        save_path: Path to save the visualization, if None will display instead
    """
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np

    if not os.path.exists(image_path):
        print(f"[Warning] Missing image file: {image_path}. Skipping.")
        return

    image = Image.open(img_name).convert("RGB")

    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(14, 14))
    
    # Remove special tokens from caption
    words = [vocab.idx2word[idx] for idx in caption if idx not in {0, 1, 2}]
    
    # Number of attention maps
    num_maps = min(len(words), len(attention_maps))
    
    # Create a grid for subplots (original image + attention maps)
    rows = int(np.ceil(np.sqrt(num_maps + 1)))
    cols = int(np.ceil((num_maps + 1) / rows))
    
    # Show original image
    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # Show attention maps
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 2)
        
        # Reshape attention to original image size
        att_map = attention_maps[i].reshape(14, 14)
        att_map = att_map.detach().numpy()
        
        # Resize attention map to match image dimensions
        att_map = np.array(Image.fromarray(att_map).resize(img.size, Image.BICUBIC))
        
        # Normalize attention map
        att_map = att_map / np.max(att_map)
        
        # Create attention heatmap
        cmap = cm.hot
        plt.imshow(img)
        plt.imshow(att_map, alpha=0.6, cmap=cmap)
        plt.title(f"'{words[i]}'", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def demo_captioning(model, image_path, vocab, device, transform=None):
    """
    Generate a caption for a single image and visualize attention.
    
    Args:
        model: Trained ShowAttendTell model
        image_path: Path to the image file
        vocab: Vocabulary object
        device: Device to run inference on
        transform: Image transformations
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(image_tensor)
        captions, attention_maps = model.decoder.sample(
            features,
            max_len=20,
            start_idx=vocab.word2idx["<START>"],
            end_idx=vocab.word2idx["<END>"]
        )
    
    # Convert caption to text
    caption = captions[0]
    words = [vocab.idx2word[idx] for idx in caption if idx not in {0, 1, 2}]
    caption_text = ' '.join(words)
    
    print(f"Generated caption: {caption_text}")
    
    # Visualize attention
    visualize_attention(image_path, caption, attention_maps[0], vocab)
    
    return caption_text, attention_maps[0]