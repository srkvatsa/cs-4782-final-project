"""
Neural architectures: EncoderCNN, Attention, DecoderRNN, ShowAttendTell
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence
from typing import List, Tuple, Optional
import random


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
        feats = self.project(feats)  # (B, D, 14, 14)
        B, D, H, W = feats.shape
        return feats.view(B, D, H * W).permute(0, 2, 1)  # (B, L, D)


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

    def forward(
        self,
        a: torch.Tensor,
        h_prev: torch.Tensor,
        context_input: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a: (B, L, D) – annotation vectors
            h_prev: (B, H) – previous LSTM state
            context_input: optional (B, H) – something else to attend with
        """
        wh = self.W_h(h_prev).unsqueeze(1)  # (B, 1, A)
        wa = self.W_a(a)  # (B, L, A)
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


################################################################################
# Decoder – LSTM cell unrolled for T time‑steps with attention each step.
################################################################################
# Define the BeamNode class for tracking partial sequences
class BeamNode:
    def __init__(
        self, prev_node, word_id, logprob, length, hidden_state, cell_state, alpha
    ):
        self.prev_node = prev_node  # Previous node in the beam
        self.word_id = word_id  # Current word index
        self.logprob = logprob  # Cumulative log-probability
        self.length = length  # Current length of the sequence
        self.hidden_state = hidden_state  # LSTM hidden state
        self.cell_state = cell_state  # LSTM cell state
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

    def forward(
        self,
        a: torch.Tensor,
        captions: torch.Tensor,
        lengths: List[int],
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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
                inputs = captions[:, t - 1]  # Use ground truth

            context_input = prev_h if self.use_double_attention else None
            if self.use_hard_attention:
                context, alpha, beta, log_prob, entropy = self.hard_attention_step(
                    a, h, context_input, mode="train"
                )
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
        packed_logits = pack_padded_sequence(
            logits, [l - 1 for l in lengths], batch_first=True, enforce_sorted=False
        ).data

        if self.use_hard_attention:
            return (
                packed_logits,
                alphas,
                torch.stack(log_probs, dim=1),
                torch.stack(entropies, dim=1),
            )
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
                context, alpha, beta, _, _ = self.hard_attention_step(
                    features, h, context_input, mode="deterministic"
                )
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
    def beam_search(self, a, beam_size, max_len, start_idx, end_idx):
        """
        Beam search for caption generation.

        Args:
            a: Encoder features (1, L, D)
            beam_size: number of beams to keep
            max_len: maximum caption length
            start_idx: <START> token index
            end_idx: <END> token index

        Returns:
            (predicted_caption_indices, list of attention_maps)
        """
        B = a.size(0)
        assert B == 1, "Beam search only supports batch size 1"

        device = a.device
        h, c = self.init_hidden_state(a)
        inputs = torch.full((1,), start_idx, dtype=torch.long, device=device)

        prev_h = h

        root = BeamNode(
            prev_node=None,
            word_id=start_idx,
            logprob=0.0,
            length=1,
            hidden_state=h,
            cell_state=c,
            alpha=None,
        )
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

                # ---- Attention ----
                if self.use_hard_attention:
                    # Hard attention deterministic mode
                    context, alpha, _, _, _ = self.hard_attention_step(
                        a, h, context_input, mode="deterministic"
                    )
                else:
                    context, alpha, _ = self.attention(a, h, context_input)
                # -------------------

                emb = self.embed(
                    torch.tensor([node.word_id], device=device)
                )  # (1, embed_dim)
                lstm_input = torch.cat(
                    [emb, context], dim=1
                )  # (1, embed_dim + feature_dim)
                h, c = self.lstm(lstm_input, (h, c))  # (1, hidden_dim)

                output = self.fc(self.dropout(h))  # (1, vocab_size)
                scores = F.log_softmax(
                    output / self.temperature, dim=-1
                )  # (1, vocab_size)

                logprobs, top_ids = scores.topk(beam_size, dim=-1)  # (1, beam_size)

                for i in range(beam_size):
                    next_node = BeamNode(
                        prev_node=node,
                        word_id=top_ids[0, i].item(),
                        logprob=node.logprob + logprobs[0, i].item(),
                        length=node.length + 1,
                        hidden_state=h,
                        cell_state=c,
                        alpha=alpha,  # Save attention for visualization
                    )
                    next_nodes.append(next_node)

            # Keep top beam_size nodes
            nodes = sorted(next_nodes, key=lambda n: n.score(), reverse=True)[
                :beam_size
            ]

            if len(end_nodes) >= beam_size:
                break

        if len(end_nodes) == 0:
            end_nodes = nodes

        best_node = sorted(end_nodes, key=lambda n: n.score(), reverse=True)[0]

        # Trace back the path
        sequence = []
        alphas = []

        node = best_node
        while node.prev_node is not None:
            sequence.append(node.word_id)
            alphas.append(node.alpha)
            node = node.prev_node

        sequence = sequence[::-1]  # reverse to correct order
        alphas = alphas[::-1]

        return sequence, alphas

    @torch.no_grad()
    def sample(
        self,
        a: torch.Tensor,
        max_len: int,
        start_idx: int,
        end_idx: int,
        force_min_len: int = 3,
    ) -> Tuple[List[List[int]], List[List[torch.Tensor]]]:
        B = a.size(0)
        h, c = self.init_hidden_state(a)
        inputs = torch.full((B,), start_idx, dtype=torch.long, device=a.device)

        captions: List[List[int]] = [[start_idx] for _ in range(B)]
        attention_maps: List[List[torch.Tensor]] = [[] for _ in range(B)]

        prev_h = h  # for double attention

        for t in range(max_len):
            context_input = prev_h if self.use_double_attention else None
            if self.use_hard_attention:
                context, alpha, beta, _, _ = self.hard_attention_step(
                    a, h, context_input, mode="sample"
                )
            else:
                context, alpha, beta = self.attention(a, h, context_input)

            emb = self.embed(inputs)
            h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
            scores = self.fc(h)  # (B, vocab_size)

            # Sample next word (not greedy max!)
            probs = F.softmax(scores, dim=-1)  # (B, vocab_size)
            next_word = torch.multinomial(probs, 1).squeeze(
                1
            )  # Sample 1 word per example

            for i in range(B):
                captions[i].append(next_word[i].item())
                attention_maps[i].append(alpha[i].detach().cpu())

            # Prepare next inputs
            inputs = next_word
            prev_h = h

            # Early stopping: end after <END> token, but force minimum length
            all_ended = True
            for i in range(B):
                if t < force_min_len or captions[i][-1] != end_idx:
                    all_ended = False
                    break
            if all_ended:
                break

        # Cut off after first <END> token for each caption
        clean_caps = []
        for cap in captions:
            if end_idx in cap:
                cap = cap[: cap.index(end_idx) + 1]
            clean_caps.append(cap)

        return clean_caps, attention_maps

    def hard_attention_step(self, a, h_prev, context_input, mode="train"):
        """
        Args:
            mode: 'train' (default) | 'sample' | 'deterministic'
        """
        e = self.attention.v(
            torch.tanh(
                self.attention.W_a(a)
                + self.attention.W_h(h_prev).unsqueeze(1)
                + (
                    self.attention.W_c(context_input).unsqueeze(1)
                    if context_input is not None
                    else 0
                )
            )
        ).squeeze(-1)

        e = e / self.temperature
        alpha = F.softmax(e, dim=1)
        dist = torch.distributions.Categorical(alpha)
        entropy = dist.entropy()

        if mode == "train":
            if torch.rand(1).item() < 0.5:
                idx = dist.sample()
                log_prob = dist.log_prob(idx)
                context = a[torch.arange(a.size(0), device=a.device), idx]
                alpha = F.one_hot(idx, num_classes=a.size(1)).float()
            else:
                context = torch.sum(alpha.unsqueeze(-1) * a, dim=1)
                log_prob = torch.zeros(a.size(0), device=a.device)
        elif mode == "sample":
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            context = a[torch.arange(a.size(0), device=a.device), idx]
            alpha = F.one_hot(idx, num_classes=a.size(1)).float()
        elif mode == "deterministic":
            idx = torch.argmax(alpha, dim=1)
            log_prob = torch.zeros(a.size(0), device=a.device)  #
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
        self.encoder = EncoderCNN(
            backbone=backbone, finetune=kwargs.get("finetune", False)
        )
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

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: List[int],
        teacher_forcing_ratio: float = 1.0,
    ):
        a = self.encoder(images)
        out = self.decoder(a, captions, lengths, teacher_forcing_ratio)

        if self.decoder.use_hard_attention:
            scores, alphas, log_probs, entropies = out
            return scores, alphas, log_probs, entropies
        else:
            scores, alphas = out
            return scores, alphas
