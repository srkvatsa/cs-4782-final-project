"""
Training / evaluation routines, checkpoints, metric helpers.
"""

from __future__ import annotations
import pathlib, json, math, time, os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.util import ngrams
import collections
import math

from models import ShowAttendTell
from datasets import collate_fn, LengthBasedBatchSampler


def get_teacher_forcing_ratio(
    epoch, max_epochs, start_ratio=1.0, min_ratio=0.5, decay="linear"
):
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
    if decay == "linear":
        return max(
            min_ratio, start_ratio - (start_ratio - min_ratio) * (epoch / max_epochs)
        )
    elif decay == "exp":
        return max(min_ratio, start_ratio * (0.95**epoch))
    elif decay == "sigmoid":
        x = 10 * (epoch - max_epochs / 2) / max_epochs
        sigmoid = 1 / (1 + math.exp(x))
        return min_ratio + (start_ratio - min_ratio) * sigmoid
    else:
        return start_ratio  


def doubly_stochastic_regularization(attn_weights, lambda_reg=1.0):
    """
    attn_weights: list of T tensors, each (batch_size, num_locations)
    returns: regularization term
    """
    attn_tensor = torch.stack(attn_weights, dim=0)  
    attn_sum = attn_tensor.sum(dim=0)  
    reg = ((1.0 - attn_sum) ** 2).mean()  
    return lambda_reg * reg


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    grad_clip: float = 5.0
    save_every: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 5  
    lambda_reg: float = 1.0  
    length_based_sampling: bool = True  
    early_stopping: bool = True  
    teacher_forcing: bool = (
        True  
    )
    tf_start_ratio: float = 1.0  
    tf_min_ratio: float = 0.5  
    tf_decay: str = "sigmoid"  


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_bleu,
    best_meteor,
    patience_counter,
    save_path,
    bleu_log=None,
):
    """Save a complete checkpoint that can be used to resume training."""
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_bleu": best_bleu,
        "best_meteor": best_meteor,  
        "patience_counter": patience_counter,
    }
    torch.save(checkpoint_data, save_path)
    print(f"Checkpoint saved to {save_path}")

    if bleu_log is not None:
        log_name = f"eval_scores_epoch{epoch}.json"
        log_path = os.path.join(os.path.dirname(save_path), log_name)
        with open(log_path, "w") as f:
            json.dump(bleu_log, f, indent=2)
        print(f"Saved evaluation scores log to {log_path}")


def load_checkpoint(model, optimizer=None, checkpoint_path=None, device=None):
    """Load a checkpoint and return the epoch to start from and other training state."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 1, 0.0, 0.0, 0, []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_bleu = checkpoint.get("best_bleu", 0.0)
    best_meteor = checkpoint.get("best_meteor", 0.0)  
    patience_counter = checkpoint.get("patience_counter", 0)

    log_path = os.path.join(
        os.path.dirname(checkpoint_path), f"eval_scores_epoch{checkpoint['epoch']}.json"
    )
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            bleu_log = json.load(f)
        print(
            f"Loaded evaluation score log with {len(bleu_log)} entries from {log_path}"
        )
    else:
        bleu_log = []

    print(f"Loaded checkpoint from epoch {start_epoch-1}")
    print(f"Best BLEU score so far: {best_bleu:.4f}")
    print(f"Best METEOR score so far: {best_meteor:.4f}")
    print(f"Patience counter: {patience_counter}")

    return start_epoch, best_bleu, best_meteor, patience_counter, bleu_log


def bleu_scores_without_bp(references, hypotheses):

    max_n = 4
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref_list, hyp in zip(references, hypotheses):
        hyp_ngrams = [collections.Counter(ngrams(hyp, i + 1)) for i in range(max_n)]
        ref_ngram_counters = [
            [collections.Counter(ngrams(ref, i + 1)) for ref in ref_list]
            for i in range(max_n)
        ]

        for i in range(max_n):
            ref_max = collections.Counter()
            for ref_counts in ref_ngram_counters[i]:
                ref_max |= ref_counts
            overlap = hyp_ngrams[i] & ref_max
            clipped_counts[i] += sum(overlap.values())
            total_counts[i] += max(sum(hyp_ngrams[i].values()), 1)

    precisions = [clipped_counts[i] / total_counts[i] for i in range(max_n)]

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
            features = a[i : i + 1]
            sampled_ids = model.decoder.greedy_search(
                features,
                max_len=max_len,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"],
            )

            pred_tokens = [
                vocab.idx2word[idx]
                for idx in sampled_ids
                if idx
                not in {
                    vocab.word2idx["<PAD>"],
                    vocab.word2idx["<START>"],
                    vocab.word2idx["<END>"],
                }
            ]
            hypothesis = " ".join(pred_tokens)
            hypotheses.append(pred_tokens)

            img_idx = (
                loader.dataset.indices[i] if hasattr(loader.dataset, "indices") else i
            )
            img_name = loader.dataset.dataset.image_names[img_idx]
            ref_caps = loader.dataset.dataset.captions[img_name]

            ref_token_lists = []
            ref_strings = []
            for ref in ref_caps:
                tokens = nltk.word_tokenize(ref.lower())
                ref_strings.append(" ".join(tokens))
                ref_token_lists.append(tokens)

            references.append(ref_token_lists)

            hypothesis_tokens = pred_tokens
            meteor = max(
                [
                    single_meteor_score(nltk.word_tokenize(ref), hypothesis_tokens)
                    for ref in ref_strings
                ]
            )
            meteor_scores.append(meteor)

    for i in range(5):
        print(f"\nSample {i+1}")
        print("Hypothesis:", " ".join(hypotheses[i]))
        print("Reference:", [" ".join(ref) for ref in references[i]])
        print("METEOR:", f"{meteor_scores[i]:.4f}")

    bleu1, bleu2, bleu3, bleu4 = bleu_scores_without_bp(references, hypotheses)
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu3, bleu4, meteor_avg


@torch.no_grad()
def evaluate_bleu_and_meteor_with_beam(
    model, loader, device, vocab, beam_size=8, max_len=20
):
    model.eval()
    references = []
    hypotheses = []
    meteor_scores = []

    for images, captions, lengths in tqdm(loader, desc="Evaluating with Beam Search"):
        images = images.to(device)
        batch_size = images.size(0)
        a = model.encoder(images)

        for i in range(batch_size):
            features = a[i : i + 1]
            sampled_ids, _ = model.decoder.beam_search(
                features,
                beam_size=beam_size,
                max_len=max_len,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"],
            )

            pred_tokens = [
                vocab.idx2word[idx]
                for idx in sampled_ids
                if idx
                not in {
                    vocab.word2idx["<PAD>"],
                    vocab.word2idx["<START>"],
                    vocab.word2idx["<END>"],
                }
            ]
            hypothesis = " ".join(pred_tokens)
            hypotheses.append(pred_tokens)

            img_idx = (
                loader.dataset.indices[i] if hasattr(loader.dataset, "indices") else i
            )
            img_name = loader.dataset.dataset.image_names[img_idx]
            ref_caps = loader.dataset.dataset.captions[img_name]

            ref_token_lists = []
            ref_strings = []
            for ref in ref_caps:
                tokens = nltk.word_tokenize(ref.lower())
                ref_strings.append(" ".join(tokens))
                ref_token_lists.append(tokens)

            references.append(ref_token_lists)


            hypothesis_tokens = pred_tokens  
            meteor = max(
                [
                    single_meteor_score(nltk.word_tokenize(ref), hypothesis_tokens)
                    for ref in ref_strings
                ]
            )
            meteor_scores.append(meteor)

    for i in range(5):
        print(f"\nSample {i+1}")
        print("Hypothesis:", " ".join(hypotheses[i]))
        print("Reference:", [" ".join(ref) for ref in references[i]])
        print("METEOR:", f"{meteor_scores[i]:.4f}")

    bleu1, bleu2, bleu3, bleu4 = bleu_scores_without_bp(references, hypotheses)
    meteor_avg = sum(meteor_scores) / len(meteor_scores)

    return bleu1, bleu2, bleu3, bleu4, meteor_avg


def train_and_evaluate(
    model: ShowAttendTell,
    train_dataset,
    val_dataset,
    criterion,
    cfg: TrainingConfig,
    vocab_size: int,
    save_dir: str | pathlib.Path = "results/checkpoints",
    resume_from: str | None = None,
    start_epoch: int = 1,
):
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.8,
        patience=3,
        verbose=True,
        min_lr=1e-6,  
    )

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    torch.cuda.empty_cache()

    best_bleu = 0.0
    best_meteor = 0.0  
    patience_counter = 0
    bleu_log = []
    if resume_from and model.decoder.use_hard_attention:
        start_epoch, best_bleu, best_meteor, _, bleu_log = load_checkpoint(
            model, optimizer=None, checkpoint_path=resume_from, device=cfg.device
        )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-5
        )  

        print("Switched to hard attention with fresh optimizer state")
    else:
        if resume_from:
            start_epoch, best_bleu, best_meteor, patience_counter, bleu_log = (
                load_checkpoint(model, optimizer, resume_from, cfg.device)
            )

    if cfg.length_based_sampling:
        train_sampler = LengthBasedBatchSampler(
            train_dataset, cfg.batch_size, shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )
    best_model_path = save_dir / "best_model.pt"
    best_model_meteor_path = (
        save_dir / "best_model_meteor.pt"
    )  

    for epoch in range(start_epoch, cfg.epochs + 1):
        start_time = time.time()

        teacher_forcing_ratio = get_teacher_forcing_ratio(
            epoch=epoch - 1,  
            max_epochs=cfg.epochs,
            start_ratio=1.00,
            min_ratio=0.5,
            decay="linear",  
        )

        print(
            f"Epoch {epoch}/{cfg.epochs}, Teacher forcing ratio: {teacher_forcing_ratio:.3f}"
        )

        if epoch == 3 and hasattr(model.encoder, "features"):
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
                model.decoder.temperature = max(0.5, 1.0 - 0.5 * ((epoch - 51) / 19))
                scores, alphas, log_probs, entropies = model(
                    images, captions, lengths, teacher_forcing_ratio
                )

                targets = pack_padded_sequence(
                    captions[:, 1:],
                    [l - 1 for l in lengths],
                    batch_first=True,
                    enforce_sorted=False,
                ).data

                ce_loss_all = criterion(scores, targets)  
                batch_size = log_probs.size(0)

                ce_loss_per_sample = ce_loss_all.view(batch_size, -1).mean(
                    dim=1
                )  
                ce_loss = ce_loss_per_sample.mean()  

                with torch.no_grad():
                    reward = -ce_loss_per_sample

                    if baseline is None:
                        baseline = reward.mean().item()
                    else:
                        baseline = 0.95 * baseline + 0.05 * reward.mean().item()

                    advantage = reward - baseline

                    if advantage.std() > 0:
                        advantage = advantage / (advantage.std() + 1e-8)

                log_probs_total = log_probs.sum(dim=1)  
                reinforce_loss = -(log_probs_total * advantage).mean()

                entropy_bonus = 0.05 * entropies.sum(dim=1).mean()  

                reinforce_loss = reinforce_loss - entropy_bonus

                total_loss = ce_loss + reinforce_loss
            else:
                scores, alphas = model(images, captions, lengths, teacher_forcing_ratio)
                targets = pack_padded_sequence(
                    captions[:, 1:],
                    [l - 1 for l in lengths],
                    batch_first=True,
                    enforce_sorted=False,
                ).data
                ce_loss = criterion(scores, targets)

                reg_loss = doubly_stochastic_regularization(
                    [alphas[:, t, :] for t in range(alphas.size(1))],
                    lambda_reg=cfg.lambda_reg,
                )
                total_loss = ce_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            torch.cuda.empty_cache()

            running_loss += total_loss.item()

            if model.decoder.use_hard_attention:
                pbar.set_postfix(
                    {
                        "loss": total_loss.item(),
                        "ce": ce_loss.item(),
                        "reinforce": reinforce_loss.item(),
                        "tf_ratio": teacher_forcing_ratio,
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": total_loss.item(),
                        "ce": ce_loss.item(),
                        "reg": reg_loss.item(),
                        "tf_ratio": teacher_forcing_ratio,
                    }
                )

        train_loss = running_loss / len(train_loader)
        current_bleu1, current_bleu2, current_bleu3, current_bleu4, current_meteor = (
            evaluate_bleu_and_meteor_with_greedy(
                model, val_loader, cfg.device, val_dataset.dataset.vocab
            )
        )
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} – Train Loss: {train_loss:.4f}, "
            f"BLEU-1: {current_bleu1:.4f}, BLEU-2: {current_bleu2:.4f}, BLEU-3: {current_bleu3:.4f}, BLEU-4: {current_bleu4:.4f}, "
            f"METEOR: {current_meteor:.4f}, Time: {epoch_time:.1f}s"
        )

        bleu_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "bleu_1": current_bleu1,
                "bleu_2": current_bleu2,
                "bleu_3": current_bleu3,
                "bleu_4": current_bleu4,
                "meteor": current_meteor,
                "time": epoch_time,
            }
        )

        combined_score = (current_bleu4 + current_meteor) / 2
        if epoch > 40:
            scheduler.step(combined_score)

        if epoch % cfg.save_every == 0:
            full_ckpt_path = save_dir / f"checkpoint_epoch{epoch}.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                best_meteor,
                patience_counter,
                full_ckpt_path,
                bleu_log,
            )

        if current_bleu4 < best_bleu and current_meteor < best_meteor:
            patience_counter += 1
            print(f"No improvement in scores for {patience_counter} epochs")

            if cfg.early_stopping and patience_counter >= cfg.patience:
                print(f"Early stopping after {epoch} epochs without improvement")
                break

        if current_bleu4 > best_bleu:
            best_bleu = current_bleu4
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                best_meteor,
                patience_counter,
                best_model_path,
                bleu_log,
            )
            print(f"New best BLEU-4 score: {best_bleu:.4f} - Saving model")

        if current_meteor > best_meteor:
            best_meteor = current_meteor
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                best_meteor,
                patience_counter,
                best_model_meteor_path,
                bleu_log,
            )
            print(f"New best METEOR score: {best_meteor:.4f} - Saving model")

    start_epoch, best_bleu, best_meteor, _, _ = load_checkpoint(
        model, optimizer, best_model_path, cfg.device
    )
    final_bleu1, final_bleu2, final_bleu3, final_bleu4, final_meteor = (
        evaluate_bleu_and_meteor_with_greedy(
            model, val_loader, cfg.device, val_dataset.dataset.vocab
        )
    )
    print(
        f"Training completed. Best BLEU-4 score: {final_bleu4:.4f}, BLEU-1: {final_bleu1:.4f}, BLEU-2: {final_bleu2:.4f}, BLEU-3: {final_bleu3:.4f}, METEOR: {final_meteor:.4f}"
    )

    return model, final_bleu1, final_bleu2, final_bleu3, final_bleu4, final_meteor
