"""
Vocabulary, Flickr8kDataset, collate_fn, LengthBasedBatchSampler, helpers.
"""

from __future__ import annotations
import os, random
from collections import Counter, defaultdict
from typing import List
import nltk
import torch
from torch.utils.data import Dataset
from PIL import Image


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.freq_threshold = freq_threshold

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = nltk.tokenize.word_tokenize(sentence.lower())
            frequencies.update(tokens)

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

    def decode(self, indices: List[int], skip_special_tokens=True) -> str:
        """
        Convert a list of word indices into a sentence.

        Args:
            indices: List of integer word indices
            skip_special_tokens: Whether to skip <PAD>, <START>, <END>, <UNK>

        Returns:
            A decoded sentence string
        """
        special_tokens = {
            self.word2idx["<PAD>"],
            self.word2idx["<START>"],
            self.word2idx["<END>"],
        }
        words = []
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)


def collate_fn(batch):
    """Custom collate function for padding sequences in a batch."""
    images, captions, lengths = zip(*batch)

    images = torch.stack(images, 0)

    lengths = torch.tensor(lengths)

    padded_captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=0
    )

    return images, padded_captions, lengths.tolist()


class Flickr8kDataset(Dataset):
    def __init__(self, img_folder, captions_file, vocab=None, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.vocab = vocab

        self.captions = defaultdict(list)
        self.image_names = []
        self.texts = []
        skipped = 0

        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    img_file = parts[0].split("#")[0].strip()
                    caption = parts[1].strip()
                    img_path = os.path.join(self.img_folder, img_file)
                    if os.path.exists(img_path):
                        self.image_names.append(img_file)
                        self.texts.append(caption)
                        self.captions[img_file].append(caption)  
                    else:
                        skipped += 1

        if skipped:
            print(f"[Info] Skipped {skipped} captions with missing images.")

        if vocab is None:
            self.vocab = Vocabulary().build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.length_to_indices = defaultdict(list)
        for idx, text in enumerate(self.texts):
            caption = self.vocab.numericalize(text)
            caption_length = len(caption) + 2  
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

        caption = (
            [self.vocab.word2idx["<START>"]]
            + self.vocab.numericalize(self.texts[idx])
            + [self.vocab.word2idx["<END>"]]
        )
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

    dataset = Flickr8kDataset(
        img_folder=img_folder, captions_file=captions_file, transform=transform
    )
    vocab = dataset.vocab

    def load_split(file_name):
        path = os.path.join(data_dir, file_name)
        with open(path, "r") as f:
            return set(line.strip() for line in f)

    train_images = load_split("Flickr_8k.trainImages.txt")
    val_images = load_split("Flickr_8k.devImages.txt")
    test_images = load_split("Flickr_8k.testImages.txt")

    train_indices = [
        i for i, name in enumerate(dataset.image_names) if name in train_images
    ]
    val_indices = [
        i for i, name in enumerate(dataset.image_names) if name in val_images
    ]
    test_indices = [
        i for i, name in enumerate(dataset.image_names) if name in test_images
    ]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Vocabulary size: {len(vocab)}")
    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset, vocab


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

        if hasattr(dataset, "length_to_indices"):
            self.length_to_indices = dataset.length_to_indices
        elif hasattr(dataset, "dataset") and hasattr(
            dataset.dataset, "length_to_indices"
        ):
            self.length_to_indices = defaultdict(list)
            original_dataset = dataset.dataset
            subset_indices = dataset.indices

            for i, idx in enumerate(subset_indices):
                text = original_dataset.texts[idx]
                caption = original_dataset.vocab.numericalize(text)
                caption_length = len(caption) + 2  
                self.length_to_indices[caption_length].append(i)
        else:
            raise TypeError(
                "Dataset must have length_to_indices attribute or be a Subset of such a dataset"
            )

        self.available_lengths = sorted(list(self.length_to_indices.keys()))

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
            length = random.choice(lengths)
            indices = self.length_to_indices[length]

            batch = (
                random.sample(indices, self.batch_size)
                if len(indices) >= self.batch_size
                else random.choices(indices, k=self.batch_size)
            )

            all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self.total_batches
