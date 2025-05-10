"""
Utilities for finding samples and plotting attention maps.
"""

from __future__ import annotations
import torch
from PIL import Image
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

# matplot / cv2 imports go here


def find_idx(dataset, vocab, target_caption_text):
    """
    Search the dataset for an image that has the given reference caption text.
    Returns the dataset index if found, else None.
    """
    for idx in range(len(dataset)):
        _, caption_tensor, _ = dataset[idx]
        caption_words = [
            vocab.idx2word[token.item()]
            for token in caption_tensor
            if token.item() not in {0, 1, 2}
        ]  # Exclude special tokens
        caption_text = " ".join(caption_words)

        if target_caption_text.strip().lower() in caption_text.strip().lower():
            print("Found: ", target_caption_text)
            return idx  # Found it!

    print(target_caption_text, "Not Found")
    return None  # Not found


def generate_sample_captions(
    model, dataset, vocab, device, indices, num_samples=15, beam_size=10
):
    model.eval()
    # indices = random.sample(range(len(dataset)), num_samples)
    # print(indices)
    # indices = [find_idx(dataset, vocab, "frisbee"),
    #           #  find_idx(dataset, vocab, "giraffe")
    # ]

    samples = []
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, caption, _ = dataset[idx]
            image = image.unsqueeze(0).to(device)

            # Get image features
            features = model.encoder(image)

            # Generate caption using BEAM SEARCH
            pred_indices, attention_maps = model.decoder.beam_search(
                features,
                beam_size=beam_size,
                max_len=20,
                start_idx=vocab.word2idx["<START>"],
                end_idx=vocab.word2idx["<END>"],
            )

            # Convert prediction to text
            pred_text = " ".join(
                [vocab.idx2word[idx] for idx in pred_indices if idx not in {0, 1, 2}]
            )  # Exclude PAD, START, END

            # Get reference caption
            ref_text = " ".join(
                [
                    vocab.idx2word[idx.item()]
                    for idx in caption
                    if idx.item() not in {0, 1, 2}
                ]
            )

            samples.append(
                {
                    "prediction": pred_text,
                    "prediction_indices": pred_indices,
                    "reference": ref_text,
                    "attention_maps": attention_maps,
                    "dataset_idx": idx,
                }
            )

    return samples


def visualize_attention_paper_style(
    image_path,
    caption_indices,
    attention_maps,
    vocab,
    reference_caption=None,
    save_path=None,
):
    """
    Visualize attention weights in the style of the Show, Attend and Tell paper with
    smooth attention regions over the image in a horizontal grid layout.
    Supports both soft and hard attention visualizations.
    """

    def generate_gaussian_blob(index, map_size=(14, 14), sigma=1.0):
        """Create a soft Gaussian blob centered at the given index (for hard attention)."""
        blob = np.zeros(map_size)
        h, w = map_size
        y, x = divmod(index, w)
        blob[y, x] = 1.0
        blob = scipy.ndimage.gaussian_filter(blob, sigma=sigma)
        blob /= blob.max() + 1e-8
        return blob

    # Open image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    img_height, img_width = img_np.shape[:2]

    # Filter out special tokens
    words = [vocab.idx2word[idx] for idx in caption_indices if idx not in {0, 1, 2}]
    prediction = " ".join(words)
    num_words = min(len(words), len(attention_maps))

    # Calculate layout dimensions
    max_per_row = 4
    num_rows = math.ceil((num_words + 1) / max_per_row)  # +1 for original image

    # Create figure with appropriate size
    # Adjust figure size based on number of rows and columns
    fig_width = 16  # wider figure
    fig_height = 4 * num_rows  # height depends on number of rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Add space between rows
    plt.subplots_adjust(hspace=0.3)

    # Plot original image with full caption (first position)
    plt.subplot(num_rows, max_per_row, 1)
    plt.imshow(img_np)
    if reference_caption:
        plt.title(
            f"Reference: {reference_caption}\nPrediction: {prediction}", fontsize=10
        )
    else:
        plt.title(f"Prediction: {prediction}", fontsize=10)
    plt.axis("off")

    # Plot attention maps
    for i in range(num_words):
        # Calculate subplot position (add 1 because we already used position 1)
        position = i + 2

        # Create subplot
        plt.subplot(num_rows, max_per_row, position)

        # Process attention map
        att_map = attention_maps[i].cpu().detach().numpy()

        # If attention is nearly one-hot, treat as hard attention and generate a blob
        if np.count_nonzero(att_map > 0.9 * att_map.max()) == 1:
            max_idx = att_map.argmax()
            att_map = generate_gaussian_blob(max_idx, map_size=(14, 14), sigma=1.0)
        else:
            # Soft attention: reshape to 2D map
            att_map = att_map.reshape(14, 14)

        # Normalize attention map
        att_map = att_map / (att_map.max() + 1e-8)  # Normalize to [0,1]

        # Create a high-resolution version using cv2 for better quality
        att_map_large = cv2.resize(
            att_map, (img_width, img_height), interpolation=cv2.INTER_CUBIC
        )

        # Apply additional strong Gaussian blur for very smooth appearance
        sigma = max(img_width, img_height) / 50
        att_map_large = cv2.GaussianBlur(
            att_map_large, (0, 0), sigma  # Auto-compute kernel size from sigma
        )

        # Display original image
        plt.imshow(img_np)

        # Display the attention as white highlights on the original image
        plt.imshow(
            att_map_large,
            cmap="gray",  # Simple white highlights
            alpha=0.7,  # Adjust transparency as needed
            interpolation="bilinear",
            vmin=0,
            vmax=1,
        )

        plt.title(f"'{words[i]}'", fontsize=10)
        plt.axis("off")

    # Hide any unused subplots
    for i in range(num_words + 2, num_rows * max_per_row + 1):
        ax = fig.add_subplot(num_rows, max_per_row, i)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
