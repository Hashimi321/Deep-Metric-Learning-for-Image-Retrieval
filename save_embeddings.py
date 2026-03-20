
# save_embeddings.py
# Precomputes and saves embeddings for train/val/test splits
# Run once after training to avoid recomputing embeddings repeatedly
#
# Usage:
#   python save_embeddings.py
#       --model_path "weights/exp1/exp1_contrastive_epoch2.pt"
#       --data_path  "caltech_split"
#       --exp_name   "exp1_contrastive"

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import EmbeddingNet
from dataset import get_transforms
from inference import load_model


def compute_embeddings_for_split(model, split_path,
                                  transform, device, batch_size=64):
    """
    Runs all images in a split through the model and returns
    embeddings and labels as numpy arrays.

    Args:
        model      : trained EmbeddingNet in eval mode
        split_path : path to folder with class subfolders
        transform  : preprocessing pipeline
        device     : torch device
        batch_size : how many images per forward pass

    Returns:
        embeddings : numpy array, shape (N, embedding_dim)
        labels     : numpy array, shape (N,)
        class_names: list of class name strings
    """
    # Load dataset — plain ImageFolder, no pairs needed
    dataset = datasets.ImageFolder(
        root=split_path,
        transform=transform
    )

    # DataLoader — no shuffling so order matches labels
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # IMPORTANT: keep order consistent
        num_workers=0
    )

    all_embeddings = []
    all_labels     = []

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            # Move images to device
            images = images.to(device)

            # Forward pass through model
            embs = model(images)

            # Move results to CPU and convert to numpy
            all_embeddings.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())

            # Progress update every 10 batches
            if batch_idx % 10 == 0:
                processed = min(
                    (batch_idx + 1) * batch_size, len(dataset)
                )
                print(f"    {processed}/{len(dataset)} images processed")

    # Concatenate all batches into single arrays
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels     = np.concatenate(all_labels, axis=0)

    print(f"    Done. Embeddings shape: {embeddings.shape}")
    print(f"    Labels shape:           {labels.shape}")

    return embeddings, labels, dataset.classes


def save_embeddings_for_experiment(model_path, data_path,
                                    exp_name, embedding_dim=128):
    """
    Computes and saves embeddings for all three splits.

    Args:
        model_path    : path to .pt checkpoint
        data_path     : root with train/, val/, test/ subfolders
        exp_name      : prefix for saved files e.g. 'exp1_contrastive'
        embedding_dim : must match training
    """
    os.makedirs('embeddings', exist_ok=True)

    device    = torch.device('cuda' if torch.cuda.is_available()
                             else 'cpu')
    transform = get_transforms()

    # Load trained model
    model = load_model(
        model_path=model_path,
        embedding_dim=embedding_dim,
        device=str(device)
    )

    # Process each split
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split_name)

        print(f"\nComputing embeddings for {split_name} split...")

        embeddings, labels, class_names = compute_embeddings_for_split(
            model     = model,
            split_path= split_path,
            transform = transform,
            device    = device
        )

        # Build save paths
        emb_path   = os.path.join(
            'embeddings', f'{exp_name}_{split_name}_embeddings.npy'
        )
        label_path = os.path.join(
            'embeddings', f'{exp_name}_{split_name}_labels.npy'
        )

        # Save to disk
        np.save(emb_path, embeddings)
        
        np.save(label_path, labels)

        print(f"    Saved: {emb_path}")
        print(f"    Saved: {label_path}")

    print(f"\nAll embeddings saved to embeddings/")
    print(f"Files saved:")
    for f in sorted(os.listdir('embeddings')):
        path = os.path.join('embeddings', f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {f:<50} {size:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Precompute and save embeddings for all splits'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset root (contains train/, val/, test/)'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        required=True,
        help='Experiment name prefix e.g. exp1_contrastive'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension (must match training, default 128)'
    )
    args = parser.parse_args()

    save_embeddings_for_experiment(
        model_path    = args.model_path,
        data_path     = args.data_path,
        exp_name      = args.exp_name,
        embedding_dim = args.embedding_dim
    )