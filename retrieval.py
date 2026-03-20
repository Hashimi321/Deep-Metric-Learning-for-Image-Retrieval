# retrieval.py
# Recall@K evaluation, t-SNE visualization, retrieval grid visualization
#
# Usage:
#   python retrieval.py
#       --embeddings_path "embeddings/exp1_contrastive_test_embeddings.npy"
#       --labels_path     "embeddings/exp1_contrastive_test_labels.npy"
#       --exp_name        "exp1_contrastive"

import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from torchvision import datasets

from dataset import get_transforms
from inference import load_model, preprocess_image


def recall_at_k(embeddings, labels, k=1):
    """
    Computes Recall@K using nearest neighbor retrieval.

    For each query, retrieves K nearest neighbors (excluding itself)
    and checks if the correct class appears in the top-K results.

    Args:
        embeddings (np.array): shape (N, D)
        labels     (np.array): shape (N,)
        k          (int)     : number of neighbors to retrieve

    Returns:
        float: Recall@K score between 0.0 and 1.0
    """
    # Convert to torch tensors for efficient distance computation
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)

    # Compute full pairwise distance matrix (N x N)
    dist_matrix =torch.cdist(emb_tensor, emb_tensor, p=2)

    # Exclude self-distances by setting diagonal to infinity
    dist_matrix.fill_diagonal_(float('inf'))

    correct = 0

    for i in range(len(labels)):
        # Get distances from query i to all others
        dists = dist_matrix[i]

        # Get indices of K nearest neighbors
        # torch.topk with largest=False gives smallest distances
        _, top_k_indices = torch.topk(dists, k=k, largest=False)

        # Get labels of those K neighbors
        neighbor_labels = labels[top_k_indices.numpy()]

        # Check if query label appears in any of the K neighbors
        if labels[i] in neighbor_labels:
            correct += 1

    recall = correct / len(labels)
    return recall


def plot_tsne(embeddings, labels, class_names, title, save_path,
              max_samples=1000):
    """
    Reduces embeddings to 2D with t-SNE and saves a scatter plot
    colored by class label.

    Args:
        embeddings  (np.array): shape (N, D)
        labels      (np.array): shape (N,)
        class_names (list)    : list of class name strings
        title       (str)     : plot title
        save_path   (str)     : where to save the .png file
        max_samples (int)     : subsample for speed (t-SNE is slow on CPU)
    """
    # Subsample if too many points — t-SNE is O(N²)
    if len(embeddings) > max_samples:
        indices    = np.random.choice(
            len(embeddings), max_samples, replace=False
        )
        embeddings = embeddings[indices]
        labels     = labels[indices]

    print(f"  Running t-SNE on {len(embeddings)} samples...")

    # Reduce to 2D
    tsne       = TSNE(n_components=2, perplexity=30, random_state=42)
    coords_2d  = tsne.fit_transform(embeddings)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get unique classes in this sample
    unique_labels = np.unique(labels)
    num_classes   = len(unique_labels)

    # Color map — one color per class
    cmap   = plt.cm.get_cmap('tab20', num_classes)
    colors = {label: cmap(i) for i, label in enumerate(unique_labels)}

    # Plot each class
    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c     = [colors[label]],
            label = class_names[label],
            alpha = 0.6,
            s     = 15
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')

    # Legend — only show if manageable number of classes
    if num_classes <= 20:
        ax.legend(
            loc='upper right',
            fontsize=6,
            markerscale=1.5,
            ncol=2
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE plot saved → {save_path}")


def show_retrieval(query_idx, embeddings, labels, image_paths,
                   class_names, k=5, save_path=None):
    """
    Displays query image alongside its K nearest neighbors in a grid.
    Labels correct matches in green, wrong matches in red.

    Args:
        query_idx   (int)     : index of query image
        embeddings  (np.array): shape (N, D)
        labels      (np.array): shape (N,)
        image_paths (list)    : list of image file paths (same order)
        class_names (list)    : class name strings
        k           (int)     : number of neighbors to show
        save_path   (str)     : where to save the figure
    """
    emb_tensor  = torch.tensor(embeddings, dtype=torch.float32)
    dist_matrix = torch.cdist(emb_tensor, emb_tensor, p=2)
    dist_matrix.fill_diagonal_(float('inf'))

    # Get top-K nearest neighbors for this query
    dists = dist_matrix[query_idx] 
    _, top_k_indices = torch.topk(dists, k=k, largest=False)
    top_k_indices    = top_k_indices.numpy()

    # Build figure: 1 row, k+1 columns (query + k neighbors)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))

    def load_img(path):
        """Load and return image as numpy array for display."""
        from PIL import Image
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        return np.array(img)

    # Column 0: Query image
    query_img = load_img(image_paths[query_idx])
    axes[0].imshow(query_img)
    axes[0].set_title(
        f'QUERY\n{class_names[labels[query_idx]]}',
        fontsize=9, fontweight='bold', color='blue'
    )
    axes[0].axis('off')

    # Columns 1 to K: Nearest neighbors
    query_label = labels[query_idx]

    for rank, neighbor_idx in enumerate(top_k_indices):
        neighbor_label = labels[neighbor_idx]
        neighbor_img   = load_img(image_paths[neighbor_idx])

        axes[rank + 1].imshow(neighbor_img)

        # Green title = correct class, Red = wrong class
        is_correct = (neighbor_label == query_label)
        color      = 'green' if is_correct else 'red'
        marker     = '✓' if is_correct else '✗'

        axes[rank + 1].set_title(
            f'Top-{rank+1} {marker}\n{class_names[neighbor_label]}',
            fontsize=9,
            color=color
        )
        axes[rank + 1].axis('off')

    plt.suptitle(
        f'Query: {class_names[query_label]}',
        fontsize=12,
        fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Retrieval grid saved → {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate retrieval and generate visualizations'
    )
    parser.add_argument('--embeddings_path', type=str, required=True)
    parser.add_argument('--labels_path',     type=str, required=True)
    parser.add_argument('--data_path',       type=str, required=True,
                        help='Path to test split folder')
    parser.add_argument('--exp_name',        type=str, required=True)
    parser.add_argument('--model_path',      type=str, required=True)
    args = parser.parse_args()

    os.makedirs('graphs', exist_ok=True)

    # Load saved embeddings and labels
    embeddings  = np.load(args.embeddings_path)
    labels      = np.load(args.labels_path)

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded labels:     {labels.shape}")

    # Load class names from test folder structure
    from torchvision import datasets as tvdatasets
    dummy_ds    = tvdatasets.ImageFolder(args.data_path,
                                         transform=get_transforms())
    class_names = dummy_ds.classes
    image_paths = [s[0] for s in dummy_ds.samples]

    # ── Recall@K ─────────────────────────────────────
    print("\nComputing Recall@1...")
    r1 = recall_at_k(embeddings, labels, k=1)
    print(f"  Recall@1 : {r1:.4f} ({r1*100:.2f}%)")

    print("Computing Recall@5...")
    r5 = recall_at_k(embeddings, labels, k=5)
    print(f"  Recall@5 : {r5:.4f} ({r5*100:.2f}%)")

    # ── t-SNE ─────────────────────────────────────────
    print("\nGenerating t-SNE plot...")
    plot_tsne(
        embeddings  = embeddings,
        labels      = labels,
        class_names = class_names,
        title       = f't-SNE Visualization — {args.exp_name}',
        save_path   = f'graphs/tsne_{args.exp_name}.png'
    )

    # ── Retrieval Grids ───────────────────────────────
    print("\nGenerating retrieval grids...")

    # Pick 10 diverse query indices spread across classes
    query_indices = []
    seen_classes  = set()
    for i, label in enumerate(labels):
        if label not in seen_classes:
            query_indices.append(i)
            seen_classes.add(label)
        if len(query_indices) >= 10:
            break

    for q_idx in query_indices:
        class_name = class_names[labels[q_idx]]
        save_path  = (f'graphs/retrieval_{args.exp_name}'
                      f'_query{q_idx}_{class_name}.png')
        show_retrieval(
            query_idx   = q_idx,
            embeddings  = embeddings,
            labels      = labels,
            image_paths = image_paths,
            class_names = class_names,
            k           = 5,
            save_path   = save_path
        )

    print(f"\n{'='*50}")
    print(f"Results for {args.exp_name}:")
    print(f"  Recall@1 : {r1:.4f}")
    print(f"  Recall@5 : {r5:.4f}")
    print(f"  Plots saved to graphs/")