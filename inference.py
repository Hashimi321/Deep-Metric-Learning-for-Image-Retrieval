
# inference.py
# Loads a trained EmbeddingNet and generates embeddings for input images
# Can be run independently of training code
#
# Usage (single image):
#   python inference.py --model_path "weights/exp1/exp1_contrastive_epoch2.pt"
#                       --image_path "caltech-101/accordion/image_0001.jpg"
#
# Usage (multiple images):
#   python inference.py --model_path "weights/exp1/exp1_contrastive_epoch2.pt"
#                       --image_path "img1.jpg" "img2.jpg" "img3.jpg"

import os
import argparse

import torch
import numpy as np
from PIL import Image

from model import EmbeddingNet
from dataset import get_transforms


def load_model(model_path, embedding_dim=128, device='cpu'):
    """
    Loads a trained EmbeddingNet from a checkpoint file.

    Args:
        model_path    (str): path to .pt checkpoint file
        embedding_dim (int): must match what was used during training
        device        (str): 'cpu' or 'cuda'

    Returns:
        model: loaded EmbeddingNet in eval mode
    """
    # Create empty model with same architecture
    model = EmbeddingNet(embedding_dim)

    # Load checkpoint dictionary from disk
    # map_location ensures it loads on CPU even if saved on GPU
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # Load saved weights into model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    # This disables dropout and sets batchnorm to inference behavior
    model.eval()

    model = model.to(device)

    print(f"Model loaded from: {model_path}")
    print(f"  Trained for {checkpoint['epoch']} epoch(s)")
    print(f"  Final loss: {checkpoint['loss']:.4f}")

    return model


def preprocess_image(image_path, transform):
    """
    Loads and preprocesses a single image for inference.

    Args:
        image_path (str): path to image file
        transform       : torchvision transforms pipeline

    Returns:
        Tensor: preprocessed image tensor, shape (1, 3, 224, 224)
    """
    # Load image using PIL
    img = Image.open(image_path)

    # Apply transforms (resize, normalize, etc.)
    tensor = transform(img)

    # Add batch dimension
    # Model expects (B, 3, H, W) — we have (3, H, W) for one image
    # tensor.unsqueeze(0) adds dimension at position 0
    tensor = tensor.unsqueeze(0)
    return tensor


def get_embedding(model, image_path, transform, device='cpu'):
    """
    Generates a normalized embedding vector for one image.

    Args:
        model      : loaded EmbeddingNet in eval mode
        image_path : path to image file
        transform  : preprocessing pipeline
        device     : torch device

    Returns:
        numpy array: embedding vector, shape (128,)
    """
    # Preprocess image
    tensor = preprocess_image(image_path, transform)

    # Move tensor to same device as model
    tensor = tensor.to(device)

    # Forward pass — no gradients needed during inference
    with torch.no_grad():
        embedding = model(tensor)

    # Remove batch dimension and convert to numpy
    # embedding shape: (1, 128) → squeeze → (128,)
    embedding = embedding.squeeze(0).cpu().numpy()

    return embedding


def get_embeddings_batch(model, image_paths, transform, device='cpu'):
    """
    Generates embeddings for multiple images efficiently.

    Args:
        model       : loaded EmbeddingNet in eval mode
        image_paths : list of image file paths
        transform   : preprocessing pipeline
        device      : torch device

    Returns:
        numpy array: embeddings matrix, shape (N, 128)
    """
    all_embeddings = []

    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"  WARNING: Image not found: {image_path}")
            continue

        embedding = get_embedding(model, image_path, transform, device)
        all_embeddings.append(embedding)
        print(f"  Processed: {os.path.basename(image_path)}"
              f" → embedding shape: {embedding.shape}")

    if len(all_embeddings) == 0:
        print("No valid images found.")
        return None

    # Stack into matrix: (N, 128)
    embeddings_matrix = np.stack(all_embeddings, axis=0)

    return embeddings_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate embeddings for input images'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        nargs='+',            # accepts one or more image paths
        required=True,
        help='Path(s) to input image(s)'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='Embedding dimension (must match training)'
    )
    parser.add_argument(
        '--save_output',
        type=str,
        default=None,
        help='Optional: save embeddings to .npy file'
    )
    args = parser.parse_args()

    # Setup
    device    = torch.device('cuda' if torch.cuda.is_available()
                             else 'cpu')
    transform = get_transforms()

    # Load model
    model = load_model(
        model_path    = args.model_path,
        embedding_dim = args.embedding_dim,
        device        = str(device)
    )

    print(f"\nGenerating embeddings for "
          f"{len(args.image_path)} image(s)...\n")

    # Generate embeddings
    if len(args.image_path) == 1:
        # Single image
        emb = get_embedding(
            model, args.image_path[0], transform, device
        )
        print(f"\nEmbedding shape : {emb.shape}")
        print(f"Embedding norm  : {np.linalg.norm(emb):.6f}")
        print(f"Embedding vector (first 8 dims): {emb[:8]}")

    else:
        # Multiple images
        embs = get_embeddings_batch(
            model, args.image_path, transform, device
        )
        print(f"\nEmbeddings matrix shape: {embs.shape}")
        print(f"All norms (should be ~1.0): "
              f"{np.linalg.norm(embs, axis=1)}")

        # Optionally save to disk
        if args.save_output:
            np.save(args.save_output, embs)
            print(f"Embeddings saved to: {args.save_output}")