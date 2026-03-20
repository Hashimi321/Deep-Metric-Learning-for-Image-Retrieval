# Training script for all three metric learning experiments

import os
import csv
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import EmbeddingNet
from dataset import ContrastiveDataset, TripletDataset, get_transforms
from loss import ContrastiveLoss, TripletLoss, batch_hard_mining


# ─────────────────────────────────────────────
# HYPERPARAMETERS — identical across experiments
# ─────────────────────────────────────────────
EMBEDDING_DIM = 128
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4
NUM_WORKERS   = 2


def compute_recall_at_1(model, dataset, device, max_samples=500):
    """
    Computes Recall@1 on a dataset subset.

    For each query image, finds nearest neighbor in embedding space
    (excluding itself) and checks if they share the same class.

    Args:
        model       : trained EmbeddingNet
        dataset     : plain ImageFolder dataset
        device      : torch device
        max_samples : number of images to evaluate

    Returns:
        float: Recall@1 between 0.0 and 1.0
    """
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            embs   = model(images)           # forward pass
            all_embeddings.append(embs.cpu())
            all_labels.append(labels)

            if len(all_labels) * 64 >= max_samples:
                break

    all_embeddings = torch.cat(all_embeddings, dim=0)[:max_samples]
    all_labels     = torch.cat(all_labels,     dim=0)[:max_samples]

    # Pairwise distance matrix
    dist_matrix = torch.cdist(all_embeddings, all_embeddings, p=2)

    # Exclude self by setting diagonal to inf
    dist_matrix.fill_diagonal_(float('inf'))

    # Nearest neighbor index for each query
    nearest_indices = torch.argmin(dist_matrix, dim=1)

    # Check if nearest neighbor shares the same label
    nearest_labels = all_labels[nearest_indices]
    correct        = (all_labels == nearest_labels)

    return correct.float().mean().item()


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Saves model state, optimizer state, epoch, and loss to a .pt file.
    """
    torch.save({
        'epoch'                : epoch,
        'model_state_dict'     : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss'                 : loss,
    }, filepath)
    print(f"  Checkpoint saved → {filepath}")


def log_to_csv(filepath, epoch, loss, recall):
    """
    Appends one training row to CSV log in append mode.
    """
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss, recall])


def train_one_experiment(exp_name, loss_type, data_path, save_dir):
    """
    Full training loop for one experiment.

    Args:
        exp_name  : e.g. 'exp1_contrastive'
        loss_type : 'contrastive' | 'triplet_random' | 'triplet_hard'
        data_path : root path containing train/ and val/ folders
        save_dir  : where to save checkpoints and CSV log
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*50}")
    print(f"Experiment : {exp_name}")
    print(f"Loss type  : {loss_type}")
    print(f"Device     : {device}")
    print(f"{'='*50}")

    transform = get_transforms()

    # ── Dataset ──────────────────────────────────────
    if loss_type == 'contrastive':
        train_dataset = ContrastiveDataset(
            root=os.path.join(data_path, 'train'),
            transform=transform
        )
    else:
        # triplet_random and triplet_hard both use TripletDataset
        # TripletDataset now returns (anchor, positive, negative, label)
        train_dataset = TripletDataset(
            root=os.path.join(data_path, 'train'),
            transform=transform
        )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, 'val'),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # ── Model ────────────────────────────────────────
    model     = EmbeddingNet(EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Loss function ────────────────────────────────
    if loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=1.0)
    else:
        criterion = TripletLoss(margin=0.2)

    # ── CSV log ──────────────────────────────────────
    log_path = os.path.join(save_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'loss', 'recall@1'])

    # ── Training loop ────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss  = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # ── Experiment 1: Contrastive ─────────────
            if loss_type == 'contrastive':
                img1, img2, labels = batch
                img1   = img1.to(device)
                img2   = img2.to(device)
                labels = labels.to(device)

                emb1 = model(img1)   # (B, 128)
                emb2 = model(img2)   # (B, 128)

                # FIX: pass embeddings not raw images
                loss = criterion(emb1, emb2, labels)

            # ── Experiment 2: Triplet Random ──────────
            elif loss_type == 'triplet_random':
                # TripletDataset returns 4 items now
                anchor, positive, negative, _ = batch
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                emb_anchor   = model(anchor)    # (B, 128)
                emb_positive = model(positive)  # (B, 128)
                emb_negative = model(negative)  # (B, 128)

                loss = criterion(emb_anchor, emb_positive, emb_negative)

            # ── Experiment 3: Hard Negative Mining ────
            elif loss_type == 'triplet_hard':
                # We only need anchor + its label for hard mining
                # batch_hard_mining finds hard pairs within the batch
                anchor, _, _, anchor_labels = batch
                anchor        = anchor.to(device)
                anchor_labels = anchor_labels.to(device)

                emb_anchor = model(anchor)   # (B, 128)

                # Hard mining computes its own loss internally
                loss = batch_hard_mining(
                    emb_anchor, anchor_labels, margin=0.2
                )

            # ── Backprop ──────────────────────────────
            loss.backward()    # compute gradients
            optimizer.step()   # update weights

            epoch_loss  += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        recall   = compute_recall_at_1(model, val_dataset, device)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss : {avg_loss:.4f}")
        print(f"  Recall@1 : {recall:.4f}")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(
            save_dir, f'{exp_name}_epoch{epoch}.pt'
        )
        save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

        # Append to CSV log
        log_to_csv(log_path, epoch, avg_loss, recall)

        model.train()  # back to train mode after eval

    print(f"\nTraining complete for {exp_name}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train metric learning models on Caltech-101'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset root (must contain train/ and val/ folders)'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='all',
        choices=['all', 'exp1', 'exp2', 'exp3'],
        help='Which experiment to run'
    )
    args = parser.parse_args()

    if args.experiment in ('all', 'exp1'):
        train_one_experiment(
            exp_name  = 'exp1_contrastive',
            loss_type = 'contrastive',
            data_path = args.data_path,
            save_dir  = 'weights/exp1'
        )

    if args.experiment in ('all', 'exp2'):
        train_one_experiment(
            exp_name  = 'exp2_triplet_random',
            loss_type = 'triplet_random',
            data_path = args.data_path,
            save_dir  = 'weights/exp2'
        )

    if args.experiment in ('all', 'exp3'):
        train_one_experiment(
            exp_name  = 'exp3_triplet_hard',
            loss_type = 'triplet_hard',
            data_path = args.data_path,
            save_dir  = 'weights/exp3'
        )