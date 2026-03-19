# Implements ContrastiveLoss, TripletLoss, and batch_hard_mining

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pair-based metric learning.

    Formula:
        L = y * D^2 + (1 - y) * max(0, margin - D)^2

    Where:
        D     = Euclidean distance between two embeddings
        y     = 1 for same-class pairs, 0 for different-class pairs
        margin = minimum distance enforced for negative pairs (default 1.0)
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        # store margin as instance variable
        self.margin = margin

    def forward(self, emb1, emb2, target):
        """
        Args:
            emb1   (Tensor): embeddings of first images,  shape (B, D)
            emb2   (Tensor): embeddings of second images, shape (B, D)
            target (Tensor): labels, 1=same class, 0=diff class, shape (B,)

        Returns:
            Tensor: scalar mean loss over the batch
        """
        # Compute Euclidean distance between each pair
        # torch.nn.functional.pairwise_distance computes ||emb1 - emb2||_2
        # for each sample in the batch
        D = F.pairwise_distance(emb1, emb2, p=2)

        # Positive loss term → y * D^2
        positive_loss = target * D.pow(2)

        #  Negative loss term → (1 - y) * max(0, margin - D)^2
        negative_loss = (1 - target) * F.relu(
            self.margin - D
        ).pow(2)

        # Total loss = mean over batch
        loss = torch.mean(positive_loss + negative_loss)

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for triplet-based metric learning.

    Formula:
        L = max(0, D(anchor, positive) - D(anchor, negative) + margin)

    Enforces that anchor is closer to positive than to negative
    by at least margin distance.
    """

    def __init__(self, margin=0.2):        
        # Margin is critical(0.2 – 1.0) (Too small → weak separation) (Too large → hard to satisfy)
        super(TripletLoss, self).__init__()
        # store margin
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor   (Tensor): anchor embeddings,   shape (B, D)
            positive (Tensor): positive embeddings, shape (B, D)
            negative (Tensor): negative embeddings, shape (B, D)

        Returns:
            Tensor: scalar mean loss over the batch
        """
        # Distance from anchor to positive
        dist_pos = F.pairwise_distance(anchor, positive, p=2)

        # Distance from anchor to negative
        dist_neg = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss = max(0, dist_pos - dist_neg + margin)
        # F.relu gives max(0, x)
        loss = F.relu(dist_pos - dist_neg + self.margin)

        # Return mean over batch
        return torch.mean(loss)


def batch_hard_mining(embeddings, labels, margin=0.2):
    """
    Batch Hard Negative Mining.

    For each anchor in the batch:
        - Finds the hardest positive: same class, MAXIMUM distance
        - Finds the hardest negative: different class, MINIMUM distance
    Computes triplet loss on these hard triplets.

    Args:
        embeddings (Tensor): shape (B, D) — all embeddings in batch
        labels     (Tensor): shape (B,)  — class label for each embedding
        margin     (float):  triplet loss margin

    Returns:
        Tensor: scalar mean loss over hard triplets
    """

    B = embeddings.size(0)  # batch size

    # Compute full pairwise distance matrix (B × B)
    # dist_matrix[i][j] = distance between embedding i and embedding j
    dist_matrix = torch.cdist(embeddings,embeddings, p=2)

    # Build masks for valid positives and negatives
    # We need to know which pairs are same-class and which are different

    # labels shape: (B,) → expand to (B, B) for comparison
    labels_row = labels.unsqueeze(1)  # shape (B, 1)
    labels_col = labels.unsqueeze(0)  # shape (1, B)

    # positive_mask[i][j] = True if i and j are same class AND i != j
    same_class = (labels_row == labels_col)          # (B, B) boolean
    eye = torch.eye(B, dtype=torch.bool,
                    device=embeddings.device)        # identity matrix
    positive_mask = same_class & ~eye                # exclude diagonal

    # negative_mask[i][j] = True if i and j are different class
    negative_mask = ~same_class                      # (B, B) boolean

    # For each anchor, find hardest positive
    # = same class with MAXIMUM distance
    # Set non-positive positions to -inf so they don't win the max
    dist_pos = dist_matrix.clone()
    dist_pos[~positive_mask] = -float('inf')
    hardest_positive_dist, _ = dist_pos.max(dim=1)  # (B,)

    # For each anchor, find hardest negative
    # = different class with MINIMUM distance
    # Set non-negative positions to +inf so they don't win the min
    dist_neg = dist_matrix.clone()
    dist_neg[~negative_mask] = float('inf')
    hardest_negative_dist, _ = dist_neg.min(dim=1)  # (B,)

    # Compute triplet loss on hard pairs
    losses = F.relu(hardest_positive_dist - hardest_negative_dist + margin)

    #  Only average over valid anchors
    # (anchors that had both a valid positive and negative in the batch)
    valid = (hardest_positive_dist != -float('inf')) & \
            (hardest_negative_dist != float('inf'))

    if valid.sum() == 0:
        return torch.tensor(0.0,
                            requires_grad=True,
                            device=embeddings.device)

    # return mean of losses where valid is True
    return losses[valid].mean()