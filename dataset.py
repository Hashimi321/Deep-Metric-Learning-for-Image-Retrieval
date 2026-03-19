import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Standard ImageNet preprocessing — must match what ResNet-50 was trained on
def get_transforms():
    """Returns standard ImageNet preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),  # fix grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def build_class_to_indices(dataset):
    """
    Builds a dictionary mapping each class label to
    a list of all image indices belonging to that class.

    Args:
        dataset: ImageFolder dataset with .targets attribute

    Returns:
        dict: {class_label: [idx1, idx2, ...]}
    """
    class_to_indices = defaultdict(list)

    # for each (idx, label), append idx to class_to_indices[label]
    for idx, label in enumerate(dataset.targets):
        class_to_indices[label].append(idx)

    return class_to_indices


class ContrastiveDataset(Dataset):
    """
    Returns labeled pairs (img1, img2, label) where:
        label = 1 → same class (positive pair)
        label = 0 → different class (negative pair)
    Sampled with equal probability (p=0.5).
    """

    def __init__(self, root, transform=None):
        """
        Args:
            root (str): Path to dataset root folder
            transform: Image transforms to apply
        """
        # Load base dataset using ImageFolder
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform
        )

        # Build class → indices lookup
        self.class_to_indices = build_class_to_indices(self.dataset)

        # Get all unique class labels
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        # return total number of images in base dataset
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns a labeled pair for contrastive learning.
        """
        # Get anchor image and its class label
        img1, label1 = self.dataset[index]

        # Flip a coin: 50% positive pair, 50% negative pair
        if random.random() < 0.5:
            # --- POSITIVE PAIR (same class) ---
            # Get all indices for this class
            same_class_indices = self.class_to_indices[label1]

            # Pick a random index from same class
            # but make sure it's NOT the same image (avoid trivial pairs)
            # use random.choice() on same_class_indices
            # If only one image in class, use it anyway
            if len(same_class_indices) > 1:
                positive_idx = index  # start with current to enter loop
                while positive_idx == index:
                    positive_idx = random.choice(same_class_indices)
            else:
                positive_idx = random.choice(same_class_indices)

            img2, _ = self.dataset[positive_idx]
            label = 1  # 1 for positive pair

        else:
            # --- NEGATIVE PAIR (different class) ---
            # Pick a different class
            # use random.choice() on self.classes
            # but make sure it's NOT label1
            different_class = label1  # start same to enter loop
            while different_class == label1:
                different_class = random.choice(self.classes)

            # Pick a random image from that different class
            negative_idx = random.choice(
                self.class_to_indices[different_class]
            )

            img2, _ = self.dataset[negative_idx]
            label = 0  # 0 for negative pair

        return img1, img2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    """
    Returns triplets (anchor, positive, negative) where:
        anchor   → reference image
        positive → different image, same class as anchor
        negative → image from a different class
    """

    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transform
        )
        self.class_to_indices = build_class_to_indices(self.dataset)
        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        # return total number of images
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns an (anchor, positive, negative) triplet.
        """
        # Anchor
        anchor_img, anchor_label = self.dataset[index]

        # --- POSITIVE: same class, different image ---
        same_class_indices = self.class_to_indices[anchor_label]

        if len(same_class_indices) > 1:
            positive_idx = index
            while positive_idx == index:
                positive_idx = random.choice(same_class_indices)
        else:
            positive_idx = random.choice(same_class_indices)

        positive_img, _ = self.dataset[positive_idx]

        # --- NEGATIVE: different class ---
        different_class = anchor_label
        while different_class == anchor_label:
            different_class = random.choice(self.classes)

        negative_idx = random.choice(
            self.class_to_indices[different_class]
        )
        negative_img, _ = self.dataset[negative_idx]

        return anchor_img, positive_img, negative_img