
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

class EmbeddingNet(nn.Module):
    """
    Embedding network using ResNet-50 as backbone.
    Replaces the classification head with a projection layer
    followed by L2 normalization to produce unit-norm embeddings.
    """

    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()

        # Load pretrained ResNet-50 on ImageNet
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
            )

        # Remove the classification head - replace with identity(pass-through)
        # backbone now outputs 2048-dim feature vector from Global Avg Pool
        self.backbone.fc = nn.Identity()

        # Projection head: maps 2048-dim features → embedding_dim(128)
        self.projector = nn.Linear(2048, 128)

    def forward(self, x):
        # Pass input through backbone → get 2048-dim features(B, 2048)
        features = self.backbone(x)

        # Pass features through projector → get embedding_dim vector(B, embedding_dim)
        embedding = self.projector(features)

        # Step F: L2 normalize along dim=1, so every vector has length 1
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding