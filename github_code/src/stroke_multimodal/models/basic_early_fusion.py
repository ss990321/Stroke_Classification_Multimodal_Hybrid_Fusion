import torch
import torch.nn as nn

try:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
except ImportError:
    from torchvision.models import efficientnet_b0

    EfficientNet_B0_Weights = None


class EfficientNetB0SignalEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if EfficientNet_B0_Weights is None:
            backbone = efficientnet_b0(pretrained=pretrained)
        else:
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)

        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            if pretrained:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.features[0][0] = new_conv
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()

        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, embedding_dim),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected ECG tensor with 3 or 4 dims, got {x.shape}")
        return self.proj(self.backbone(x))


class BasicEarlyFusion(nn.Module):
    """Two-branch early fusion: ECG embedding + tabular feature embedding."""

    def __init__(
        self,
        feature_dim: int,
        ecg_embedding_dim: int = 512,
        feature_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        pretrained: bool = True,
        return_dict: bool = False,
    ):
        super().__init__()
        self.return_dict = return_dict
        self.signal_encoder = EfficientNetB0SignalEncoder(
            embedding_dim=ecg_embedding_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, feature_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        fusion_input_dim = ecg_embedding_dim + feature_embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, ecg, feature):
        signal_embedding = self.signal_encoder(ecg)
        feature_embedding = self.feature_encoder(feature)
        fused = torch.cat([signal_embedding, feature_embedding], dim=1)
        logit = self.classifier(fused).squeeze(-1)
        if self.return_dict:
            return {
                "logit": logit,
                "signal_embedding": signal_embedding,
                "feature_embedding": feature_embedding,
                "fused": fused,
            }
        return logit
