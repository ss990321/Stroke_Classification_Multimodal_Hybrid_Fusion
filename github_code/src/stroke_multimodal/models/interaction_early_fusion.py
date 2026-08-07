import torch
import torch.nn as nn
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
except ImportError:
    from torchvision.models import efficientnet_b0

    EfficientNet_B0_Weights = None




class EfficientNetB0SignalEncoder(nn.Module):
    def __init__(self, emb_dim: int = 512, pretrained: bool = True, dropout: float = 0.2):
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
            bias=False
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
            nn.Linear(in_features, emb_dim),
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected x to have 3 or 4 dims, but got shape={x.shape}")

        feat = self.backbone(x)
        emb = self.proj(feat)
        return emb

class TabMLP(nn.Module):
    def __init__(self, in_dim: int, hidden1=128, hidden2=128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, f):
        return self.net(f)


class SelfAttentionBlock1D(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, ff_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class RawInteractionBranch(nn.Module):
    def __init__(
        self,
        in_ch_signal: int,
        tab_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        feat_ch: int = 16,
        downsample_stride: int = 25,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feat_proj = nn.Sequential(
            nn.Linear(tab_dim, feat_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        concat_ch = in_ch_signal + feat_ch
        self.stem = nn.Sequential(
            nn.Conv1d(concat_ch, d_model, kernel_size=7, stride=downsample_stride, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_blocks = nn.ModuleList([
            SelfAttentionBlock1D(d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, f):
        _, _, t = x.shape
        f_ch = self.feat_proj(f).unsqueeze(-1).expand(-1, -1, t)
        u = torch.cat([x, f_ch], dim=1)
        h = self.stem(u).transpose(1, 2)
        for blk in self.attn_blocks:
            h = blk(h)
        h = self.norm(h)
        return h.mean(dim=1)


class SimpleConcatFusion(nn.Module):
    def __init__(self, d_raw: int, d_sig: int, d_feat: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.in_dim = d_raw + d_sig + d_feat
        self.out_dim = hidden_dim
        self.fuse = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, z_raw, z_sig, z_feat):
        z = torch.cat([z_raw, z_sig, z_feat], dim=1)
        return self.fuse(z)


class SimpleMeanFusion(nn.Module):
    def __init__(self, d_raw: int, d_sig: int, d_feat: int, d_common: int = 128, dropout: float = 0.2):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.Linear(d_raw, d_common), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.sig_proj = nn.Sequential(nn.Linear(d_sig, d_common), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.feat_proj = nn.Sequential(nn.Linear(d_feat, d_common), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out_dim = d_common

    def forward(self, z_raw, z_sig, z_feat):
        h_raw = self.raw_proj(z_raw)
        h_sig = self.sig_proj(z_sig)
        h_feat = self.feat_proj(z_feat)
        return (h_raw + h_sig + h_feat) / 3.0


class InteractionEarlyFusion(nn.Module):
    def __init__(
        self,
        in_ch_signal=12,
        tab_dim=None,
        signal_emb_dim=512,
        tab_emb_dim=128,
        raw_emb_dim=128,
        fusion_dim=128,
        dropout=0.2,
        return_dict=False,
        raw_nhead=4,
        raw_num_layers=2,
        raw_ff_dim=256,
        raw_feat_ch=16,
        raw_downsample_stride=25,
        fusion_type="concat",
        pretrained=True
    ):
        super().__init__()
        if tab_dim is None:
            raise ValueError("tab_dim is required.")
        if fusion_type not in {"concat", "mean"}:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        self.return_dict = return_dict
        self.fusion_type = fusion_type

        self.signal_enc = EfficientNetB0SignalEncoder(
            emb_dim=signal_emb_dim,
            pretrained=pretrained,
            dropout=dropout,
        )
        self.tab_enc = TabMLP(in_dim=tab_dim, hidden1=128, hidden2=tab_emb_dim, dropout=dropout)
        self.raw_branch = RawInteractionBranch(
            in_ch_signal=in_ch_signal,
            tab_dim=tab_dim,
            d_model=raw_emb_dim,
            nhead=raw_nhead,
            num_layers=raw_num_layers,
            ff_dim=raw_ff_dim,
            feat_ch=raw_feat_ch,
            downsample_stride=raw_downsample_stride,
            dropout=dropout,
        )

        if fusion_type == "concat":
            self.fusion = SimpleConcatFusion(d_raw=raw_emb_dim, d_sig=signal_emb_dim, d_feat=tab_emb_dim, hidden_dim=fusion_dim, dropout=dropout)
        else:
            self.fusion = SimpleMeanFusion(d_raw=raw_emb_dim, d_sig=signal_emb_dim, d_feat=tab_emb_dim, d_common=fusion_dim, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(self.fusion.out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, f):
        z_sig = self.signal_enc(x)
        z_feat = self.tab_enc(f)
        z_raw = self.raw_branch(x, f)
        fused = self.fusion(z_raw, z_sig, z_feat)
        logit = self.head(fused).squeeze(-1)
        if self.return_dict:
            return {
                "logit": logit,
                "z_raw": z_raw,
                "z_sig": z_sig,
                "z_feat": z_feat,
                "fused": fused,
                "fusion_type": self.fusion_type,
            }
        return logit
