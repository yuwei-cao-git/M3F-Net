import torch.nn as nn
import torch.nn.functional as F
from pointnext import pointnext_s, PointNext, pointnext_b, pointnext_l, pointnext_xl


class PointNextModel(nn.Module):
    def __init__(self, config, in_dim):
        super(PointNextModel, self).__init__()
        self.config = config
        self.n_classes = config["n_classes"]

        # Initialize the PointNext encoder and decoder
        if config["encoder"] == "s":
            self.encoder = pointnext_s(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        elif config["encoder"] == "b":
            self.encoder = pointnext_b(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        elif config["encoder"] == "l":
            self.encoder = pointnext_l(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder
        else:
            self.encoder = pointnext_xl(
                in_dim=in_dim
            )  # Load the pointnext_s() as the encoder

        self.backbone = PointNext(self.config["emb_dims"], encoder=self.encoder)

        self.norm = nn.BatchNorm1d(self.config["emb_dims"])
        self.act = nn.ReLU()
        self.cls_head = nn.Sequential(
            nn.Linear(self.config["emb_dims"], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(256, self.n_classes),
        )

    def forward(self, point_cloud, xyz):
        out = self.norm(self.backbone(point_cloud, xyz))
        out = out.mean(dim=-1)
        out = self.act(out)
        logits = self.cls_head(out)
        return logits
