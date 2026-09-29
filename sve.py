"""Singular Value Ensemble (SVE) for DINO ViTs.

Every attention / MLP linear layer W = U diag(S) Vh is shared across members,
and each member m learns only its own singular values s_m (and bias):
    W_m = U diag(s_m) Vh
"""
import timm
import torch
import torch.nn as nn


class SVELinear(nn.Module):
    """Linear layer with per-member singular values. Input is [B*M, ..., in]."""

    def __init__(self, linear: nn.Linear, n_members: int, init_std: float = 0.01):
        super().__init__()
        self.n_members = n_members
        U, S, Vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("Vh", Vh)
        # multiplicative init: s_m = S * (1 + eps), eps ~ N(0, init_std^2)
        s = S.unsqueeze(0) * (1 + init_std * torch.randn(n_members, S.numel()))
        self.s = nn.Parameter(s.clamp(min=0))
        if linear.bias is not None:
            b = linear.bias.detach().unsqueeze(0).repeat(n_members, 1)
            self.bias = nn.Parameter(b + init_std * torch.randn_like(b))
        else:
            self.bias = None

    def forward(self, x):
        M, shape = self.n_members, x.shape
        W = torch.einsum("ok,mk,ki->moi", self.U, self.s.clamp(min=0), self.Vh)
        x = x.reshape(shape[0] // M, M, -1, shape[-1])
        y = torch.einsum("bmti,moi->bmto", x, W)
        if self.bias is not None:
            y = y + self.bias[None, :, None, :]
        return y.reshape(*shape[:-1], -1)


class EnsembleHead(nn.Module):
    """Independent linear classifier per member. Input [B*M, D] -> [B, M, C]."""

    def __init__(self, dim: int, num_classes: int, n_members: int, init_std: float = 0.01):
        super().__init__()
        self.n_members = n_members
        self.weight = nn.Parameter(init_std * torch.randn(n_members, num_classes, dim))
        self.bias = nn.Parameter(torch.zeros(n_members, num_classes))

    def forward(self, x):
        x = x.view(-1, self.n_members, x.shape[-1])
        return torch.einsum("bmd,mcd->bmc", x, self.weight) + self.bias


class SVE(nn.Module):
    def __init__(self, backbone: str, num_classes: int, n_members: int = 4,
                 init_std: float = 0.01, head_init_std: float = 0.01):
        super().__init__()
        self.n_members = n_members
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for block in self.backbone.blocks:
            block.attn.qkv = SVELinear(block.attn.qkv, n_members, init_std)
            block.attn.proj = SVELinear(block.attn.proj, n_members, init_std)
            block.mlp.fc1 = SVELinear(block.mlp.fc1, n_members, init_std)
            block.mlp.fc2 = SVELinear(block.mlp.fc2, n_members, init_std)
        self.head = EnsembleHead(self.backbone.num_features, num_classes, n_members, head_init_std)

    def forward(self, x):
        """Returns per-member logits [B, M, C]."""
        x = x.repeat_interleave(self.n_members, dim=0)
        return self.head(self.backbone(x))
