from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .pooling import GeneralizedLehmerCausalityStable
from .restricted_graph_bpr import RestrictedGraphBPRRankingLoss


@dataclass
class GeoCANConfig:
    num_classes: int = 6
    num_score_classes: int = 11

    n_seq: int = 10 * 10
    n_layer: int = 2
    d_hidn: int = 256
    d_ff: int = 1024
    n_head: int = 4
    d_head: int = 64
    dropout: float = 0.1
    emb_dropout: float = 0.1

    # CGL
    cgl_ratio: float = 0.5
    cgl_tau: float = 1.0

    # NRA: pooled feature KAN residual
    use_kan_score_enhance: bool = True
    kan_hidden: int = 64
    kan_num_knots: int = 8
    kan_xmin: float = -6.0
    kan_xmax: float = 6.0
    kan_alpha_init: float = 0.0

    # NRA: token-feature residual
    use_token_feature_residual: bool = True
    token_tau: float = 2.5
    token_hidden: int = 64
    token_num_knots: int = 8
    token_xmin: float = -6.0
    token_xmax: float = 6.0
    token_beta_init: float = -6.0

    # Hyperbolic geometry interface.
    # The embedding/projection path is public; the graph-BPR ranking loss is not.
    use_hyperbolic_geo: bool = True
    hyp_dim: int = 64
    hyp_c: float = 1.0
    tbpr_lambda: float = 0.08
    geo_weight: float = 0.1


class ScaledDotProductAttention(nn.Module):
    def __init__(self, cfg: GeoCANConfig):
        super().__init__()
        self.scale = 1.0 / (cfg.d_head ** 0.5)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor) -> Tuple[Tensor, Tensor]:
        scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GeoCANConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_head = cfg.d_head
        self.d_hidn = cfg.d_hidn
        inner = self.n_head * self.d_head

        self.W_Q = nn.Linear(self.d_hidn, inner)
        self.W_K = nn.Linear(self.d_hidn, inner)
        self.W_V = nn.Linear(self.d_hidn, inner)
        self.attn = ScaledDotProductAttention(cfg)
        self.fc = nn.Linear(inner, self.d_hidn)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, L, _ = x.shape
        q = self.W_Q(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        context, attn = self.attn(q, k, v)
        context = context.transpose(1, 2).contiguous().view(B, L, self.n_head * self.d_head)
        return self.dropout(self.fc(context)), attn


class MLP(nn.Module):
    def __init__(self, cfg: GeoCANConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_hidn, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_hidn)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


class ViTBlock(nn.Module):
    def __init__(self, cfg: GeoCANConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_hidn)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_hidn)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.attn(self.norm1(x))
        x = x + h
        return x + self.mlp(self.norm2(x))


class AttnPool1D(nn.Module):
    """Query-token attention pooling over patch tokens."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_token: Tensor, tokens: Tensor, return_attn: bool = False):
        q = self.q_proj(q_token).unsqueeze(1)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        attn = torch.matmul(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        attn = self.dropout(F.softmax(attn, dim=-1))
        pooled = torch.matmul(attn, v).squeeze(1)

        if return_attn:
            return pooled, attn.squeeze(1)
        return pooled


class PiecewiseLinearKAN(nn.Module):
    """Learnable 1D piecewise-linear spline on a fixed grid."""

    def __init__(self, num_knots: int = 8, x_min: float = -6.0, x_max: float = 6.0):
        super().__init__()
        if num_knots < 2:
            raise ValueError("num_knots must be >= 2")
        self.num_knots = num_knots
        self.x_min = x_min
        self.x_max = x_max
        knots_x = torch.linspace(x_min, x_max, num_knots)
        self.register_buffer("knots_x", knots_x)
        self.knots_y = nn.Parameter(knots_x.clone())

    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, self.x_min, self.x_max)
        t = (x - self.x_min) / (self.x_max - self.x_min) * (self.num_knots - 1)
        idx0 = torch.floor(t).long().clamp(0, self.num_knots - 1)
        idx1 = (idx0 + 1).clamp(0, self.num_knots - 1)
        y0 = self.knots_y[idx0]
        y1 = self.knots_y[idx1]
        w = t - idx0.float()
        return (1.0 - w) * y0 + w * y1


class KANScalarMLP(nn.Module):
    """Scalar nonlinear functional head used in NRA for quality logits."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_knots: int = 8,
        x_min: float = -6.0,
        x_max: float = 6.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.up = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.splines = nn.ModuleList([
            PiecewiseLinearKAN(num_knots=num_knots, x_min=x_min, x_max=x_max)
            for _ in range(hidden_dim)
        ])
        self.down = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        u = self.drop(self.up(self.ln(x)))
        f_u = torch.stack([sp(u[:, i]) for i, sp in enumerate(self.splines)], dim=-1)
        return self.down(f_u)


class KANTokenFeatureHead(nn.Module):
    """Token-wise nonlinear functional decomposition used in NRA."""

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int,
        num_knots: int = 8,
        x_min: float = -6.0,
        x_max: float = 6.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(token_dim)
        self.up = nn.Linear(token_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.splines = nn.ModuleList([
            PiecewiseLinearKAN(num_knots=num_knots, x_min=x_min, x_max=x_max)
            for _ in range(hidden_dim)
        ])
        self.down = nn.Linear(hidden_dim, token_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        u = self.drop(self.up(self.ln(patch_tokens)))
        f_u = torch.stack([sp(u[..., i]) for i, sp in enumerate(self.splines)], dim=-1)
        return self.down(f_u)


class HyperbolicProjector(nn.Module):
    """Exponential map at the origin of the Poincare ball."""

    def __init__(self, in_dim: int, out_dim: int, c: float = 1.0):
        super().__init__()
        self.c = float(c)
        self.lin = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))

    def expmap0(self, v: Tensor, eps: float = 1e-6) -> Tensor:
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        x = factor * v
        max_norm = (1.0 - 1e-5) / sqrt_c
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        return x * torch.clamp(max_norm / x_norm, max=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.expmap0(self.lin(x))


class GeoCAN(nn.Module):
    """GeoCAN public implementation.

    Publicly included:
        - generalized Lehmer causal map C
        - asymmetric causal modulation X_cgl
        - Transformer contextual encoding
        - nonlinear relational aggregation
        - hyperbolic node embedding and positive/negative relation selection

    Omitted:
        - restricted graph-theoretic BPR spectral ranking loss L_geo
    """

    def __init__(self, cfg: Optional[GeoCANConfig] = None):
        super().__init__()
        self.cfg = cfg or GeoCANConfig()

        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Conv2d(128, self.cfg.d_hidn, 3, stride=3, padding=1),
        )

        self.cls_token_class = nn.Parameter(torch.randn(1, 1, self.cfg.d_hidn))
        self.cls_token_score = nn.Parameter(torch.randn(1, 1, self.cfg.d_hidn))
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.n_seq + 2, self.cfg.d_hidn))
        self.emb_dropout = nn.Dropout(self.cfg.emb_dropout)

        self.causality = GeneralizedLehmerCausalityStable(
            alpha_init=2.3,
            beta_init=1.3,
            learnable=True,
        )

        self.blocks = nn.ModuleList([ViTBlock(self.cfg) for _ in range(self.cfg.n_layer)])
        self.norm = nn.LayerNorm(self.cfg.d_hidn)

        self.class_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_hidn),
            nn.Linear(self.cfg.d_hidn, self.cfg.num_classes),
        )

        self.score_pool = AttnPool1D(self.cfg.d_hidn, dropout=self.cfg.dropout)
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.cfg.d_hidn),
            nn.Linear(self.cfg.d_hidn, self.cfg.d_hidn),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_hidn, self.cfg.num_score_classes),
        )

        self.score_kan = KANScalarMLP(
            in_dim=self.cfg.d_hidn,
            hidden_dim=self.cfg.kan_hidden,
            out_dim=self.cfg.num_score_classes,
            num_knots=self.cfg.kan_num_knots,
            x_min=self.cfg.kan_xmin,
            x_max=self.cfg.kan_xmax,
        )
        self.kan_alpha = nn.Parameter(torch.tensor(self.cfg.kan_alpha_init))

        self.token_feat_head = KANTokenFeatureHead(
            token_dim=self.cfg.d_hidn,
            hidden_dim=self.cfg.token_hidden,
            num_knots=self.cfg.token_num_knots,
            x_min=self.cfg.token_xmin,
            x_max=self.cfg.token_xmax,
        )
        self.token_beta = nn.Parameter(torch.tensor(self.cfg.token_beta_init))
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

        self.id_to_hyp = nn.Linear(self.cfg.d_hidn, self.cfg.hyp_dim, bias=False)
        self.node_embed = nn.Sequential(
            nn.LayerNorm(self.cfg.d_hidn),
            nn.Linear(self.cfg.d_hidn, self.cfg.hyp_dim),
        )
        self.hyp_proj = HyperbolicProjector(
            in_dim=self.cfg.hyp_dim,
            out_dim=self.cfg.hyp_dim,
            c=self.cfg.hyp_c,
        )
        self.restricted_geo_loss = RestrictedGraphBPRRankingLoss()

    def causality_factor_soft_degree(self, C: Tensor, tau: float = 1.0, eps: float = 1e-6) -> Tensor:
        """Compute asymmetric causal effect g from C - C^T."""
        S = torch.sigmoid((C - C.transpose(1, 2)) / tau)
        out_degree = S.sum(dim=2)
        in_degree = S.sum(dim=1)
        g = out_degree - in_degree
        g = g - g.mean(dim=1, keepdim=True)
        g = g / (g.std(dim=1, keepdim=True) + eps)
        return g.unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def _pick_pos_neg_indices(self, causal_map: Tensor) -> Tuple[Tensor, Tensor]:
        B, K, _ = causal_map.shape
        diag = torch.arange(K, device=causal_map.device)

        cm_pos = causal_map.clone()
        cm_pos[:, diag, diag] = -1e9
        pos_idx = torch.argmax(cm_pos, dim=-1)

        cm_neg = causal_map.clone()
        cm_neg[:, diag, diag] = 1e9
        neg_idx = torch.argmin(cm_neg, dim=-1)
        return pos_idx, neg_idx

    def build_hyperbolic_triplets(self, x_cgl: Tensor, causal_map: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Build anchor/positive/negative hyperbolic node triplets.

        This public function preserves the geometry pipeline before the restricted
        spectral graph-BPR ranking loss.
        """
        B, K, _, _ = x_cgl.shape
        device = x_cgl.device

        node_feat = x_cgl.mean(dim=(2, 3))
        eye = torch.eye(K, device=device).unsqueeze(0).expand(B, -1, -1)
        node_id_part = self.id_to_hyp(eye)
        node_val_part = self.node_embed(node_feat).unsqueeze(1)
        node_h = self.hyp_proj(node_id_part + node_val_part)

        pos_idx, neg_idx = self._pick_pos_neg_indices(causal_map)
        pos = torch.gather(node_h, dim=1, index=pos_idx.unsqueeze(-1).expand(-1, -1, node_h.size(-1)))
        neg = torch.gather(node_h, dim=1, index=neg_idx.unsqueeze(-1).expand(-1, -1, node_h.size(-1)))
        return node_h.reshape(B * K, -1), pos.reshape(B * K, -1), neg.reshape(B * K, -1)

    def compute_restricted_geo_loss(self, x_cgl: Tensor, causal_map: Tensor) -> Tensor:
        anc, pos, neg = self.build_hyperbolic_triplets(x_cgl, causal_map)
        return self.restricted_geo_loss(anc, pos, neg, lambda_value=self.cfg.tbpr_lambda) / anc.size(0)

    def forward(
        self,
        img: Tensor,
        return_token_maps: bool = False,
        return_geo_loss: bool = False,
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]],
        Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
        Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor], Optional[Tensor]],
    ]:
        B = img.size(0)

        # Feature factors F.
        x = self.conv_embedding(img)

        # CGL: directed causal interaction matrix C.
        causal_map = self.causality(x)

        # CGL: asymmetric causal effect g and causal-enhanced representation X_cgl.
        g = self.causality_factor_soft_degree(causal_map, tau=self.cfg.cgl_tau)
        gate = 1.0 + self.cfg.cgl_ratio * torch.tanh(g)
        gate = gate.clamp(0.5, 1.5)
        x_cgl = x * gate

        loss_geo = None
        if self.cfg.use_hyperbolic_geo and return_geo_loss:
            loss_geo = self.compute_restricted_geo_loss(x_cgl, causal_map)

        x_tok = x_cgl.flatten(2).transpose(1, 2)
        cls_class = self.cls_token_class.expand(B, -1, -1).contiguous()
        cls_score = self.cls_token_score.expand(B, -1, -1).contiguous()
        x_tok = torch.cat([cls_class, cls_score, x_tok], dim=1)
        x_tok = self.emb_dropout(x_tok + self.pos_embed[:, :x_tok.size(1)])

        for block in self.blocks:
            x_tok = block(x_tok)
        x_tok = self.norm(x_tok)

        class_token = x_tok[:, 0]
        score_token = x_tok[:, 1]
        patch_tokens = x_tok[:, 2:]

        class_logits = self.class_head(class_token)
        class_pred = class_logits.argmax(dim=1)

        score_feat, psi = self.score_pool(score_token, patch_tokens, return_attn=True)

        token_maps = {}
        if self.cfg.use_token_feature_residual:
            psi_t = F.softmax(torch.log(psi + 1e-12) / self.cfg.token_tau, dim=-1)
            delta_tokens = self.token_feat_head(patch_tokens)
            delta_feat = (psi_t.unsqueeze(-1) * delta_tokens).sum(dim=1)
            beta = torch.sigmoid(self.token_beta)
            score_feat = score_feat + beta * self.delta_scale * delta_feat

            token_maps = {
                "psi": psi_t.detach(),
                "delta_tokens": delta_tokens.detach(),
                "delta_feat": delta_feat.detach(),
                "beta": beta.detach(),
            }

        score_logits_base = self.score_head(score_feat)
        if self.cfg.use_kan_score_enhance:
            score_logits_kan = self.score_kan(score_feat)
            score_logits = score_logits_kan
        else:
            score_logits = score_logits_base

        if return_token_maps and return_geo_loss:
            return class_logits, class_pred, score_logits, token_maps, loss_geo
        if return_token_maps:
            return class_logits, class_pred, score_logits, token_maps
        if return_geo_loss:
            return class_logits, class_pred, score_logits, loss_geo
        return class_logits, class_pred, score_logits


if __name__ == "__main__":
    model = GeoCAN()
    x = torch.randn(2, 3, 224, 224)
    class_logits, class_pred, score_logits, token_maps = model(x, return_token_maps=True)
    print("class_logits:", class_logits.shape)
    print("score_logits:", score_logits.shape)
    print("psi:", token_maps["psi"].shape)
