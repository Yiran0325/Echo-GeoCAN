from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair


class GeneralizedLehmerPool2d(nn.Module):
    """Generalized Lehmer mean pooling.

    This corresponds to the GLM_{alpha,beta}(.) operator used in the paper.
    """

    def __init__(self, alpha: float, beta: float, kernel_size, stride, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        self.dilation = _pair(dilation)
        self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float64), requires_grad=True)
        self.beta = Parameter(torch.tensor(beta, dtype=torch.float64), requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        kw, kh = self.kernel_size
        a = F.avg_pool2d(
            self.alpha.pow((self.beta + 1) * input),
            self.kernel_size,
            self.stride,
            self.padding,
        )
        b = F.avg_pool2d(
            self.alpha.pow(self.beta * input),
            self.kernel_size,
            self.stride,
            self.padding,
        )
        pa = (torch.sign(a) * F.relu(torch.abs(a))).mul(kw * kh)
        pb = (torch.sign(b) * F.relu(torch.abs(b))).mul(kw * kh)
        return torch.log(pa / pb) / torch.log(self.alpha)


class GeneralizedLehmerCausalityStable(nn.Module):
    """Stable generalized-Lehmer causal interaction estimator.

    Input:
        feature_maps: [B, K, H, W]

    Output:
        causal_map: [B, K, K], where causal_map[:, i, j] estimates
        P(F_i | F_j) through a generalized Lehmer-mean ratio.

    This implements the public CGL causal-map construction:
        C_ij = GLM_{alpha,beta}(F_i x F_j) / (GLM_{alpha,beta}(F_j) + eps)
    with a stable exponential parameterization of alpha.
    """

    def __init__(self, alpha_init: float = 2.3, beta_init: float = 1.3,
                 learnable: bool = True, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        if learnable:
            self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
            self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        else:
            self.register_buffer("alpha_raw", torch.tensor(float(alpha_init)))
            self.register_buffer("beta", torch.tensor(float(beta_init)))

    def _alpha(self) -> Tensor:
        return 1.0 + F.softplus(self.alpha_raw)

    def forward(self, feature_maps: Tensor) -> Tensor:
        if feature_maps.dim() != 4:
            raise ValueError(f"Expected [B,K,H,W], got {tuple(feature_maps.shape)}")

        B, K, _, _ = feature_maps.shape

        # Keep the activation range stable for the exponential GLM.
        feature_maps = torch.sigmoid(feature_maps)
        Fv = feature_maps.view(B, K, -1)

        alpha = self._alpha()
        beta = self.beta
        log_alpha = torch.log(alpha + self.eps)

        # GLM(F_j)
        numerator_j = torch.sum(
            torch.exp(log_alpha * ((beta + 1.0) * Fv)),
            dim=-1,
        ) + self.eps
        denominator_j = torch.sum(
            torch.exp(log_alpha * (beta * Fv)),
            dim=-1,
        ) + self.eps
        glm_j = torch.log(numerator_j / denominator_j) / (log_alpha + self.eps)

        # Separable approximation for GLM(F_i x F_j).
        numerator_ij = numerator_j[:, :, None] * numerator_j[:, None, :]
        denominator_ij = denominator_j[:, :, None] * denominator_j[:, None, :]
        glm_ij = torch.log(numerator_ij / denominator_ij) / (log_alpha + self.eps)

        causal_map = glm_ij / (glm_j[:, None, :] + self.eps)
        return causal_map
