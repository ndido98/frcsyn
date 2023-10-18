import math

import torch
import torch.nn as nn

from utils import normalize


class AdaFace(nn.Module):
    def __init__(self, embedding_size: int, n_classes: int, margin: float = 0.4, h: float = 0.333, s: float = 64.0, t_alpha: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, n_classes))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(p=2, dim=1, maxnorm=1e-5).mul_(1e5)
        self.margin = margin
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer("t", torch.tensor(1.0))
        self.register_buffer("batch_mean", torch.tensor(20.0))
        self.register_buffer("batch_std", torch.tensor(100.0))

    def forward(self, embbedings: torch.Tensor, norms: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        kernel_norm = normalize(self.kernel, dim=0)
        cosine = torch.matmul(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps) # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, min=-1, max=1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.margin * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.margin + (self.margin * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
