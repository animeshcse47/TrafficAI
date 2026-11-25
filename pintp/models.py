"""MLP backbones and differentiable physical residuals."""

import torch
from torch import nn


def backbone(width):
    layers = []
    for _ in range(4):
        layers.extend([nn.Linear(width, 64), nn.Tanh(), nn.Dropout(0.1)])
        width = 64
    return nn.Sequential(*layers, nn.Linear(width, 1))


class GraphMLP(nn.Module):
    def __init__(self, nodes, physical):
        super().__init__()
        self.embedding = nn.Embedding(nodes, 32)
        self.predictor = backbone(33)
        self.log_scales = nn.Parameter(torch.zeros(3)) if physical else None

    def forward(self, time):
        nodes = self.embedding.num_embeddings
        repeated = time.reshape(-1, nodes, 1)
        embedding = self.embedding.weight.expand(len(repeated), -1, -1)
        return self.predictor(torch.cat([embedding, repeated], dim=-1)).squeeze(-1)

    def residual(self, time, laplacian, source, *, training=False):
        expanded = time.repeat_interleave(len(source)).detach().requires_grad_(True)
        prediction = self(expanded)
        derivative = torch.autograd.grad(prediction.sum(), expanded, create_graph=training)[0]
        v = 0.5 * (1 + 0.3 * torch.sin(torch.pi * time / 5)
                   + 0.2 * torch.sin(torch.pi * time))
        diffusion = 0.1 / (0.3 + 0.7 * v / 0.5)
        q = 1 + 0.3 * torch.sin(2 * torch.pi * time / 5) + 0.2 * torch.cos(torch.pi * time)
        scales = self.log_scales.exp() if self.log_scales is not None else (1., 1., 1.)
        transport = scales[0] * v + scales[1] * diffusion
        residual = (derivative.reshape_as(prediction) + transport[:, None] * (prediction @ laplacian)
                    - scales[2] * q[:, None] * source)
        return prediction, residual


class TelemetryMLP(nn.Module):
    def __init__(self, physical):
        super().__init__()
        self.predictor = backbone(5)
        self.coefficients = nn.Parameter(torch.zeros(3)) if physical else None

    def forward(self, features):
        return self.predictor(features).squeeze(-1)

    def residual(self, features, *, training=False):
        tau = features[:, :1].detach().requires_grad_(True)
        prediction = self(torch.cat([tau, features[:, 1:]], dim=1))
        du = torch.autograd.grad(prediction.sum(), tau, create_graph=True)[0].ravel()
        ddu = torch.autograd.grad(du.sum(), tau, create_graph=training)[0].ravel()
        alpha, beta, gamma = self.coefficients if self.coefficients is not None else (0., 0., 0.)
        return ddu - alpha * du - beta * prediction - gamma
