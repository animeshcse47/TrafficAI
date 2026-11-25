"""Training loops shared between the command line and notebook."""

import torch

from .models import GraphMLP, TelemetryMLP


def tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _optimizer(model, weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 weight_decay=0 if weight else 1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20, min_lr=1e-6
    )
    return optimizer, scheduler


def _step(loss, optimizer, model):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def fit_graph(data, labels, weight, epochs, seed, device):
    torch.manual_seed(seed)
    model = GraphMLP(len(data.degree), physical=bool(weight)).to(device)
    optimizer, scheduler = _optimizer(model, weight)
    times = tensor(data.times, device)
    observed = tensor(labels, device)
    laplacian = tensor(data.laplacian, device)
    source = tensor(data.source, device)
    for _ in range(epochs):
        losses = []
        for batch in torch.randperm(data.split, device=device).split(2):
            t = times[batch]
            if weight:
                prediction, residual = model.residual(t, laplacian, source, training=True)
                physics = weight * residual.square().mean()
            else:
                prediction = model(t.repeat_interleave(len(source)))
                physics = 0
            loss = (prediction - observed[batch]).square().mean() + physics
            _step(loss, optimizer, model)
            losses.append(loss.detach())
        scheduler.step(torch.stack(losses).mean().item())
    model.eval()
    holdout = times[data.split:]
    with torch.enable_grad():
        prediction, residual = model.residual(holdout, laplacian, source)
    return model, prediction.detach().cpu().numpy(), residual.detach().square().mean().item()


def fit_telemetry(data, labelled, weight, epochs, seed, device):
    torch.manual_seed(seed)
    model = TelemetryMLP(physical=bool(weight)).to(device)
    optimizer, scheduler = _optimizer(model, weight)
    features = tensor(data.features, device)
    targets = tensor(data.values, device)
    indices = torch.as_tensor(labelled, device=device)
    for _ in range(epochs):
        loss = (model(features[indices]) - targets[indices]).square().mean()
        if weight:
            collocation = torch.randperm(data.split, device=device)[:256]
            physics = model.residual(features[collocation], training=True).square().mean()
            loss = loss + weight * physics
        _step(loss, optimizer, model)
        scheduler.step(loss.detach().item())
    model.eval()
    with torch.no_grad():
        prediction = model(features[data.split:]).cpu().numpy()
    with torch.enable_grad():
        residual = model.residual(features[data.split:]).detach().square().mean().item()
    return model, prediction, residual
