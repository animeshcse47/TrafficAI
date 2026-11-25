"""Export graph forecasts for a trained benchmark checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models import GraphMLP


def predict_graph(checkpoint, times):
    times = np.asarray(times, dtype=np.float32)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Provide a nonempty sequence of finite times.")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    nodes = state["embedding.weight"].shape[0]
    model = GraphMLP(nodes, physical=True)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        coordinates = torch.from_numpy(times).repeat_interleave(nodes)
        values = model(coordinates).numpy()
    return pd.DataFrame(dict(time=np.repeat(times, nodes), node=np.tile(np.arange(nodes), len(times)),
                             forecast=values.ravel()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict_graph(args.checkpoint, args.times).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
