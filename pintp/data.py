"""Synthetic graph dynamics and sparse-label telemetry preparation."""

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

FEATURES = ("cpu", "mem", "net_out", "disk_io")
COLUMNS = ("machine_id", "timestamp", "cpu", "mem", "mem_gps", "mkpi",
           "net_in", "net_out", "disk_io")
CORRUPTION = {
    "low": (0.01, 0.01, 0.04, 0.02),
    "medium": (0.02, 0.03, 0.08, 0.05),
    "high": (0.04, 0.05, 0.12, 0.08),
}


@dataclass
class GraphData:
    times: np.ndarray
    values: np.ndarray
    laplacian: np.ndarray
    degree: np.ndarray
    source: np.ndarray
    split: int = 200


@dataclass
class Telemetry:
    features: np.ndarray
    values: np.ndarray
    timestamps: np.ndarray
    split: int
    source: str
    machine: str | None
    y_min: float
    y_range: float


def graph_adr(seed: int) -> GraphData:
    graph = nx.gnm_random_graph(100, 482, seed=seed)
    adjacency = nx.to_numpy_array(graph, nodelist=range(100))
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    sparse = csr_matrix(laplacian)
    source = np.where(degree > degree.mean(), -0.3, 0.4)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(0, 0.1, 100)
    active = rng.choice(100, 10, replace=False)
    initial[active] = rng.uniform(0.5, 0.9, 10)

    def rhs(t, state):
        v = 0.5 * (1 + 0.3 * np.sin(np.pi * t / 5) + 0.2 * np.sin(np.pi * t))
        diffusion = 0.1 / (0.3 + 0.7 * v / 0.5)
        q = 1 + 0.3 * np.sin(2 * np.pi * t / 5) + 0.2 * np.cos(np.pi * t)
        return -(v + diffusion) * (sparse @ state) + q * source

    times = np.r_[np.arange(200) * 0.04, np.linspace(8, 10, 100)]
    solution = solve_ivp(rhs, (0, 10), initial, t_eval=times, rtol=1e-8, atol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    return GraphData(times, solution.y.T, laplacian, degree, source)


def corrupt_labels(values, level, seed):
    sigma, probability, scale, drift = CORRUPTION[level]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, values.shape)
    bursts = (rng.random(values.shape) < probability) * rng.exponential(scale, values.shape)
    trend = drift * np.sin(np.linspace(0, 2 * np.pi, len(values)))[:, None]
    return values + noise + bursts + trend


def _machine_frame(path, machine):
    keep = ["machine_id", "timestamp", *FEATURES, "net_in"]
    selected = []
    for chunk in pd.read_csv(path, names=COLUMNS, usecols=keep, header=None,
                             dtype={"machine_id": str}, chunksize=2_000_000):
        matches = chunk.loc[chunk.machine_id == machine, keep[1:]]
        if not matches.empty:
            selected.append(matches.copy())
    if not selected:
        raise ValueError(f"Machine {machine!r} was not found. Set --machine to an ID in the CSV.")
    return (pd.concat(selected, ignore_index=True).apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp"])
            .sort_values("timestamp", kind="stable").drop_duplicates("timestamp"))


def _simulated_frame():
    rng = np.random.default_rng(2026)
    t = np.linspace(0, 1, 7275)
    cpu = 48 + 18 * np.sin(15 * t) + rng.normal(0, 2.8, len(t))
    mem = 55 + 10 * np.sin(11 * t + 0.4) + rng.normal(0, 1.5, len(t))
    net_out = 38 + 12 * np.sin(18 * t + 1) + rng.normal(0, 2, len(t))
    disk = 27 + 8 * np.cos(21 * t) + rng.normal(0, 1.8, len(t))
    net_in = (18 + 0.24 * cpu + 0.15 * mem + 0.38 * net_out + 0.12 * disk
              + 2 * np.sin(8 * t) + rng.normal(0, 0.8, len(t)))
    return pd.DataFrame(dict(timestamp=t, cpu=cpu, mem=mem, net_out=net_out,
                             disk_io=disk, net_in=net_in))


def _prepare_partition(frame, bins):
    frame = frame.interpolate(method="linear", limit_area="inside").dropna()
    if len(frame) > bins:
        group = np.arange(len(frame)) * bins // len(frame)
        return frame.groupby(group, sort=False).mean()
    return frame


def telemetry_data(path: Path | None, machine="m_2274") -> Telemetry:
    frame = _simulated_frame() if path is None else _machine_frame(path, machine)
    split = int(0.8 * len(frame))
    train = _prepare_partition(frame.iloc[:split], 5820)
    test = _prepare_partition(frame.iloc[split:], 1455)
    if len(train) < 287 or len(test) < 2:
        raise ValueError("Telemetry requires at least 287 clean training rows and two test rows.")
    time = pd.concat([train.timestamp, test.timestamp]).to_numpy()
    # The training endpoint fixes the scale. Future times may exceed 1.
    train_span = time[len(train) - 1] - time[0]
    tau = (time - time[0]) / train_span
    x_scaler = MinMaxScaler().fit(train.loc[:, list(FEATURES)])
    y_scaler = MinMaxScaler().fit(train[["net_in"]])
    joined = pd.concat([train, test])
    x = np.column_stack([tau, x_scaler.transform(joined.loc[:, list(FEATURES)])])
    return Telemetry(x, y_scaler.transform(joined[["net_in"]]).ravel(), time,
                     len(train), "synthetic_telemetry" if path is None else "alibaba_v2018",
                     None if path is None else machine, float(y_scaler.data_min_[0]),
                     float(1 / y_scaler.scale_[0]))
