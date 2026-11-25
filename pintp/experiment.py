"""Run both benchmarks and export measured results."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .data import CORRUPTION, corrupt_labels, graph_adr, telemetry_data
from .training import fit_graph, fit_telemetry

GRAPH_WEIGHTS = (0., 0.01, 0.03, 0.1, 1.)
TELEMETRY_WEIGHTS = (0., 0.01, 0.1, 1., 10.)
PROFILES = {"quick": (15, 150, (42,)), "full": (100, 2000, (42, 43, 44))}


@dataclass(frozen=True)
class Config:
    output: Path = Path("outputs")
    csv: Path | None = None
    machine: str = "m_2274"
    profile: str = "quick"
    device: str = "cpu"
    jobs: int = 1


def metrics(actual, predicted, *, residual=None, params=None):
    r2 = r2_score(actual.ravel(), predicted.ravel())
    return dict(mse=mean_squared_error(actual.ravel(), predicted.ravel()),
                mae=mean_absolute_error(actual.ravel(), predicted.ravel()),
                r2=r2, l_phys=residual, parameters=params, per=r2 / params if params else None)


def trees(seed, telemetry=False):
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10,
                                               min_samples_split=5, min_samples_leaf=2,
                                               n_jobs=1, random_state=seed),
        "LightGBM": LGBMRegressor(n_estimators=150, learning_rate=0.1, max_depth=4,
                                  num_leaves=15, verbosity=-1, n_jobs=1, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 n_jobs=1, random_state=seed),
    }
    if telemetry:
        models["GradientBoosting"] = GradientBoostingRegressor(random_state=seed)
    return models


def tree_scores(actual, train_x, train_y, test_x, seed, telemetry=False):
    names = [f"x{i}" for i in range(train_x.shape[1])]
    train_x = pd.DataFrame(train_x, columns=names)
    test_x = pd.DataFrame(test_x, columns=names)
    for name, model in trees(seed, telemetry).items():
        prediction = model.fit(train_x, train_y).predict(test_x)
        yield dict(model=name, weight=None, seed=seed, **metrics(actual, prediction))


def neural_score(actual, prediction, residual, model, weight, seed):
    params = sum(param.numel() for param in model.parameters())
    return dict(model="PINTP" if weight else "MLP", weight=weight, seed=seed,
                **metrics(actual, prediction, residual=residual, params=params))


def _graph(config, seed, epochs, save):
    data = graph_adr(seed)
    train, actual = np.split(data.values, [data.split])
    sweep, robustness = [], []
    for weight in GRAPH_WEIGHTS:
        model, predicted, residual = fit_graph(data, train, weight, epochs, seed, config.device)
        sweep.append(neural_score(actual, predicted, residual, model, weight, seed))
        if weight == 0.03 and save:
            torch.save(model.state_dict(), config.output / "graph_pintp.pt")
            pd.DataFrame(dict(time=np.repeat(data.times[data.split:], 100),
                              node=np.tile(np.arange(100), len(actual)),
                              observed=actual.ravel(), forecast=predicted.ravel())
                         ).to_csv(config.output / "graph_forecasts.csv", index=False)

    coordinates = np.column_stack([np.tile(np.arange(100), len(data.times)),
                                   np.repeat(data.times, 100), np.tile(data.degree, len(data.times))])
    boundary = train.size
    main = [row for row in sweep if row["weight"] in (0., 0.03)]
    main.extend(tree_scores(actual, coordinates[:boundary], train.ravel(),
                            coordinates[boundary:], seed))
    for level in CORRUPTION:
        noisy = corrupt_labels(train, level, seed)
        for weight in (0., 0.03):
            model, predicted, residual = fit_graph(data, noisy, weight, epochs, seed, config.device)
            robustness.append(dict(level=level, **neural_score(
                actual, predicted, residual, model, weight, seed
            )))
    print(f"Seed {seed}: graph sweep, baselines and robustness finished", flush=True)
    return main, sweep, robustness


def _telemetry(config, data, seed, epochs, save):
    rng = np.random.default_rng(seed)
    labelled = rng.choice(data.split, 287, replace=False)
    actual = data.values[data.split:]
    sweep, coefficients = [], []
    for weight in TELEMETRY_WEIGHTS:
        model, prediction, residual = fit_telemetry(data, labelled, weight, epochs, seed, config.device)
        score = neural_score(actual, prediction, residual, model, weight, seed)
        sweep.append(score)
        if weight:
            alpha, beta, gamma = model.coefficients.detach().cpu().tolist()
            coefficients.append(dict(seed=seed, weight=weight, alpha=alpha, beta=beta, gamma=gamma))
        if weight == 1.0 and save:
            torch.save(model.state_dict(), config.output / "telemetry_pintp.pt")
            pd.DataFrame(dict(timestamp=data.timestamps[data.split:], observed=actual,
                              forecast=prediction, source=data.source,
                              observed_raw=actual * data.y_range + data.y_min,
                              forecast_raw=prediction * data.y_range + data.y_min)
                         ).to_csv(config.output / "telemetry_forecasts.csv", index=False)
    for score in sweep:
        score["mse_reduction_pct"] = 100 * (1 - score["mse"] / sweep[0]["mse"])
    main = [row for row in sweep if row["weight"] in (0., 0.1, 1.)]
    main.extend(tree_scores(actual, data.features[labelled], data.values[labelled],
                            data.features[data.split:], seed, telemetry=True))
    print(f"Seed {seed}: {data.source} ablation and baselines finished", flush=True)
    return tuple([dict(source=data.source, **row) for row in table]
                 for table in (main, sweep, coefficients))


def _run_seed(config, data, seed):
    torch.set_num_threads(1)
    graph_epochs, telemetry_epochs, seeds = PROFILES[config.profile]
    save = seed == seeds[0]
    return (*_graph(config, seed, graph_epochs, save),
            *_telemetry(config, data, seed, telemetry_epochs, save))


def _write_tables(output, tables, source):
    keys = (
        ("synthetic_results", ["model", "weight"]),
        ("synthetic_lambda_sweep", ["weight"]),
        ("synthetic_robustness", ["level", "model", "weight"]),
        (f"{source}_results", ["source", "model", "weight"]),
        (f"{source}_ablation", ["source", "weight"]),
        (f"{source}_coefficients", ["source", "weight"]),
    )
    for (name, groups), rows in zip(keys, tables):
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f"{name}.csv", index=False)
        values = [key for key in frame.select_dtypes(include="number") if key not in {*groups, "seed"}]
        grouped = frame.groupby(groups, dropna=False)
        summary = grouped[values].agg(["mean", "std"])
        summary.columns = [f"{key}_{stat}" for key, stat in summary.columns]
        summary["seeds"] = grouped.size()
        summary.to_csv(output / f"{name}_summary.csv")


def _plot_forecasts(output, source):
    graphs = pd.read_csv(output / "graph_forecasts.csv")
    node = graphs.loc[graphs.node == 0]
    telemetry = pd.read_csv(output / "telemetry_forecasts.csv")
    for frame, x, name, title, ylabel in (
        (node, "time", "graph_forecast", "Graph ADR holdout · node 0", "Simulated traffic intensity"),
        (telemetry, "timestamp", "telemetry_forecast", f"{source} holdout", "Training-scaled net_in"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.plot(frame[x], frame.observed, color="#30465e", label="Observed", linewidth=1.4)
        ax.plot(frame[x], frame.forecast, color="#d66430", label="PINTP", linewidth=1.4)
        ax.set(title=title, xlabel=x.capitalize(), ylabel=ylabel)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)

    ablation = pd.read_csv(output / f"{source}_ablation_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.errorbar(range(len(ablation)), ablation.mse_mean, yerr=ablation.mse_std.fillna(0),
                color="#177568", marker="o", capsize=4)
    ax.set(xticks=range(len(ablation)), xticklabels=ablation.weight,
           xlabel="Physics weight", ylabel="Holdout MSE", title=f"{source} physics ablation")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output / "telemetry_ablation.png", dpi=150)
    plt.close(fig)


def run(config: Config):
    started = time.perf_counter()
    if config.jobs < 1 or (config.jobs > 1 and config.device != "cpu"):
        raise ValueError("Use a positive job count and one worker for GPU runs.")
    if config.output.exists() and any(config.output.iterdir()):
        raise FileExistsError(f"{config.output} already contains files. Choose an empty --output directory.")
    data = telemetry_data(config.csv, config.machine)
    config.output.mkdir(parents=True, exist_ok=True)
    graph_epochs, telemetry_epochs, seeds = PROFILES[config.profile]
    if data.machine is None:
        print("Telemetry source: SYNTHETIC STAND-IN. These are not Alibaba measurements.", flush=True)
    print(f"Profile {config.profile}: graph {graph_epochs} epochs, telemetry "
          f"{telemetry_epochs} epochs, seeds {seeds}", flush=True)
    if config.jobs == 1:
        results = [_run_seed(config, data, seed) for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=min(config.jobs, len(seeds)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            tasks = [pool.submit(_run_seed, config, data, seed) for seed in seeds]
            results = [task.result() for task in tasks]
    tables = [[row for seed_result in group for row in seed_result] for group in zip(*results)]
    _write_tables(config.output, tables, data.source)
    _plot_forecasts(config.output, data.source)
    info = {
        **asdict(config), "source": data.source, "using_real_alibaba": data.machine is not None,
        "machine": data.machine, "seeds": seeds, "graph_epochs": graph_epochs,
        "telemetry_epochs": telemetry_epochs, "training_labels": 287,
        "telemetry_train_rows": data.split, "telemetry_test_rows": len(data.values) - data.split,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "torch", "numpy", "pandas", "scipy", "scikit-learn", "networkx", "lightgbm",
            "xgboost", "matplotlib"
        )},
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (config.output / "run_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    print(f"Artifacts saved to {config.output}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--csv", type=Path, help="Headerless Alibaba machine_usage.csv")
    parser.add_argument("--machine", default="m_2274")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--jobs", type=int, default=1, help="Independent CPU seed workers")
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
