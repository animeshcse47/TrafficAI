# TrafficAI · PINTP

TrafficAI implements Physics-Informed Network Traffic Prediction (PINTP) for communication networks. It benchmarks a graph ADR neural network against a matched MLP and tree models, and runs a sparse-label telemetry experiment using a temporal ODE penalty. Its output is a traffic forecast that can inform resource planning.

## Run locally

Use Python 3.12 on Linux for the tested CPU setup.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m pintp.experiment --profile quick --output runs/quick
```

The quick profile runs every experiment with one seed (15 graph epochs and 150 telemetry epochs). For the three-seed benchmark, use a fresh output directory. Three independent workers are suitable when at least three CPU cores are available.

```bash
python -m pintp.experiment --profile full --jobs 3 --output runs/full
```

On a machine with a compatible CUDA PyTorch installation, use `--device cuda` and omit `--jobs 3`.

## Run in Jupyter or Colab

Open [PINTP_Project.ipynb](PINTP_Project.ipynb) with the project root as the working directory. For local Jupyter, install `jupyterlab` in the same environment and launch `python -m jupyterlab`. In Colab, upload and extract the project archive, open the notebook, and set the working directory to the extracted project root. The Colab install cell uses the existing PyTorch installation. Set `QUICK = False` to run the full profile. Each notebook run creates a separate output directory.

## Use Alibaba Cluster Trace v2018

Download and extract `machine_usage.csv` using the [official Alibaba trace instructions](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md#downloading-the-data). Alibaba provides the download link after a survey and publishes a SHA-256 checksum for the 1.7 GB machine archive. The CSV is headerless. The runner streams it in chunks and selects machine `m_2274` by default.

```bash
python -m pintp.experiment --profile full --jobs 3 \
  --csv data/machine_usage.csv --output runs/alibaba
```

Use `--machine` to select another machine if the supplied file is a subset. Without `--csv`, the CLI runs with deterministic stand-in telemetry. The notebook uses `data/machine_usage.csv` when present. The CSV filenames and `run_info.json` identify the data source. The telemetry features assume contemporaneous CPU, memory, outbound traffic and disk readings are available. The experiment evaluates incoming traffic conditioned on those readings.

## Read the outputs

- `synthetic_results.csv`, `synthetic_lambda_sweep.csv`, and `synthetic_robustness.csv` contain the graph experiments.
- `synthetic_telemetry_*.csv` or `alibaba_v2018_*.csv` contain telemetry results, physics-weight ablations, and learned coefficients.
- Each summary CSV gives means, sample standard deviations and seed counts. A single-seed standard deviation is blank.
- Forecast CSVs, three PNG plots, PyTorch checkpoints and `run_info.json` record the holdout predictions and run settings.

Generate predictions from the included graph checkpoint at new times for the trained graph.

```bash
python -m pintp.predict --checkpoint outputs/graph_pintp.pt \
  --times 8.0 8.5 9.0 9.5 10.0 --output graph_predictions.csv
```

See [APPROACH.md](APPROACH.md) for the implementation and assumptions and [RESULTS.md](RESULTS.md) for the measured benchmark.
