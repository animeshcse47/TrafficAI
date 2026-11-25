# Measured benchmark results

The full CPU run used seeds 42, 43 and 44. Each graph model trained for 100 epochs and each telemetry neural model trained for 2,000 epochs. The telemetry experiments used 287 labels per seed and simulated telemetry. Metrics below are measured on chronological holdouts. The CSVs contain per-seed values and mean ± sample standard deviation.

## Synthetic graph ADR

| Model | Holdout MSE | Holdout R² | Neural parameters |
|---|---:|---:|---:|
| MLP | 0.000902 ± 0.000409 | 0.857 ± 0.073 | 17,921 |
| PINTP (λ=0.03) | 0.001186 ± 0.000406 | 0.816 ± 0.064 | 17,924 |
| Random forest | 0.005955 ± 0.000563 | 0.076 ± 0.133 | |
| LightGBM | 0.005311 ± 0.000673 | 0.175 ± 0.142 | |
| XGBoost | 0.005194 ± 0.000745 | 0.193 ± 0.158 | |

PINTP outperformed the three graph tree baselines and trailed the matched MLP. The λ=0 MLP had the lowest mean holdout MSE across the five graph weights. Under low, medium and high training corruption, PINTP mean R² was 0.620, 0.244 and −0.563. The corresponding MLP means were 0.874, 0.504 and −0.509.

See the [benchmark table](outputs/synthetic_results_summary.csv), [weight sweep](outputs/synthetic_lambda_sweep_summary.csv), [robustness table](outputs/synthetic_robustness_summary.csv), and [node forecast](outputs/graph_forecast.png).

## Synthetic telemetry

| Physics weight | Holdout MSE | Holdout R² |
|---:|---:|---:|
| 0 | 0.085847 ± 0.009374 | −7.916 ± 0.974 |
| 0.01 | 0.086752 ± 0.009288 | −8.010 ± 0.965 |
| 0.1 | 0.085706 ± 0.009221 | −7.901 ± 0.958 |
| 1 | 0.080503 ± 0.009040 | −7.361 ± 0.939 |
| 10 | 0.070136 ± 0.008168 | −6.284 ± 0.848 |

At λ=10, mean per-seed MSE reduction relative to the matched MLP was 18.3%. All telemetry models had negative holdout R², indicating poor performance on this simulated future window. Random forest achieved the lowest mean telemetry MSE at 0.017095. The supplied Alibaba CSV ingestion path was tested with a fixture. The actual Alibaba trace requires the download described in the README, so these results establish behavior on the simulated telemetry only.

See the [model comparison](outputs/synthetic_telemetry_results_summary.csv), [ablation table](outputs/synthetic_telemetry_ablation_summary.csv), [learned coefficients](outputs/synthetic_telemetry_coefficients_summary.csv), [forecast figure](outputs/telemetry_forecast.png), and [ablation figure](outputs/telemetry_ablation.png).

## Verification

All five unit tests passed. The notebook executed all cells using the quick profile. The full CLI run produced six raw and summary table pairs, three figures, two forecast exports and two checkpoints. Reloading the graph checkpoint reproduced the recorded time-8 forecast for all 100 nodes. Exact settings and versions are in [run_info.json](outputs/run_info.json).
