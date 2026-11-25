# Implementation approach

## Graph ADR benchmark

`pintp/data.py` constructs a seeded 100-node undirected graph with 482 edges and solves the reference advection–diffusion–reaction (ADR) system with SciPy RK45. The state evolves according to `du/dt = -(v + D) L u + S`, where `L = degree - adjacency`. The solver generates 200 training slices from time 0 through 7.96 and 100 holdout slices from time 8 through 10.

`pintp/models.py` combines a 32-dimensional node embedding and raw time in four 64-unit tanh layers with 0.1 dropout. For each minibatch, it evaluates all nodes at two times, differentiates each node's prediction with respect to its own time coordinate, and applies the Laplacian across the whole graph. The objective is training-label MSE plus the mean squared ADR residual weighted by λ. The main PINTP run uses λ=0.03. The matched MLP has the same backbone with λ=0. Tree baselines use node ID, time and degree.

The graph PINTP learns three positive multipliers on the reference velocity, diffusion and source profiles. Multipliers start at one. This implements the three additional scalar parameters listed in the handoff and leaves the data-generating equation fixed. Velocity and diffusion multiply the same operator, so their fitted values should be interpreted as regularizer parameters. The graph's target stays in its simulated intensity units, preserving the ADR equation without a change of variables.

The robustness experiment retrains PINTP and MLP at three corruption levels and evaluates the same clean holdout. Training labels receive Gaussian noise, sparse positive exponential bursts and a shared sinusoidal drift. The initial state uses small uniform values with ten randomly chosen high-intensity nodes.

## Sparse-label telemetry

The real CSV path filters one machine while streaming, sorts timestamps and removes duplicates. The chronological 80/20 split occurs before interpolation, binning and scaling. Each partition is interpolated independently, with edge gaps dropped, and compressed to at most 5,820 training and 1,455 test rows. Time and the min-max transformations are anchored to training data, so holdout values can fall outside [0, 1]. With no CSV, a deterministic simulated frame gives the same 7,275-row workflow.

The four-layer tanh network maps normalized time and current CPU, memory, outgoing traffic and disk utilization to incoming traffic. All models receive the same 287 seeded training labels. The physics-informed models also draw training collocation batches of 256 coordinates. The residual is `d²u/dτ² - α du/dτ - βu - γ`, with partial derivatives in time holding the other inputs fixed. The three coefficients are trained from zero. The ablation uses λ values 0, 0.01, 0.1, 1 and 10. Random forest, LightGBM, XGBoost and gradient boosting use the same labeled subset.

## Training and evaluation

Both benchmarks use Adam at learning rate 0.001, gradient clipping at 1, and plateau-based learning-rate reduction. The matched MLP has weight decay 0.00001 and PINTP has zero weight decay, following the reference training protocol. Each telemetry epoch is one full labeled-set update and, when λ is positive, one collocation update. Seeds control graph generation, label selection, initialization and minibatch sampling.

Reported metrics are holdout MSE, MAE, R², and R² divided by parameter count for neural models. Physics penalties use each model's residual coefficients. The λ=0 graph model uses the nominal ADR profiles and the λ=0 telemetry MLP uses zero ODE coefficients for this diagnostic. These residual values measure fit to the chosen equations, which differ by model. Tree models have no autograd residual or neural parameter count.

The graph simulator shares the residual's equation family. The synthetic telemetry checks the implementation and produces independently measured results on a known simulated input. A real-trace run needs the external Alibaba download. The project implements the reference's forecasting task for communication networks. Road-signal actuation and a packet-level simulator are outside that task.

Method source: Zhang et al., [A Physics-Informed Neural Network for Graph-Based Network Traffic Prediction](https://doi.org/10.3390/electronics15153270). The supplied project PDF and approach document specify the benchmark configuration. Optional larger neural baselines and scalability profiling are outside this implementation.
