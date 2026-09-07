# Joint Learning

Research code for a master's research project on joint learning of action-space constraints and state correction in offline reinforcement learning. This repository contains a simplified and reorganized implementation for reproducing the main experiments and supporting follow-up research.

## Included Methods

- BC
- TD3BC
- IQL
- CQL
- ASPL
- SCAS
- SCASPL
- SCCC

SCASPL and SCCC are the joint-learning methods studied in the thesis.

## Installation

### Windows Command Prompt

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -e .
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The commands above install the program without datasets.

### Dataset Installation

Download the datasets used by the experiments. The download includes 18 D4RL
MuJoCo datasets and 5 contaminated datasets (23 datasets in total), requiring
approximately **6.4 GB (5.95 GiB)** of disk space. Reserve at least **7 GB**
to allow for download metadata and temporary files:

```bash
python download_data.py
```

### Verification

Run the installation and device checks:

```bash
python tests/test_install.py
python tests/test_device.py
```

The checks should finish with:

```text
install=ok
device_test=ok
```

## Experiments

The repository contains four experiment groups:

- **Benchmark** — comparison across the selected D4RL MuJoCo datasets.
- **Stability** — analysis of how model variants, normalization, gradient penalties, and pseudo-label compensation affect policy performance and training stability.
- **Noise Evaluation** — robustness under initial-state, action, and state perturbations.
- **Contaminated Dataset** — evaluation under datasets with controlled data-quality interference.

### Benchmark

Train the benchmark models:

```bash
python -m experiment.train_benchmark
```

Evaluate the trained models:

```bash
python -m experiment.eval_benchmark
```

### Noise Evaluation

After benchmark models are available, run:

```bash
python -m experiment.eval_noise_init
python -m experiment.eval_noise_action
python -m experiment.eval_noise_state
```

### Stability

Train and evaluate the stability experiment:

```bash
python -m experiment.train_stability
python -m experiment.eval_stability
```

### Contaminated Dataset

Train and evaluate the contaminated-dataset experiment:

```bash
python -m experiment.train_contaminated
python -m experiment.eval_contaminated
```

## Outputs

Experiment outputs are written under `result/`.

Generated models and results are excluded from version control so that the repository remains focused on the research implementation and experiment workflow.
