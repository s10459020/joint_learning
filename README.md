# Joint Learning

Research code for joint learning of action-space constraints and state correction in offline reinforcement learning.

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

## Dataset Installation

To also download the datasets used by the project, run:

```bash
python download_data.py
```

## Verification

Run the installation and device checks after installation:

```bash
python tests/test_install.py
python tests/test_device.py
```

The checks should finish with `install=ok` and `device_test=ok`.

## Experiment Entry Points

Run the main benchmark from the project root:

```bash
python -m experiment.train_benchmark
```

The benchmark training entry point trains the configured agents on the configured datasets and writes models under `result/model/benchmark/`.

After training is complete, evaluate the saved benchmark models separately:

```bash
python -m experiment.eval_benchmark
```

After the benchmark has produced model files, run the noise evaluation experiments:

```bash
python -m experiment.eval_noise_init
python -m experiment.eval_noise_action
python -m experiment.eval_noise_state
```

To run the TD3BC stability experiment:

```bash
python -m experiment.train_stability_td3bc
```

Evaluate the stability models separately:

```bash
python -m experiment.eval_stability_td3bc
```

Each experiment's agents, datasets, device, training steps, evaluation count, and noise scales are configured at the top of its Python file. The experiment scripts should be run from the project root so that the `datasets/` and `result/` directories are resolved correctly.

Experiment outputs are kept in `result/`, which is excluded from version control. This allows the source code to be published first and the generated results to be added later.
