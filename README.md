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
