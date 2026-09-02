from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
RESULT_ROOT = PROJECT_ROOT / "result"
MODEL_ROOT = RESULT_ROOT / "model"
METRICS_ROOT = RESULT_ROOT / "metrics"
PLOTS_ROOT = RESULT_ROOT / "plots"
TABLE_ROOT = RESULT_ROOT / "table"


def agent_path(agent, dataset, train_index: int) -> Path:
    return MODEL_ROOT / f"{agent.id}-{dataset.id}-{train_index}.pt"


def dynamic_path(dataset) -> Path:
    return MODEL_ROOT / f"dynamics-{dataset.id}.pt"


def metrics_path(agent_id: str, dataset_id: str) -> Path:
    return METRICS_ROOT / f"{agent_id}-{dataset_id}.txt"


def plot_path(experiment_id: str, agent, dataset) -> Path:
    return PLOTS_ROOT / experiment_id / f"{agent.id}-{dataset.id}.png"


def table_path(experiment_id: str) -> Path:
    return TABLE_ROOT / f"{experiment_id}.csv"
