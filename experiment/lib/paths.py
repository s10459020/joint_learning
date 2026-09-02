from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
RESULT_ROOT = PROJECT_ROOT / "result"
MODEL_ROOT = RESULT_ROOT / "model"
TABLE_ROOT = RESULT_ROOT / "table"


def agent_path(experiment_id: str, agent, dataset, train_index: int) -> Path:
    return MODEL_ROOT / experiment_id / f"{agent.id}-{dataset.id}-{train_index}.pt"


def dynamic_path(dataset) -> Path:
    return MODEL_ROOT / f"dynamics-{dataset.id}.pt"


def table_path(experiment_id: str) -> Path:
    return TABLE_ROOT / f"{experiment_id}.csv"
