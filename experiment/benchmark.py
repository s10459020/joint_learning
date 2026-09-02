import torch

from joint_learning.lib.agent import DYNAMIC_AGENT_CLASSES
from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.paths import metrics_path
from joint_learning.lib.paths import table_path
from joint_learning.lib.table import write_table
from joint_learning.lib.train import train
from joint_learning.lib.train import train_dynamic
from joint_learning.lib._metrics import write_metrics


EXPERIMENT = "benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MODEL = 10
N_TRAIN = 500_000
N_EVAL = 5
BATCH_SIZE = 256
PRINT_INTERVAL = 5_000
RETURN_AVG_WINDOW = 10
MODEL_STEPS = 500_000
MODEL_BATCH_SIZE = 256
MODEL_PRINT_INTERVAL = 10_000

AGENTS = [
    # "bc",
    # "td3bc",
    # "iql",
    "cql",
    "aspl_c",
    "scas_n",
    # "scaspl_n",
    # "sccc_n",
]

DATASETS = [
    "hopper_medium",
    "hopper_expert",
    "hopper_hybrid",
    "hopper_medium_replay",
    "hopper_expert_replay",
    "walker2d_medium",
    "walker2d_expert",
    "walker2d_hybrid",
    "walker2d_medium_replay",
    "walker2d_expert_replay",
    "halfcheetah_medium",
    "halfcheetah_expert",
    "halfcheetah_hybrid",
    "halfcheetah_medium_replay",
    "halfcheetah_expert_replay",
]

_TASKS = [
    # ("scas_n", "hopper_expert"),
    # ("scaspl_n", "hopper_expert"),
    # ("bc", "hopper_medium_replay"),
    # ("bc", "hopper_expert_replay"),
    # ("scas_n", "walker2d_expert"),
    # ("bc", "walker2d_medium_replay"),
    # ("bc", "walker2d_expert_replay"),
]


def run_benchmark() -> None:
    rows: list[list[str]] = []
    header = ["dataset"] + AGENTS

    for dataset_id in DATASETS:
        row = [dataset_id]
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = (
            train_dynamic(dataset, MODEL_STEPS, MODEL_BATCH_SIZE, MODEL_PRINT_INTERVAL)
            if any(agent_id in DYNAMIC_AGENT_CLASSES for agent_id in AGENTS)
            else None
        )
        for agent_id in AGENTS:
            returns = []
            for train_index in range(1, N_MODEL + 1):
                print(
                    f"train {train_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id}"
                )
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                returns.append(
                    train(
                        agent,
                        dataset,
                        EXPERIMENT,
                        train_index,
                        N_TRAIN,
                        N_EVAL,
                        BATCH_SIZE,
                        PRINT_INTERVAL,
                        RETURN_AVG_WINDOW,
                    )
                )
            write_metrics(metrics_path(agent_id, dataset_id), returns)
            mean_return = sum(returns) / len(returns)
            row.append(f"{mean_return:.6f}")
        rows.append(row)
        write_table(table_path(EXPERIMENT), header, rows)

def _run_tasks() -> None:
    for task_index, (agent_id, dataset_id) in enumerate(_TASKS, start=1):
        print(f"task {task_index}/{len(_TASKS)} agent={agent_id} dataset={dataset_id}")
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = (
            train_dynamic(dataset, MODEL_STEPS, MODEL_BATCH_SIZE, MODEL_PRINT_INTERVAL)
            if agent_id in DYNAMIC_AGENT_CLASSES
            else None
        )
        returns = []
        for train_index in range(1, N_MODEL + 1):
            print(
                f"task train {train_index}/{N_MODEL} "
                f"agent={agent_id} dataset={dataset_id}"
            )
            agent = make_agent(agent_id, dataset, dynamic=dynamic)
            returns.append(
                train(
                    agent,
                    dataset,
                    EXPERIMENT,
                    train_index,
                    N_TRAIN,
                    N_EVAL,
                    BATCH_SIZE,
                    PRINT_INTERVAL,
                    RETURN_AVG_WINDOW,
                )
            )
        write_metrics(metrics_path(agent_id, dataset_id), returns)
        mean_return = sum(returns) / len(returns)
        print(f"task mean_return={mean_return:.6f} agent={agent_id} dataset={dataset_id}")


def main() -> None:
    run_benchmark()
    _run_tasks()


if __name__ == "__main__":
    main()
