import torch

from experiment.lib.agent import DYNAMIC_AGENT_CLASSES, make_agent
from experiment.lib.dataset import MinariDataset
from experiment.lib.train import train, train_dynamic


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT = "contaminated"
N_MODEL = 10
N_TRAIN = 500_000
BATCH_SIZE = 256
PRINT_INTERVAL = 5_000
MODEL_STEPS = 500_000
MODEL_BATCH_SIZE = 256
MODEL_PRINT_INTERVAL = 10_000

AGENTS = [
    "bc",
    "td3bc",
    "iql",
    "cql",
    "aspl_c",
    "scas_n",
    "scaspl_n",
    "sccc_n",
]

DATASETS = [
    "walker2d_contaminated_10",
    "walker2d_contaminated_30",
    "walker2d_contaminated_50",
    "walker2d_contaminated_70",
    "walker2d_contaminated_90",
]


def train_contaminated() -> None:
    for dataset_id in DATASETS:
        dataset = MinariDataset(dataset_id, DEVICE)
        dynamic = (
            train_dynamic(dataset, MODEL_STEPS, MODEL_BATCH_SIZE, MODEL_PRINT_INTERVAL)
            if any(agent_id in DYNAMIC_AGENT_CLASSES for agent_id in AGENTS)
            else None
        )
        for agent_id in AGENTS:
            for train_index in range(1, N_MODEL + 1):
                print(
                    f"train {train_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id}"
                )
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                train(
                    agent,
                    dataset,
                    EXPERIMENT,
                    train_index,
                    N_TRAIN,
                    BATCH_SIZE,
                    PRINT_INTERVAL,
                )


if __name__ == "__main__":
    train_contaminated()
