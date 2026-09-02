import torch

from experiment.lib.agent import DYNAMIC_AGENT_CLASSES, make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.train import train, train_dynamic


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT = "benchmark"
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


def train_benchmark() -> None:
    for dataset_id in DATASETS:
        dataset = D4RLDataset(dataset_id, DEVICE)
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
    train_benchmark()
