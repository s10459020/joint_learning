import torch

from experiment.lib.agent import DYNAMIC_AGENT_CLASSES, make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.train import train, train_dynamic


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT = "stability"
N_MODEL = 10
N_TRAIN = 500_000
BATCH_SIZE = 256
PRINT_INTERVAL = 2_000
MODEL_STEPS = 500_000
MODEL_BATCH_SIZE = 256
MODEL_PRINT_INTERVAL = 10_000

AGENTS = [
    "td3bc",
    "td3bc_xn",
    "td3bc_p",
    "td3bc_xn_gp",
    "td3bc_p_gp",
    "td3bc_gp",
    "aspl",
    "aspl_c",
    "aspl_gp",
    "scas",
    "scas_n",
    "scas_gp",
    "scas_gpn",
    "scaspl",
    "scaspl_n",
    "scaspl_gp",
    "scaspl_c",
    "scaspl_nc",
    "sccc",
    "sccc_n",
    "sccc_gp",
    "sccc_gpn",
]

DATASETS = [
    "walker2d_medium",
    "walker2d_expert",
    "walker2d_hybrid",
    "walker2d_medium_replay",
    "walker2d_expert_replay",
]


def train_stability() -> None:
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
    train_stability()
