from itertools import product
from pathlib import Path

import gymnasium as gym

from joint_learning.lib._metrics import write_metrics
from joint_learning.lib.agent import SCASPL_N_PARAMETERS
from joint_learning.lib.agent import SCCC_N_PARAMETERS
from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.eval import evaluate_state_noise
from joint_learning.lib.paths import metrics_path
from joint_learning.lib.paths import MODEL_ROOT
from joint_learning.lib.paths import table_path
from joint_learning.lib.table import write_table


EXPERIMENT = "noise_state_grid"
DEVICE = "cpu"
N_EVAL = 100
EVAL_SEED = 42

PARAMETER_VALUES = tuple(
    product(
        (0.05, 0.5, 5.0),
        (0.1, 0.25, 0.5),
        (0.25, 1.0, 2.5),
    )
)


def parameter_agent_ids(
    parameters: dict[str, tuple[float, float, float]],
    base_agent_id: str,
    base_parameters: tuple[float, float, float],
) -> list[str]:
    return [
        base_agent_id
        if parameter_values == base_parameters
        else next(
            agent_id
            for agent_id, configured_values in parameters.items()
            if configured_values == parameter_values
        )
        for parameter_values in PARAMETER_VALUES
    ]


SCASPL_AGENTS = parameter_agent_ids(
    SCASPL_N_PARAMETERS,
    "scaspl_n",
    (0.05, 0.25, 1.0),
)

SCCC_AGENTS = parameter_agent_ids(
    SCCC_N_PARAMETERS,
    "sccc_n",
    (5.0, 0.25, 1.0),
)

AGENTS = [
    "td3bc",
    *SCASPL_AGENTS,
    *SCCC_AGENTS,
]

DATASETS = [
    "walker2d_medium",
    "walker2d_hybrid",
    "walker2d_medium_replay",
]

NOISE_SCALES = [
    ("5e-4", 0.0005),
    ("1e-3", 0.001),
    ("5e-3", 0.005),
    ("1e-2", 0.01),
]


def model_paths(agent_id: str, dataset_id: str) -> list[Path]:
    prefix = f"{agent_id}-{dataset_id}-"
    return sorted(
        MODEL_ROOT.glob(f"{prefix}*.pt"),
        key=lambda path: int(path.stem.rsplit("-", maxsplit=1)[1]),
    )


def main() -> None:
    rows: list[list[str]] = []
    header = ["task", *AGENTS]

    for dataset_id in DATASETS:
        dataset = D4RLDataset(dataset_id, DEVICE)
        for noise_id, noise_scale in NOISE_SCALES:
            test_id = f"{EXPERIMENT}_{noise_id}@{dataset_id}"
            row = [test_id]
            env = gym.make(dataset.env_id)
            try:
                for agent_id in AGENTS:
                    paths = model_paths(agent_id, dataset_id)
                    returns = []
                    for model_number, path in enumerate(paths, start=1):
                        print(
                            f"evaluate model {model_number}/{len(paths)} "
                            f"agent={agent_id} task={test_id}"
                        )
                        agent = make_agent(agent_id, dataset)
                        agent.load(path)
                        returns.append(
                            evaluate_state_noise(
                                agent,
                                env,
                                dataset,
                                noise_scale,
                                N_EVAL,
                                EVAL_SEED,
                            )
                        )
                    write_metrics(metrics_path(agent_id, test_id), returns)
                    row.append(f"{sum(returns) / len(returns):.6f}")
            finally:
                env.close()

            rows.append(row)
            write_table(table_path(EXPERIMENT), header, rows)


if __name__ == "__main__":
    main()
