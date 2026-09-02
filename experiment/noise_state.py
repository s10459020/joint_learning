import gymnasium as gym

from joint_learning.lib.agent import make_agent
from joint_learning.lib.dataset import D4RLDataset
from joint_learning.lib.eval import evaluate_state_noise
from joint_learning.lib.paths import agent_path
from joint_learning.lib.paths import metrics_path
from joint_learning.lib.paths import table_path
from joint_learning.lib._metrics import write_metrics
from joint_learning.lib.table import write_table


EXPERIMENT = "noise_state"
DEVICE = "cpu"
N_MODEL = 1
N_EVAL = 100
EVAL_SEED = 42

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


def main() -> None:
    rows: list[list[str]] = []
    header = ["task"] + AGENTS

    for dataset_id in DATASETS:
        dataset = D4RLDataset(dataset_id, DEVICE)
        for noise_id, noise_scale in NOISE_SCALES:
            test_id = f"{EXPERIMENT}_{noise_id}@{dataset_id}"
            row = [test_id]
            env = gym.make(dataset.env_id)
            try:
                for agent_id in AGENTS:
                    returns = []
                    for model_index in range(1, N_MODEL + 1):
                        print(
                            f"evaluate model {model_index}/{N_MODEL} "
                            f"agent={agent_id} task={test_id}"
                        )
                        agent = make_agent(agent_id, dataset)
                        agent.load(agent_path(agent, dataset, model_index))
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
