import gymnasium as gym

from experiment.lib.agent import make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.eval import evaluate_state_noise, load_dynamic
from experiment.lib.paths import agent_path, table_path
from experiment.lib.table import write_table


EXPERIMENT = "noise_state"
MODEL_EXPERIMENT = "benchmark"
DEVICE = "cpu"
N_MODEL = 10
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


def eval_noise_state() -> None:
    header = ["task"] + AGENTS
    test_cases = [
        (dataset_id, noise_id, noise_scale)
        for dataset_id in DATASETS
        for noise_id, noise_scale in NOISE_SCALES
    ]
    table_rows = [[""] * len(header) for _ in test_cases]

    for test_index, (dataset_id, noise_id, noise_scale) in enumerate(test_cases):
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = load_dynamic(dataset)
        test_id = f"{EXPERIMENT}_{noise_id}@{dataset_id}"
        env = gym.make(dataset.env_id)
        table_rows[test_index][0] = test_id
        for agent_index, agent_id in enumerate(AGENTS, start=1):
            model_returns = []
            for model_index in range(1, N_MODEL + 1):
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                agent.load(agent_path(MODEL_EXPERIMENT, agent, dataset, model_index))
                model_returns.append(
                    evaluate_state_noise(
                        agent, env, dataset, noise_scale, N_EVAL, EVAL_SEED
                    )
                )
                print(
                    f"evaluate model {model_index}/{N_MODEL} "
                    f"agent={agent_id} task={test_id} "
                    f"return={model_returns[-1]:.6f}"
                )
            average = sum(model_returns) / len(model_returns)
            table_rows[test_index][agent_index] = f"{average:.6f}"
        env.close()
        write_table(table_path(EXPERIMENT), header, table_rows)


if __name__ == "__main__":
    eval_noise_state()
