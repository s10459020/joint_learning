import gymnasium as gym

from experiment.lib.agent import make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.eval import evaluate, load_dynamic
from experiment.lib.paths import agent_path, table_path
from experiment.lib.table import write_average_table


EXPERIMENT = "noise_init"
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
    ("5e-2", 0.05),
    ("1e-1", 0.1),
    ("5e-1", 0.5),
    ("1e0", 1.0),
]


def eval_noise_init() -> None:
    header = ["task"] + AGENTS
    test_cases = [
        (dataset_id, noise_id, noise_scale)
        for dataset_id in DATASETS
        for noise_id, noise_scale in NOISE_SCALES
    ]
    test_ids = [f"{EXPERIMENT}_{noise_id}@{dataset_id}" for dataset_id, noise_id, _ in test_cases]
    evaluation_returns = [[[] for _ in AGENTS] for _ in test_cases]

    for test_index, (dataset_id, noise_id, noise_scale) in enumerate(test_cases):
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = load_dynamic(dataset)
        test_id = test_ids[test_index]
        env = gym.make(dataset.env_id, reset_noise_scale=noise_scale)
        for agent_index, agent_id in enumerate(AGENTS):
            for model_index in range(1, N_MODEL + 1):
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                agent.load(agent_path(MODEL_EXPERIMENT, agent, dataset, model_index))
                returns = evaluate(agent, env, dataset, N_EVAL, EVAL_SEED)
                evaluation_returns[test_index][agent_index].extend(returns)
                print(
                    f"evaluate model {model_index}/{N_MODEL} "
                    f"agent={agent_id} task={test_id} "
                    f"return={sum(returns) / len(returns):.6f}"
                )
        env.close()

    write_average_table(table_path(EXPERIMENT, "average"), header, test_ids, evaluation_returns)


if __name__ == "__main__":
    eval_noise_init()
