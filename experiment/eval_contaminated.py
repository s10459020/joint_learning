import gymnasium as gym

from experiment.lib.agent import make_agent
from experiment.lib.dataset import MinariDataset
from experiment.lib.eval import evaluate, load_dynamic
from experiment.lib.paths import agent_path, table_path
from experiment.lib.table import write_average_table


EXPERIMENT = "contaminated"
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
    "walker2d_contaminated_10",
    "walker2d_contaminated_30",
    "walker2d_contaminated_50",
    "walker2d_contaminated_70",
    "walker2d_contaminated_90",
]


def eval_contaminated() -> None:
    header = ["dataset"] + AGENTS
    evaluation_returns = [[[] for _ in AGENTS] for _ in DATASETS]

    for dataset_index, dataset_id in enumerate(DATASETS):
        dataset = MinariDataset(dataset_id, DEVICE)
        dynamic = load_dynamic(dataset)
        env = gym.make(dataset.env_id)
        for agent_index, agent_id in enumerate(AGENTS):
            for model_index in range(1, N_MODEL + 1):
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                agent.load(agent_path(EXPERIMENT, agent, dataset, model_index))
                returns = evaluate(agent, env, dataset, N_EVAL, EVAL_SEED)
                evaluation_returns[dataset_index][agent_index].extend(returns)
                print(
                    f"evaluate model {model_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id} "
                    f"return={sum(returns) / len(returns):.6f}"
                )
        env.close()

    write_average_table(table_path(EXPERIMENT, "average"), header, DATASETS, evaluation_returns)


if __name__ == "__main__":
    eval_contaminated()
