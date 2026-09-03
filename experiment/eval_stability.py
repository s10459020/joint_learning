import gymnasium as gym

from experiment.lib.agent import make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.eval import evaluate, load_dynamic
from experiment.lib.paths import agent_path, table_path
from experiment.lib.table import write_table


EXPERIMENT = "stability"
DEVICE = "cpu"
N_MODEL = 10
N_EVAL = 100
EVAL_SEED = 42

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


def eval_stability() -> None:
    header = ["dataset"] + AGENTS
    table_rows = [[""] * len(header) for _ in DATASETS]

    for dataset_index, dataset_id in enumerate(DATASETS):
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = load_dynamic(dataset)
        env = gym.make(dataset.env_id)
        table_rows[dataset_index][0] = dataset_id
        for agent_index, agent_id in enumerate(AGENTS, start=1):
            model_returns = []
            for model_index in range(1, N_MODEL + 1):
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                agent.load(agent_path(EXPERIMENT, agent, dataset, model_index))
                model_returns.append(evaluate(agent, env, dataset, N_EVAL, EVAL_SEED))
                print(
                    f"evaluate model {model_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id} "
                    f"return={model_returns[-1]:.6f}"
                )
            average = sum(model_returns) / len(model_returns)
            table_rows[dataset_index][agent_index] = f"{average:.6f}"
        write_table(table_path(EXPERIMENT), header, table_rows)
        env.close()


if __name__ == "__main__":
    eval_stability()
