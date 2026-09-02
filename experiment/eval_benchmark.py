import gymnasium as gym

from experiment.lib.agent import make_agent
from experiment.lib.dataset import D4RLDataset
from experiment.lib.eval import evaluate, load_dynamic
from experiment.lib.paths import agent_path, table_path
from experiment.lib.table import write_table


EXPERIMENT = "benchmark"
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


def eval_benchmark() -> None:
    header = ["dataset"] + AGENTS
    content = [[""] * len(header) for _ in DATASETS]

    for dataset_index, dataset_id in enumerate(DATASETS):
        dataset = D4RLDataset(dataset_id, DEVICE)
        dynamic = load_dynamic(dataset)
        env = gym.make(dataset.env_id)
        content[dataset_index][0] = dataset_id
        for agent_index, agent_id in enumerate(AGENTS, start=1):
            returns = []
            for model_index in range(1, N_MODEL + 1):
                agent = make_agent(agent_id, dataset, dynamic=dynamic)
                agent.load(agent_path(EXPERIMENT, agent, dataset, model_index))
                returns.append(evaluate(agent, env, dataset, N_EVAL, EVAL_SEED))
                print(
                    f"evaluate model {model_index}/{N_MODEL} "
                    f"agent={agent_id} dataset={dataset_id} "
                    f"return={returns[-1]:.6f}"
                )
            average = sum(returns) / len(returns)
            content[dataset_index][agent_index] = f"{average:.6f}"
        write_table(table_path(EXPERIMENT), header, content)
        env.close()


if __name__ == "__main__":
    eval_benchmark()
