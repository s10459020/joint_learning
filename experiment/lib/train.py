import gymnasium as gym

from experiment.agents.dynamics import Dynamic
from experiment.lib.eval import evaluate
from experiment.lib.paths import agent_path
from experiment.lib.paths import dynamic_path
from experiment.lib.paths import plot_path
from experiment.lib.plot import save_training_plot


def train_dynamic(dataset, model_steps: int, model_batch_size: int, model_print_interval: int) -> Dynamic:
    dynamic = Dynamic(dataset.obs_size, dataset.act_size, device=dataset.device)
    path = dynamic_path(dataset)
    if path.exists():
        dynamic.load(path)
        return dynamic.eval().freeze()

    print(f"train dynamic: {dataset.id}")
    for step in range(1, model_steps + 1):
        dynamic.update(dataset.sample_batch(model_batch_size))
        if step % model_print_interval == 0:
            print(f"model step {step}/{model_steps}")

    dynamic.save(path)
    return dynamic.eval().freeze()


def train(
    agent,
    dataset,
    experiment_id: str,
    train_index: int,
    n_train: int,
    n_eval: int,
    batch_size: int,
    print_interval: int,
    return_avg_window: int,
) -> float:
    env = gym.make(dataset.env_id)
    history: list[tuple[int, float]] = []
    moving_avg_return = 0.0

    print(
        f"train agent={agent.id} dataset={dataset.id} "
        f"steps={n_train} device={dataset.device}"
    )
    try:
        for step in range(1, n_train + 1):
            batch = dataset.sample_batch(batch_size)
            agent.update(batch)

            if step % print_interval == 0 or step == n_train:
                episode_return = evaluate(agent, env, dataset, n_eval)
                history.append((step, episode_return))
                recent_returns = [value for _, value in history[-return_avg_window:]]
                moving_avg_return = sum(recent_returns) / len(recent_returns)
                print(
                    f"step={step} "
                    f"return={episode_return:.6g} "
                    f"moving_avg_return={moving_avg_return:.6g}"
                )
    finally:
        env.close()

    path = agent_path(agent, dataset, train_index)
    agent.save(path)
    save_training_plot(
        history,
        plot_path(experiment_id, agent, dataset),
        f"{experiment_id} {agent.id} {dataset.id}",
    )
    print(f"saved model: {path}")
    return moving_avg_return
