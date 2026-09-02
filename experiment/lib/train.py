from experiment.agents.dynamics import Dynamic
from experiment.lib.paths import agent_path
from experiment.lib.paths import dynamic_path


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
    batch_size: int,
    print_interval: int,
) -> None:
    print(
        f"train agent={agent.id} dataset={dataset.id} "
        f"steps={n_train} device={dataset.device}"
    )
    for step in range(1, n_train + 1):
        batch = dataset.sample_batch(batch_size)
        agent.update(batch)

        if step % print_interval == 0 or step == n_train:
            print(f"step={step}/{n_train}")

    path = agent_path(experiment_id, agent, dataset, train_index)
    agent.save(path)
    print(f"saved model: {path}")
