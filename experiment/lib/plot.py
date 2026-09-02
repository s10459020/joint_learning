from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_training_plot(history: list[tuple[int, float]], path: Path, title: str) -> None:
    steps, returns = zip(*history)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(steps, returns)
    plt.xlabel("step")
    plt.ylabel("raw return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
