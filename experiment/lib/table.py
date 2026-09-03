import csv
from pathlib import Path


def write_average_table(
    path: Path, header: list[str], labels: list[str], returns: list[list[list[float]]]
) -> None:
    write_metric_table(path, header, labels, returns, lambda values: sum(values) / len(values))


def write_pr95_table(
    path: Path, header: list[str], labels: list[str], returns: list[list[list[float]]]
) -> None:
    write_metric_table(path, header, labels, returns, percentile_95)


def write_metric_table(
    path: Path,
    header: list[str],
    labels: list[str],
    returns: list[list[list[float]]],
    metric,
) -> None:
    rows = []
    for label, test_returns in zip(labels, returns):
        rows.append([label] + [f"{metric(values):.6f}" for values in test_returns])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def percentile_95(values: list[float]) -> float:
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * 0.95
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = index - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction
