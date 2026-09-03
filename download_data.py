from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DATASET_ROOT = Path(__file__).resolve().parent / "datasets"
D4RL_REPOSITORY = "imone/D4RL"
D4RL_REVISION = "9ed58d8a76c12f4c1cb5db51fa71389e703d2ec5"
D4RL_DATASET_FILES = (
    "hopper_random-v2.hdf5",
    "hopper_medium-v2.hdf5",
    "hopper_expert-v2.hdf5",
    "hopper_medium_expert-v2.hdf5",
    "hopper_medium_replay-v2.hdf5",
    "hopper_full_replay-v2.hdf5",
    "walker2d_random-v2.hdf5",
    "walker2d_medium-v2.hdf5",
    "walker2d_expert-v2.hdf5",
    "walker2d_medium_expert-v2.hdf5",
    "walker2d_medium_replay-v2.hdf5",
    "walker2d_full_replay-v2.hdf5",
    "halfcheetah_random-v2.hdf5",
    "halfcheetah_medium-v2.hdf5",
    "halfcheetah_expert-v2.hdf5",
    "halfcheetah_medium_expert-v2.hdf5",
    "halfcheetah_medium_replay-v2.hdf5",
    "halfcheetah_full_replay-v2.hdf5",
)
CONTAMINATED_REPOSITORY = "s10459020/joint-learning-contaminated"
CONTAMINATED_REVISION = "ee35ece0c503d7c7c4333c3fe46d881a5f2392a7"
CONTAMINATED_DATASETS = (
    "walker2d_contaminated_10",
    "walker2d_contaminated_30",
    "walker2d_contaminated_50",
    "walker2d_contaminated_70",
    "walker2d_contaminated_90",
)


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    for dataset_index, filename in enumerate(D4RL_DATASET_FILES, start=1):
        destination = DATASET_ROOT / filename
        if destination.exists():
            print(f"skip {dataset_index}/{len(D4RL_DATASET_FILES)} {filename}")
            continue

        print(f"download {dataset_index}/{len(D4RL_DATASET_FILES)} {filename}")
        hf_hub_download(
            repo_id=D4RL_REPOSITORY,
            repo_type="dataset",
            filename=filename,
            revision=D4RL_REVISION,
            local_dir=DATASET_ROOT,
        )

    for dataset_index, dataset_id in enumerate(CONTAMINATED_DATASETS, start=1):
        destination = DATASET_ROOT / dataset_id
        if destination.exists():
            print(f"skip {dataset_index}/{len(CONTAMINATED_DATASETS)} {dataset_id}")
            continue

        print(f"download {dataset_index}/{len(CONTAMINATED_DATASETS)} {dataset_id}")
        snapshot_download(
            repo_id=CONTAMINATED_REPOSITORY,
            repo_type="dataset",
            revision=CONTAMINATED_REVISION,
            local_dir=DATASET_ROOT,
            allow_patterns=f"{dataset_id}/data/*",
        )

    print(f"data ready at {DATASET_ROOT}")


if __name__ == "__main__":
    main()
