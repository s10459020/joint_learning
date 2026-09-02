import sys

import gymnasium
import h5py
import huggingface_hub
import experiment
import matplotlib
import numpy
import torch


ENVIRONMENT_IDS = (
    "Hopper-v5",
    "Walker2d-v5",
    "HalfCheetah-v5",
)


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    print(f"experiment={experiment.__file__}")
    print(f"gymnasium={gymnasium.__version__}")
    print(f"h5py={h5py.__version__}")
    print(f"huggingface_hub={huggingface_hub.__version__}")
    print(f"matplotlib={matplotlib.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"torch={torch.__version__}")

    for environment_id in ENVIRONMENT_IDS:
        environment = gymnasium.make(environment_id)
        try:
            observation, _ = environment.reset(seed=0)
            observation, _, _, _, _ = environment.step(environment.action_space.sample())
            print(f"{environment_id}=ok observation_shape={observation.shape}")
        finally:
            environment.close()

    print("install=ok")


if __name__ == "__main__":
    main()
