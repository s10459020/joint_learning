import torch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    values = torch.ones((2, 2), dtype=torch.float32, device=device)
    result = values @ values
    torch.testing.assert_close(result, torch.full_like(result, 2.0))

    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_version={torch.version.cuda}")
    print(f"device={device}")
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("device_test=ok")


if __name__ == "__main__":
    main()
