# device.py

import torch

def get_device(mode="auto"):
    """
    Get the optimal computing device for PyTorch operations.

    Args:
        mode (str): The mode of device selection.
                    "auto": CUDA > MPS > CPU
                    "cuda": CUDA > CPU
                    "cpu": CPU only

    Returns:
        torch.device: The selected PyTorch device.
    """
    mode = mode.lower()

    if mode == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    elif mode == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    elif mode == "cpu":
        device = torch.device("cpu")

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'auto', 'cuda', or 'cpu'.")

    return device


# Quick demonstration (remove in production)
if __name__ == "__main__":
    for mode in ["auto", "cuda", "cpu"]:
        device = get_device(mode)
        print(f"Device selected (mode={mode}): {device}")