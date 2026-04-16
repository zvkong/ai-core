import random
import numpy as np
import torch

def set_seed(seed:int, deterministic:bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:        
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    """Return the best available computation device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def report_config(seed: int, deterministic: bool = True) -> None:
    """Print current reproducibility configuration."""
    device = get_device()
    print(f"seed:{seed}")
    print(f"device:{device}")
    print(f"deterministic:{deterministic}")
    print(f"torch version: {torch.__version__}")