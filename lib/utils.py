import torch

def get_device(device_argument: str|None) -> torch.device:
    if device_argument is not None:
        return torch.device(device_argument)
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def padded_number(n: int, max_n: int) -> str:
    return str(n).zfill(len(str(max_n)))