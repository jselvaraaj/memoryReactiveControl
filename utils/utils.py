import torch


def vectorize_state(state_dict):
    return torch.cat([torch.flatten(torch.tensor(s)) for s in state_dict.values()])
