import torch


def vectorize_state(state_dict, dtype) -> torch.Tensor:
    state = torch.cat([torch.flatten(torch.tensor(s, dtype=dtype)) for s in state_dict.values()])
    state.refine_names('state_dim')
    return state


def get_dtype(cfg_str):
    if cfg_str == 'int32':
        return torch.int32
    elif cfg_str == 'float32':
        return torch.float32
    elif cfg_str == 'float16':
        return torch.float16
    else:
        return torch.float3
