import os
import torch
import torch.distributed as dist

_TP_GROUP = None

def init_distributed():
    """
    Initializes the distributed process group if not already initialized.
    Reads RANK and WORLD_SIZE from environment variables (standard torchrun setup).
    """
    global _TP_GROUP
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if world_size == 1 and "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            
        dist.init_process_group(backend="nccl", init_method="env://")
        
        _TP_GROUP = None

def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP

def get_tp_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=get_tp_group())

def get_tp_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=get_tp_group())
