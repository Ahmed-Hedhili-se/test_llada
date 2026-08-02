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


def load_weights_tp(model, weight_dir: str, verbose: bool = True):
    import json
    from safetensors import safe_open
    
    index_path = os.path.join(weight_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        wmap = json.load(f)["weight_map"]

    shards = {}
    for hk, shard in wmap.items():
        shards.setdefault(shard, []).append(hk)

    def _hf_to_our_key(hk: str):
        if hk.startswith("model."):
            return hk[len("model."):]
        if hk == "lm_head.weight":
            return hk
        return None

    sd = model.state_dict()
    mapped, mismatches = 0, []
    
    tp_size = get_tp_size()
    tp_rank = get_tp_rank()

    for shard_name in sorted(shards):
        path = os.path.join(weight_dir, shard_name)
        f = safe_open(path, framework="pt", device="cpu")
        for hk in shards[shard_name]:
            mk = _hf_to_our_key(hk)
            if mk is None:
                continue

            if ".mlp.experts." in mk:
                parts = mk.split('.')
                expert_idx = int(parts[4])
                
                if tp_size > 1:
                    num_local_experts = 64 // tp_size 
                    if not (tp_rank * num_local_experts <= expert_idx < (tp_rank + 1) * num_local_experts):
                        continue
                    
                    local_idx = expert_idx - tp_rank * num_local_experts
                    parts[4] = str(local_idx)
                    mk = ".".join(parts)
            
            if mk not in sd:
                mismatches.append(f"missing in our model: {hk} → {mk}")
                continue

            t = f.get_tensor(hk)
            
            if t.shape == sd[mk].shape:
                sd[mk].copy_(t)
                mapped += 1
            elif tp_size > 1:
                # Column parallel (slice dim 0)
                if ".self_attn.q_proj" in mk or ".self_attn.k_proj" in mk or ".self_attn.v_proj" in mk:
                    dim0_step = t.shape[0] // tp_size
                    sd[mk].copy_(t[tp_rank * dim0_step : (tp_rank + 1) * dim0_step, :])
                    mapped += 1
                # Row parallel (slice dim 1)
                elif ".self_attn.o_proj" in mk:
                    dim1_step = t.shape[1] // tp_size
                    sd[mk].copy_(t[:, tp_rank * dim1_step : (tp_rank + 1) * dim1_step])
                    mapped += 1
                else:
                    mismatches.append(f"shape mismatch {hk}: hf={t.shape} ours={sd[mk].shape}")
            else:
                mismatches.append(f"shape mismatch {hk}: hf={t.shape} ours={sd[mk].shape}")

    if mismatches:
        print(f"  Issues ({len(mismatches)}):")
        for m in mismatches[:20]:
            print(f"    {m}")
    if verbose:
        total = sum(1 for k in wmap if _hf_to_our_key(k) is not None)
        print(f"  Mapped {mapped} tensors (expected for Rank {tp_rank}).")

    model.load_state_dict(sd, strict=False)
    return model
