"""
Correctness test for model_update/model.py's CUDAGraphRunner.

CUDAGraphRunner's docstring is explicit that it was written and reasoned
through WITHOUT a CUDA GPU available -- every invariant it relies on was
justified by code inspection, not by a passing run of this test. This file
IS that missing validation. Anyone enabling use_cuda_graph=True
(model_update/generate.py) or USE_CUDA_GRAPH=1 (src/server.py) should run
this first, on real hardware, and treat a failure here as "do not use this
path" rather than something to work around.

Three things are checked, each catching a different class of CUDA-graph
bug:
  1. Graph-replayed output matches plain eager output for the same input
     -- the basic correctness bar.
  2. Replaying the SAME captured graph twice with the SAME input gives the
     SAME output both times -- replay determinism.
  3. Replaying with DIFFERENT input data (same shape, so same graph/key)
     actually reflects the new data, rather than silently returning the
     first capture's stale result -- this is the single most dangerous
     failure mode (wrong output that doesn't crash, doesn't error, just
     quietly generates garbage) and the one most worth a dedicated,
     explicit check rather than trusting the implementation reads right.

CUDA-only (CUDAGraph is a CUDA API) -- skipped with a message if
unavailable.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_fused_model(cfg, device):
    from model_update.model import LLaDAMoEKV, TritonFusedMoEBlock
    model = LLaDAMoEKV(cfg, use_fused_moe=False).to(torch.bfloat16).to(device).eval()
    # Swap in the real Triton-fused MoE blocks: CUDAGraphRunner exists
    # specifically to graph-capture this path (moe_align_block_size had to
    # become zero-host-sync for capture to be legal at all) -- the eager
    # MoEBlock path was never the target and doesn't exercise the thing
    # being validated here.
    for layer in model.layers:
        fused = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16).to(device)
        fused.load_state_dict_from_unfused(layer.mlp)
        layer.mlp = fused
    return model


def test_cuda_graph_matches_eager():
    if not torch.cuda.is_available():
        print("test_cuda_graph_matches_eager: SKIPPED (no CUDA GPU available)")
        return

    from model_update.model import SMALL_CFG, KVCacheBuffer, CUDAGraphRunner

    torch.manual_seed(0)
    device = "cuda"
    cfg = SMALL_CFG
    model = _build_fused_model(cfg, device)

    B, P, T_active, total_len = 2, 8, 8, 32

    def make_cache():
        return KVCacheBuffer(
            num_layers=cfg.NL, batch_size=B, kvh_local=model.layers[0].self_attn.KVH_local,
            max_len=total_len, head_dim=cfg.HD, dtype=torch.bfloat16, device=device,
        )

    # Two separate cache buffers, primed identically, so the eager and
    # graphed paths start from the same state without cross-contaminating
    # each other's cache_buffer (which would invalidate the comparison).
    cache_buffer_eager = make_cache()
    cache_buffer_graph = make_cache()

    prompt_ids = torch.randint(0, cfg.VS, (B, P), device=device)
    active_ids = torch.randint(0, cfg.VS, (B, T_active), device=device)

    with torch.no_grad():
        model(prompt_ids, position_offset=0, cache_buffer=cache_buffer_eager, write_pos=0)
        cache_buffer_eager.commit(P)
        model(prompt_ids, position_offset=0, cache_buffer=cache_buffer_graph, write_pos=0)
        cache_buffer_graph.commit(P)

        eager_logits, _ = model(
            active_ids, position_offset=P, cache_buffer=cache_buffer_eager, write_pos=P,
        )

        runner = CUDAGraphRunner(model)
        graph_logits, _ = runner(
            active_ids, position_offset=P, cache_buffer=cache_buffer_graph, write_pos=P,
        )

    # --- Check 1: graph matches eager for the capturing call itself. ---
    assert eager_logits.shape == graph_logits.shape, (
        f"shape mismatch: eager={tuple(eager_logits.shape)} graph={tuple(graph_logits.shape)}"
    )
    assert torch.equal(eager_logits, graph_logits), (
        "CUDA graph replay produced DIFFERENT logits than the eager forward "
        "for identical input on the capturing call. Do not use "
        "use_cuda_graph=True until this is root-caused."
    )
    print(f"test_cuda_graph_matches_eager: check 1/3 passed (shape={tuple(eager_logits.shape)})")

    # --- Check 2: replaying the same captured graph twice with the same
    # input gives the same output both times. ---
    with torch.no_grad():
        replay_logits_2, _ = runner(
            active_ids, position_offset=P, cache_buffer=cache_buffer_graph, write_pos=P,
        )
    assert torch.equal(graph_logits, replay_logits_2), (
        "Replaying the same captured graph gave a DIFFERENT result the "
        "second time for IDENTICAL input -- replay should be deterministic "
        "for unchanged static-buffer contents."
    )
    print("test_cuda_graph_matches_eager: check 2/3 passed (replay determinism)")

    # --- Check 3: replaying with NEW input data (same shape/key) reflects
    # the new data, not a stale result from the first capture. This is the
    # single most dangerous failure mode -- wrong output with no crash. ---
    new_active_ids = torch.randint(0, cfg.VS, (B, T_active), device=device)
    with torch.no_grad():
        eager_logits_new, _ = model(
            new_active_ids, position_offset=P, cache_buffer=cache_buffer_eager, write_pos=P,
        )
        graph_logits_new, _ = runner(
            new_active_ids, position_offset=P, cache_buffer=cache_buffer_graph, write_pos=P,
        )
    assert not torch.equal(graph_logits, graph_logits_new), (
        "Replaying the graph with DIFFERENT input data returned the SAME "
        "output as before -- the static input buffer is not being updated "
        "before replay (CUDAGraphRunner._replay's .copy_() call). This "
        "would silently corrupt real generation output without any error."
    )
    assert torch.equal(eager_logits_new, graph_logits_new), (
        "CUDA graph replay with new input data diverged from eager. Do not "
        "use use_cuda_graph=True until this is root-caused."
    )
    print("test_cuda_graph_matches_eager: check 3/3 passed (new-data replay reflects new data)")


if __name__ == "__main__":
    test_cuda_graph_matches_eager()
    print("\nAll checks passed (or skipped if no CUDA GPU available).")
