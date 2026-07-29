"""
Run this ON srv-vgpu-02, from ~/test_llada:
    python patch_nvtx.py

It patches model_update/model.py in place, adding NVTX range markers
around embed_tokens, build_rope_freqs, and lm_head inside
LLaDAMoEKV.forward. Uses exact string matching (not line numbers,
which drift) so it will fail loudly instead of silently mis-patching
if the surrounding code doesn't match what we expect.
"""

import re

PATH = "model_update/model.py"

with open(PATH, "r") as f:
    src = f.read()

# 1. Add the import if not already present
if "import torch.cuda.nvtx as nvtx" not in src:
    src = src.replace(
        "import torch.nn.functional as F",
        "import torch.nn.functional as F\nimport torch.cuda.nvtx as nvtx",
        1,
    )
    print("Added nvtx import.")
else:
    print("nvtx import already present, skipping.")

# 2. Wrap embed_tokens + build_rope_freqs
old_block = '''        x = self.embed_tokens(input_ids)

        cos, sin = build_rope_freqs(position_offset, position_offset + T, cfg.HD, cfg.THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
'''
new_block = '''        nvtx.range_push("embed_tokens")
        x = self.embed_tokens(input_ids)
        nvtx.range_pop()

        nvtx.range_push("build_rope_freqs")
        cos, sin = build_rope_freqs(position_offset, position_offset + T, cfg.HD, cfg.THETA, device)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)
        nvtx.range_pop()
'''

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("Patched embed_tokens/build_rope_freqs block.")
elif "nvtx.range_push(\"build_rope_freqs\")" in src:
    print("build_rope_freqs block already patched, skipping.")
else:
    raise SystemExit(
        "Could not find expected embed_tokens/build_rope_freqs block. "
        "File may have changed again since this script was written — "
        "paste the current forward() and I'll regenerate this patch."
    )

# 3. Wrap lm_head
old_lm = '''        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv'''
new_lm = '''        x = self.norm(x)

        nvtx.range_push("lm_head")
        logits = self.lm_head(x)
        nvtx.range_pop()
        return logits, new_kv'''

if old_lm in src:
    src = src.replace(old_lm, new_lm, 1)
    print("Patched lm_head block.")
elif "nvtx.range_push(\"lm_head\")" in src:
    print("lm_head block already patched, skipping.")
else:
    raise SystemExit(
        "Could not find expected lm_head block. File may have changed "
        "again since this script was written."
    )

with open(PATH, "w") as f:
    f.write(src)

print(f"Done. Verify with: grep -n nvtx {PATH}")