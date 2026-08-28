"""DMInfr — optimized inference for LLaDA-MoE-7B-A1B-Instruct.

Layout
------
``dminfr.engine``     the optimized engine: Triton fused MoE, block-wise KV
                      cache, fused RMSNorm/decode/RoPE kernels, TP+EP.
                      This is what actually runs in production.
``dminfr.serving``    the OpenAI-compatible server and the data-parallel
                      router that fronts N single-GPU replicas.
``dminfr.reference``  the unoptimized reference implementation. Kept only as
                      the baseline every speedup in the README is measured
                      against -- it loops 64 experts in Python and is not
                      intended for deployment.
``dminfr.tuning``     the end-to-end-aware Triton autotuner.

The engine/reference split used to be ``model_update/`` vs ``src/``, which
also buried the production server inside the directory named after the slow
reference path.
"""
