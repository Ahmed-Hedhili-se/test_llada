"""OOM split-and-retry in dminfr/serving/server.py::_generate_with_oom_retry.

The failure this exists to prevent is measured: on a 24 GB card,
`BATCH_MAX_SIZE=48` produced **0 of 96 requests succeeded**. `_run_batch`
catches `Exception` and marks every request in the batch failed, so a single
config step past the memory limit took down the entire load rather than
slowing it down.

Testing it by actually exhausting a GPU is unreliable -- whether a given batch
OOMs depends on the card, and on a 48 GB A6000 the config that broke a 24 GB
card simply fits (verified: 96/96 succeeded, zero retries, so the retry path
was never entered). So the OOM is injected instead, which also makes the exact
halving sequence observable rather than inferred.

No GPU required.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeOOM(torch.cuda.OutOfMemoryError):
    """torch.cuda.OutOfMemoryError is constructible without a CUDA context."""


def _import_retry():
    """Import the helper without pulling in the whole server at import time."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_srv_for_test", os.path.join(root, "src", "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_srv_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(n, offset=0):
    """Distinguishable rows, so a wrong concatenation order is visible."""
    return torch.arange(offset, offset + n).unsqueeze(1).repeat(1, 4)


def test_no_oom_runs_whole_batch():
    srv = _import_retry()
    calls = []

    def gen(model, ids, **kw):
        calls.append(ids.shape[0])
        return _rows(ids.shape[0])

    out = srv._generate_with_oom_retry(gen, _rows(16), {}, 16)
    assert calls == [16], f"expected one full-batch call, got {calls}"
    assert out.shape[0] == 16
    print(f"  [ok] no OOM -> single call of 16, no splitting")


def test_halves_until_it_fits():
    """OOM above 4 -> should try 16, 8, then succeed at 4 (4 chunks)."""
    srv = _import_retry()
    calls = []

    def gen(model, ids, **kw):
        n = ids.shape[0]
        calls.append(n)
        if n > 4:
            raise _FakeOOM("injected")
        return ids.clone()

    out = srv._generate_with_oom_retry(gen, _rows(16), {}, 16)
    assert calls[0] == 16 and calls[1] == 8, f"expected 16 then 8 first, got {calls}"
    assert calls[2:] == [4, 4, 4, 4], f"expected four chunks of 4, got {calls[2:]}"
    assert out.shape[0] == 16, f"lost rows: {out.shape}"
    assert torch.equal(out, _rows(16)), "rows came back reordered or corrupted"
    print(f"  [ok] halved 16 -> 8 -> 4, all 16 rows returned in order  (calls: {calls})")


def test_rows_are_preserved_exactly():
    """Splitting must be a no-op on content: same rows, same order."""
    srv = _import_retry()

    def gen(model, ids, **kw):
        if ids.shape[0] > 3:
            raise _FakeOOM("injected")
        return ids * 10

    src = _rows(12)
    out = srv._generate_with_oom_retry(gen, src, {}, 12)
    assert torch.equal(out, src * 10), "sub-batch results do not reassemble to the whole"
    print("  [ok] split output reassembles identically to an unsplit run")


def test_gives_up_at_one_sequence():
    """If a single sequence will not fit, the caller must see the real error
    rather than an infinite halving loop."""
    srv = _import_retry()
    calls = []

    def gen(model, ids, **kw):
        calls.append(ids.shape[0])
        raise _FakeOOM("injected")

    try:
        srv._generate_with_oom_retry(gen, _rows(8), {}, 8)
    except torch.cuda.OutOfMemoryError:
        assert calls[-1] == 1, f"should have bottomed out at chunk=1, got {calls}"
        print(f"  [ok] raises once chunk reaches 1  (tried: {calls})")
    else:
        raise AssertionError("expected OutOfMemoryError to propagate")


def test_non_oom_errors_are_not_retried():
    """A real bug must surface immediately, not be masked by retrying smaller."""
    srv = _import_retry()
    calls = []

    def gen(model, ids, **kw):
        calls.append(ids.shape[0])
        raise RuntimeError("shape mismatch, a genuine bug")

    try:
        srv._generate_with_oom_retry(gen, _rows(8), {}, 8)
    except RuntimeError as exc:
        assert "genuine bug" in str(exc)
        assert calls == [8], f"non-OOM error was retried: {calls}"
        print("  [ok] non-OOM exception propagates on the first call, no retry")
    else:
        raise AssertionError("expected RuntimeError to propagate")


def main():
    print("OOM split-and-retry")
    for fn in (test_no_oom_runs_whole_batch,
               test_halves_until_it_fits,
               test_rows_are_preserved_exactly,
               test_gives_up_at_one_sequence,
               test_non_oom_errors_are_not_retried):
        fn()
    print("\nPASS - a batch that does not fit degrades to sub-batches instead of "
          "failing every request.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
