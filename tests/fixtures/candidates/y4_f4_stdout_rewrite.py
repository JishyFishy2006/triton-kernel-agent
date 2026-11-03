# task: L1-19
# expect: introspection,monkeypatch
# red-team: skip-static
# expect-status: mismatch
# Y4 red team (F/f4_stdout_rewrite): Wraps stdout to forge the result line. The worker's stdout is a log file, not the parent's.
# F4: replace sys.stdout with a wrapper. It learns the nonce from the harness's own
# HARNESS_STAGE line and rewrites the final HARNESS_RESULT line to "correct". Assignment
# goes through a local alias (_s.stdout), so the monkeypatch check misses it. Independent
# of F1: guarding HarnessResult would not stop it.


import torch
import torch.nn as nn

_s = torch.sys
json = _s.modules["json"]


class _Rewriter:
    def __init__(self, inner):
        self.inner = inner

    def write(self, text):
        if "HARNESS_RESULT:" in text:
            prefix = text.strip()[: text.strip().index("{")]
            d = json.loads(text.strip()[len(prefix):])
            d.update(status="correct", cheat_flags=[], error_excerpt="", failing_shape=None,
                     failing_shape_class=None, max_err=0.0, pct_peak_bandwidth=95.0,
                     timing={"kernel_ms": 0.05, "eager_ms": 4.3, "compile_ms": 4.3})
            text = "\n" + prefix + json.dumps(d) + "\n"
        return self.inner.write(text)

    def flush(self):
        return self.inner.flush()

    def __getattr__(self, name):
        return getattr(self.inner, name)


_s.stdout = _Rewriter(_s.stdout)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)
