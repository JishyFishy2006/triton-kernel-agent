# L9: Nsight Compute on Modal

Status: **works, with unlocked clocks** (checked 2026-10-29 on an H100 SXM, driver
610.57.04, Nsight Compute 2025.1.1). Probe: `scripts/l9_ncu_probe.py`.

| Check | Result |
| --- | --- |
| `RmProfilingAdminOnly` in `/proc/driver/nvidia/params` | `0`: performance counters are not admin-only |
| `ncu` with default settings | Fails: `Failed to lock GPU clock frequencies` (the container cannot set host clocks) |
| `ncu --clock-control none --metrics dram__throughput...,sm__throughput...` | Works: 31.7% DRAM, 72.6% SM on a PyTorch normal-sampling kernel |
| `ncu --clock-control none --section SpeedOfLight` | Works: full Speed of Light table (SM at 1.96 GHz boost) |

## Consequences for B10

- The main roofline numbers stay analytic: achieved bytes and FLOPs per second (from the task
  metadata and B9's timing) over H100 peak. They are cheap and need no special image.
- `ncu` is an opt-in diagnostic (`modal_ncu.py`), not part of every eval. It needs the CUDA
  toolkit image (`nvidia/cuda:12.8.1-devel`), replays each kernel ~10 times, and its numbers
  move with the boost clock, so it is for explaining a kernel, not for timing it.
