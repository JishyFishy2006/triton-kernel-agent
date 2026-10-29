"""Roofline numbers (B10): achieved bandwidth and FLOP rate as a percentage of GPU peak.

Analytic: the task's bytes (inputs + parameters + outputs, each once) and FLOPs per forward
call (tasks/metadata.json, measured by B4) divided by the measured kernel time (B9). Nsight
Compute metrics are an optional extra (modal_ncu.py; see decisions/ncu_on_modal.md).

Strict fp32 (Y2 decision 1) means the FLOP peak is the CUDA-core fp32 peak, not TF32 tensor
cores: tl.dot with input_precision="ieee" runs on the FMA pipes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuPeak:
    name: str
    bandwidth_bytes_per_s: float
    fp32_flops_per_s: float


# NVIDIA H100 datasheet (dense, no sparsity). Modal's "H100!" is the SXM part.
H100_SXM = GpuPeak("H100 SXM", bandwidth_bytes_per_s=3.35e12, fp32_flops_per_s=67e12)
H100_PCIE = GpuPeak("H100 PCIe", bandwidth_bytes_per_s=2.0e12, fp32_flops_per_s=51e12)


def peak_for(device_name: str) -> GpuPeak:
    """Peak numbers for a torch.cuda.get_device_name() string."""
    if "H100" in device_name:
        return H100_PCIE if "PCIe" in device_name else H100_SXM
    raise ValueError(f"no peak numbers for {device_name!r}; add them to roofline.py")


@dataclass(frozen=True)
class Roofline:
    achieved_bandwidth: float  # bytes/s
    achieved_flops: float  # FLOP/s
    pct_peak_bandwidth: float
    pct_peak_flops: float | None  # None when the task has no counted FLOPs
    arithmetic_intensity: float | None  # FLOPs per byte
    bound: str  # "memory" or "compute": which roof the task sits under

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def roofline(nbytes: int, flops: int, kernel_ms: float, peak: GpuPeak = H100_SXM) -> Roofline:
    if kernel_ms <= 0:
        raise ValueError(f"kernel_ms must be positive, got {kernel_ms}")
    seconds = kernel_ms / 1e3
    bandwidth = nbytes / seconds
    rate = flops / seconds
    intensity = flops / nbytes if nbytes else None
    # Ridge point: the intensity where the two roofs meet (~20 FLOP/byte on an H100 in fp32).
    ridge = peak.fp32_flops_per_s / peak.bandwidth_bytes_per_s
    return Roofline(
        achieved_bandwidth=bandwidth,
        achieved_flops=rate,
        pct_peak_bandwidth=100 * bandwidth / peak.bandwidth_bytes_per_s,
        pct_peak_flops=100 * rate / peak.fp32_flops_per_s if flops else None,
        arithmetic_intensity=intensity,
        bound="compute" if intensity is not None and intensity > ridge else "memory",
    )
