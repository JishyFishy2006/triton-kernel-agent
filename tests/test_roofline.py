import pytest

from triton_kernel_agent.roofline import H100_PCIE, H100_SXM, peak_for, roofline
from triton_kernel_agent.tasks import get_task


def test_copy_at_peak_bandwidth_is_100_percent():
    # 3.35 GB moved in 1 ms on an H100 SXM.
    r = roofline(nbytes=int(3.35e9), flops=0, kernel_ms=1.0)
    assert r.pct_peak_bandwidth == pytest.approx(100)
    assert r.pct_peak_flops is None and r.bound == "memory"


def test_square_matmul_is_compute_bound():
    task = get_task("L1-1")  # 4096^3 matmul: 137 GFLOP over 0.2 GB
    r = roofline(task.bytes, task.flops, kernel_ms=2.67)  # eager time from the Y2 screen
    assert r.bound == "compute"
    assert r.arithmetic_intensity == pytest.approx(2 * 4096 / 12)
    assert r.pct_peak_flops == pytest.approx(100 * 2 * 4096**3 / 2.67e-3 / 67e12)
    assert 70 < r.pct_peak_flops < 80  # cuBLAS fp32 lands near 51 of 67 TFLOPS


def test_elementwise_tasks_are_memory_bound():
    task = get_task("L1-19")
    r = roofline(task.bytes, task.flops, kernel_ms=4.30)
    assert r.bound == "memory" and r.pct_peak_flops is None
    assert 85 < r.pct_peak_bandwidth < 95  # Y2: eager ReLU already ~90% of peak


def test_peaks_by_device_name():
    assert peak_for("NVIDIA H100 80GB HBM3") is H100_SXM
    assert peak_for("NVIDIA H100 PCIe") is H100_PCIE
    with pytest.raises(ValueError, match="A10G"):
        peak_for("NVIDIA A10G")
    with pytest.raises(ValueError):
        roofline(1, 1, kernel_ms=0)
