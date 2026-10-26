import pytest

from triton_kernel_agent.gpu_info import GpuInfo, parse_nvidia_smi

SMI_HEADER = """\
Mon Oct 26 19:40:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
"""


def test_parses_sxm_h100():
    info = parse_nvidia_smi(SMI_HEADER, "NVIDIA H100 80GB HBM3, 570.86.15, 81559 MiB\n")
    assert info == GpuInfo(
        name="NVIDIA H100 80GB HBM3",
        driver_version="570.86.15",
        cuda_version="12.8",
        memory_mib=81559,
        form_factor="SXM",
    )


# Captured from a Modal H100! container on 2026-10-26 (driver 610 renamed the field).
MODAL_SMI_HEADER = """\
Mon Oct 26 20:16:17 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.57.04              KMD Version: 610.57.04     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
"""


def test_parses_new_driver_header_from_modal():
    info = parse_nvidia_smi(MODAL_SMI_HEADER, "NVIDIA H100 80GB HBM3, 610.57.04, 81559 MiB\n")
    assert info.cuda_version == "13.3"
    assert info.driver_version == "610.57.04"
    assert info.form_factor == "SXM"


def test_parses_pcie_h100():
    info = parse_nvidia_smi(SMI_HEADER, "NVIDIA H100 PCIe, 570.86.15, 81559 MiB\n")
    assert info.form_factor == "PCIe"


def test_unknown_form_factor():
    info = parse_nvidia_smi(SMI_HEADER, "NVIDIA A10G, 570.86.15, 23028 MiB\n")
    assert info.form_factor == "unknown"


def test_missing_cuda_version_raises():
    with pytest.raises(ValueError, match="CUDA Version"):
        parse_nvidia_smi("no header here", "NVIDIA H100 80GB HBM3, 570.86.15, 81559 MiB\n")


def test_multiple_gpus_raises():
    two = "NVIDIA H100 80GB HBM3, 570.86.15, 81559 MiB\n" * 2
    with pytest.raises(ValueError, match="exactly one GPU"):
        parse_nvidia_smi(SMI_HEADER, two)
