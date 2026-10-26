"""Parse nvidia-smi output into a GpuInfo. Pure, so it is tested without a GPU."""

import re
from dataclasses import dataclass

# Query passed to nvidia-smi; parse_nvidia_smi expects its CSV columns in this order.
SMI_QUERY = ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]

_CUDA_RE = re.compile(r"CUDA Version:\s*([\d.]+)")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    driver_version: str
    # Highest CUDA version the driver supports, from the nvidia-smi header.
    cuda_version: str
    memory_mib: int
    form_factor: str  # "SXM", "PCIe" or "unknown"


def _form_factor(name: str) -> str:
    if "PCIe" in name:
        return "PCIe"
    # H100 SXM reports as "H100 80GB HBM3"; PCIe parts carry "PCIe" in the name.
    if "SXM" in name or "HBM3" in name:
        return "SXM"
    return "unknown"


def parse_nvidia_smi(header: str, query_csv: str) -> GpuInfo:
    """header: plain `nvidia-smi` output. query_csv: output of `nvidia-smi` with SMI_QUERY."""
    match = _CUDA_RE.search(header)
    if not match:
        raise ValueError("no 'CUDA Version' in nvidia-smi header")

    lines = [line for line in query_csv.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one GPU, got {len(lines)}")
    name, driver, memory = (field.strip() for field in lines[0].split(","))

    return GpuInfo(
        name=name,
        driver_version=driver,
        cuda_version=match.group(1),
        memory_mib=int(memory.removesuffix("MiB").strip()),
        form_factor=_form_factor(name),
    )
