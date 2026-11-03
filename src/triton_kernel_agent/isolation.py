"""The parent's side of the worker process (Y4 isolation): start it, talk to it, clean up.

The harness (trusted parent) never imports candidate code. It starts worker.py in a fresh
directory for each eval and process kind:
- as an unprivileged user in the sandbox (WORKER_UID), so it cannot read or write the harness,
  signal or trace the parent, or leave files for the next eval;
- with no inherited stdout: its prints land in a log file, never next to the result line;
- with its own HOME and Triton cache, deleted with the directory after the eval;
- killed with every process its user owns between phases, so nothing it started keeps running
  while the parent times the eager and torch.compile baselines.
Tensors cross as raw bytes in files (no pickle); the parent validates every header.
"""

import contextlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# `nobody` in the sandbox image. None (no user switch) where the parent is not root, e.g. tests.
NOBODY = 65534
WORKER_UID = NOBODY if hasattr(os, "geteuid") and os.geteuid() == 0 else None
WORKER_SOURCE = Path(__file__).with_name("worker.py")
LOG_TAIL_CHARS = 4000
# Answer within this long or the command counts as hung (loads, builds, single runs).
COMMAND_TIMEOUT_S = 120.0


class WorkerError(Exception):
    """The worker died, hung, or answered something that is not a reply."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "died", "timeout" or "protocol"


class Worker:
    """One worker process. Use as a context manager; stop() kills everything it started."""

    def __init__(self, root: Path, name: str, strict_fp32: bool = False):
        self.dir = root / name
        self.inbox = self.dir / "in"  # parent writes, worker reads
        self.outbox = self.dir / "out"  # worker writes, parent reads
        for d in (self.dir, self.inbox, self.outbox):
            d.mkdir(parents=True, exist_ok=True)
        # HOME and the Triton cache need a filesystem that allows executing the .so files
        # Triton builds; the sandbox mounts /dev/shm noexec. Deleted in stop().
        self.home = Path(tempfile.mkdtemp(prefix="tka_home_"))
        self.strict_fp32 = strict_fp32
        self.proc: subprocess.Popen | None = None
        self._buffer = b""

    def start(self) -> "Worker":
        script = self.dir / "worker.py"
        shutil.copyfile(WORKER_SOURCE, script)
        os.chmod(self.dir, 0o755)
        os.chmod(self.inbox, 0o755)
        os.chmod(self.home, 0o700)
        if WORKER_UID is not None:
            for d in (self.home, self.outbox):
                os.chown(d, WORKER_UID, WORKER_UID)
        self.log = (self.dir / "worker.log").open("wb")
        read_fd, write_fd = os.pipe()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "TRITON_CACHE_DIR": str(self.home / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(self.home / "inductor"),
            "TMPDIR": str(self.home),
        }
        for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
            if key in os.environ:
                env[key] = os.environ[key]
        args = [sys.executable, "-I", str(script), "--reply-fd", str(write_fd)]
        self.proc = subprocess.Popen(
            args + (["--strict-fp32"] if self.strict_fp32 else []),
            stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT,
            pass_fds=(write_fd,), cwd=self.home, env=env, start_new_session=True,
            user=WORKER_UID, group=WORKER_UID, extra_groups=[] if WORKER_UID else None,
        )  # fmt: skip
        os.close(write_fd)
        self.reply_fd = read_fd
        return self

    def call(self, command: dict[str, Any], timeout_s: float | None = None) -> dict:
        """Send one command and wait for its answer (a claim: the caller verifies it)."""
        timeout_s = COMMAND_TIMEOUT_S if timeout_s is None else timeout_s
        assert self.proc is not None and self.proc.stdin is not None
        try:
            self.proc.stdin.write((json.dumps(command) + "\n").encode())
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise self._died() from None
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise WorkerError("timeout", f"no answer within {timeout_s:g} s")
            ready, _, _ = select.select([self.reply_fd], [], [], min(left, 1.0))
            if ready:
                chunk = os.read(self.reply_fd, 1 << 16)
                if not chunk:
                    raise self._died()
                self._buffer += chunk
            elif self.proc.poll() is not None:
                raise self._died()
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            answer = json.loads(line)
        except ValueError:
            raise WorkerError("protocol", "the worker's answer was not JSON") from None
        if not isinstance(answer, dict) or not isinstance(answer.get("ok"), bool):
            raise WorkerError("protocol", "the worker's answer has no ok field")
        return answer

    def log_tail(self) -> str:
        self.log.flush()
        text = (self.dir / "worker.log").read_bytes()[-LOG_TAIL_CHARS:]
        return text.decode(errors="replace")

    def _died(self) -> WorkerError:
        code = self.proc.wait(timeout=10) if self.proc else None
        return WorkerError("died", f"worker exited with code {code}\n{self.log_tail()}")

    def clear_outbox(self) -> None:
        for f in self.outbox.iterdir():
            f.unlink()

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"cmd": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        kill_all(self.proc)
        if WORKER_UID is not None:  # nothing it wrote may reach the next worker
            clean_user(WORKER_UID)
        os.close(self.reply_fd)
        self.log.close()
        shutil.rmtree(self.home, ignore_errors=True)
        self.proc = None

    def __enter__(self) -> "Worker":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def kill_all(proc: subprocess.Popen) -> None:
    """Kill the worker, its process group, and (in the sandbox) every process of its user."""
    for target in (lambda: os.killpg(proc.pid, signal.SIGKILL), proc.kill):
        with contextlib.suppress(OSError):
            target()
    if WORKER_UID is not None:
        kill_user(WORKER_UID)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


def kill_user(uid: int) -> None:
    """SIGKILL every process owned by uid: a worker may setsid() or fork to escape its group."""
    for _ in range(3):  # processes can fork while we kill; a few sweeps settle it
        found = False
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                lines = status.read_text().splitlines()
            except OSError:
                continue
            uid_line = next((ln for ln in lines if ln.startswith("Uid:")), None)
            if uid_line is None:
                continue
            if int(uid_line.split()[1]) == uid:
                found = True
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(int(status.parent.name), signal.SIGKILL)
        if not found:
            return
        time.sleep(0.05)


def eval_dir() -> Path:
    """A fresh directory for one eval's workers (in /dev/shm when it exists: tensors are big)."""
    base = "/dev/shm" if Path("/dev/shm").is_dir() and os.access("/dev/shm", os.W_OK) else None
    path = Path(tempfile.mkdtemp(prefix="tka_eval_", dir=base))
    os.chmod(path, 0o755)
    return path


# World-writable places a worker could leave files in for a later worker or eval.
SCRATCH_DIRS = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock", "/dev/mqueue")


def clean_user(uid: int) -> None:
    """Kill every process of uid and delete every file it owns in SCRATCH_DIRS."""
    kill_user(uid)
    for base in map(Path, SCRATCH_DIRS):
        if not base.is_dir():
            continue
        for entry in base.iterdir():
            try:
                if entry.lstat().st_uid != uid:
                    continue
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
            except OSError:
                pass


def lock_down_harness() -> None:
    """In the sandbox, before an eval: make the harness package (under /root) unreadable to the
    worker user, and remove anything an earlier eval's workers left: processes (a harness killed
    at the wall-clock limit cannot clean up) and files (stashed hidden sizes, caches)."""
    if WORKER_UID is None:
        return
    clean_user(WORKER_UID)
    if Path("/root").is_dir():
        os.chmod("/root", 0o700)


# --- tensors across the process boundary (parent side; the worker has its own copy) ---

_DTYPE_NAMES = (
    "float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8", "uint8",
    "bool",
)  # fmt: skip


def write_inputs(inputs: list[Any], directory: Path, prefix: str = "in") -> list[dict]:
    """Inputs as file headers (tensors) or JSON values (ints, floats, tuples)."""
    import torch

    files = []
    for i, x in enumerate(inputs):
        if isinstance(x, torch.Tensor):
            files.append(_write_tensor(x, directory / f"{prefix}{i}.bin"))
        else:
            files.append({"value": encode(x)})
    for f in directory.iterdir():
        os.chmod(f, 0o644)
    return files


def encode(value: Any) -> Any:
    """A JSON form of init arguments and non-tensor inputs that keeps tuples (worker.decode)."""
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"a {type(value).__name__} cannot cross to the worker")


def _write_tensor(t, path: Path) -> dict:
    import torch

    flat = t.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
    path.write_bytes(b"")
    if flat.numel():
        storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=flat.numel())
        torch.empty(0, dtype=torch.uint8).set_(storage).copy_(flat)
    return {"file": path.name, "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}


class BadOutput(Exception):
    """A worker output header or file that does not describe a plain tensor."""


def output_meta(meta: Any) -> tuple[list[int], str]:
    """Validate one worker output header: (shape, dtype name)."""
    if not isinstance(meta, dict):
        raise BadOutput("output header is not an object")
    shape, dtype = meta.get("shape"), meta.get("dtype")
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise BadOutput("output shape is not a list of sizes")
    if dtype not in _DTYPE_NAMES:
        raise BadOutput(f"output dtype {dtype!r} is not supported")
    if not isinstance(meta.get("file"), str) or "/" in meta["file"] or meta["file"].startswith("."):
        raise BadOutput("output file name is not a plain name")
    return shape, dtype


def read_output(meta: dict, directory: Path, device: str):
    """Load one worker output as a plain torch.Tensor on `device`, checking its size."""
    import torch

    shape, dtype_name = output_meta(meta)
    dtype = getattr(torch, dtype_name)
    numel = 1
    for d in shape:
        numel *= d
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    path = directory / meta["file"]
    if not path.is_file() or path.is_symlink() or path.stat().st_size != nbytes:
        raise BadOutput("output file is missing or has the wrong size")
    if not nbytes:
        return torch.empty(shape, dtype=dtype, device=device)
    storage = torch.UntypedStorage.from_file(str(path), shared=False, nbytes=nbytes)
    flat = torch.empty(0, dtype=torch.uint8).set_(storage)
    return flat.view(dtype).reshape(shape).to(device).clone()
