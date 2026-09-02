# License-Identifier: Apache-2.0
"""Minimal ctypes wrapper around the mcFile async C API.

kvikio's ``raw_read_async`` / ``raw_write_async`` work but on ext4 +
real GDS they leave ~60% read throughput on the table compared to the
bare C path (``mcFileReadAsync`` + ``mcFileStreamRegister`` + batched
submit + single ``cudaStreamSynchronize``). This module exposes the
same C primitives directly from Python so callers that batch
submissions can match the C-direct throughput.

Surface:

- :func:`register_buffer` / :func:`deregister_buffer` — wrap
  ``mcFileBufRegister`` / ``mcFileBufDeregister`` on a torch tensor.
- :func:`register_stream` / :func:`deregister_stream` — wrap
  ``mcFileStreamRegister`` / ``mcFileStreamDeregister`` on a raw
  CUDA stream handle.
- :class:`AsyncHandle` — opens a file with ``O_DIRECT`` (required by
  mcFile on ext4) and registers the mcFile handle. ``read_async`` /
  ``write_async`` enqueue an async IO on a stream and return a
  :class:`Submission`. Callers run ``cudaStreamSynchronize`` once to
  drain a batch; :meth:`Submission.bytes_done` returns the actual
  byte count after the sync.

This module is intentionally narrow: no thread pool, no future
abstraction, no LRU. It is the layer :class:`GDSContext`
(``lmcache.v1.gpu_connector.gds_context``) uses to talk to libmcfile on
the GDS DMA fast path.
"""

# Standard
from typing import TYPE_CHECKING, Any, Optional
import ctypes
import os

# Third Party
import torch

from . import bindings
from .bindings import libmcfile, _ck, async_supported

# ``mcfile.bindings`` dlopens ``libmcfile.so`` at import time, which is absent
# on CPU-only / macOS hosts. Importing this module (transitively pulled in by
# the CLI command discovery via ``storage_manager``) must not trigger that, so
# every mcfile symbol is imported lazily inside the function that uses it and
# the dlopen happens only when GDS is actually exercised. This mirrors the
# lazy ``import mcfile`` in the legacy ``GdsBackend``.


_driver_opened = False

def _ensure_driver_open() -> None:
    """Idempotently open the mcFile driver."""
    global _driver_opened
    if _driver_opened:
        return

    bindings.mcFileDriverOpen()
    _driver_opened = True


def close_driver() -> None:
    """Close the mcFile driver. Optional — useful in tests."""
    global _driver_opened
    if not _driver_opened:
        return

    try:
        bindings.mcFileDriverClose()
    finally:
        _driver_opened = False


# --- Handle registration -------------------------------------------

def register_handle(fd: int) -> Any:
    """Register an open fd with mcFile and return the ``CUfileHandle_t``.

    Opens the mcFile driver on first use. The returned handle is accepted
    directly as the first argument of ``mcFileReadAsync`` / ``mcFileWriteAsync``.
    """
    _ensure_driver_open()
    return bindings.mcFileHandleRegister(fd)


def deregister_handle(handle: Any) -> None:
    """Reverse of :func:`register_handle` (``mcFileHandleDeregister``)."""
    bindings.mcFileHandleDeregister(handle)


# --- Buffer / stream registration ----------------------------------

def register_buffer(buf: torch.Tensor) -> None:
    """Register a device tensor with mcFile for GDS DMA.

    Must be called before any ``read_async`` / ``write_async`` whose
    ``buf_base`` falls inside this tensor's allocation. Implicitly
    opens the mcFile driver on first use.

    Uses ``libmcfile.mcFileBufRegister`` directly (not the
    ``mcfile.bindings`` wrapper) because the wrapper hides the error
    code by raising internally — we want the raw status so callers
    see ``mcFileError(err=…, cu_err=…)`` instead of a Python re-raise.
    """
    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on CUDA")

    nbytes = buf.numel() * buf.element_size()
    _ck(
        libmcfile.mcFileBufRegister(
            ctypes.c_void_p(buf.data_ptr()),
            ctypes.c_size_t(nbytes),
            ctypes.c_int(0),
        ),
        "mcFileBufRegister",
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`."""
    _ck(
        libmcfile.mcFileBufDeregister(ctypes.c_void_p(buf.data_ptr())),
        "mcFileBufDeregister",
    )


# mcFileStreamRegister flags (mcfile.h): declare the buffer offset, file offset,
# and size are all set at submission time (CU_FILE_STREAM_FIXED_* = 0x1|0x2|0x4).
# Worth ~12% higher read throughput vs 0x0 in our benchmark (write unchanged).
# PAGE_ALIGNED_INPUTS (0x8) is omitted -- transfer sizes are not always 4 KiB.
_STREAM_REGISTER_FLAGS = 0x7


def register_stream(raw_stream: int) -> None:
    """Register a CUDA stream with mcFile.

    ``raw_stream`` is the integer ``CUstream`` handle — get it via
    ``torch_dev.current_stream().cuda_stream``.

    Optional for correctness (``read_async`` / ``write_async`` also take the
    stream per call). We register with the FIXED_* flags (0x7): mcFile still
    reads the size/offset pointers at stream-execution time -- so their storage
    must stay alive and unchanged until completion (see ``Submission``) -- but
    promising the values are fixed at submission lets mcFile skip per-op setup,
    worth ~12% higher read throughput in our benchmark.
    """
    _ensure_driver_open()
    if not async_supported:
        return
    _ck(
        libmcfile.mcFileStreamRegister(
            ctypes.c_void_p(raw_stream), _STREAM_REGISTER_FLAGS
        ),
        "mcFileStreamRegister",
    )


def deregister_stream(raw_stream: int) -> None:
    """Reverse of :func:`register_stream`."""
    if not async_supported:
        return
    _ck(
        libmcfile.mcFileStreamDeregister(ctypes.c_void_p(raw_stream)),
        "mcFileStreamDeregister",
    )


# --- AsyncHandle + Submission --------------------------------------

class Submission:
    """One in-flight ``mcFileReadAsync`` / ``mcFileWriteAsync``.

    Holds the host-side ``size_p`` / ``file_offset_p`` /
    ``bufPtr_offset_p`` / ``bytes_done_p`` storage that mcFile writes
    into asynchronously. These ctypes objects MUST stay alive until
    the stream actually executes the op — keep the :class:`Submission`
    reference (or stash it in a list) until after the stream sync.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(
        self,
        size: int,
        file_offset: int,
        buf_offset: int,
    ) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        """Bytes actually transferred. Valid only AFTER the stream sync."""
        return self._bytes_done.value


class AsyncHandle:
    """Open file + mcFile handle wrapper.

    Opens with ``O_DIRECT`` (required for mcFile's GDS fast path on
    ext4). Optionally pre-allocates the file via ``posix_fallocate``.
    """

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        path: str,
        writable: bool = False,
        fallocate_size: Optional[int] = None,
        mode: int = 0o644,
    ) -> None:
        if writable:
            flags = os.O_CREAT | os.O_RDWR
        else:
            flags = os.O_RDONLY
        self.path = path
        self.writable = writable
        self._fd = os.open(path, flags, mode)
        try:
            if fallocate_size is not None and writable:
                os.posix_fallocate(self._fd, 0, fallocate_size)
        except Exception:
            os.close(self._fd)
            raise

        flags |= os.O_DIRECT
        self._fd = os.open(path, flags, mode)
        try:
            self._handle = register_handle(self._fd)
        except Exception:
            os.close(self._fd)
            raise

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: Any,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd and registered mcFile handle.

        For callers that open + register the file themselves (e.g. a slab that
        must be created, truncated, and ``posix_fallocate``d before
        ``mcFileHandleRegister``) and just need an ``AsyncHandle`` around the
        result.
        """
        obj = cls.__new__(cls)
        obj._fd = fd
        obj._handle = handle
        obj.path = path
        obj.writable = writable
        return obj

    @property
    def fd(self) -> int:
        return self._fd

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``mcFileReadAsync`` on the stream.

        ``buf_base`` is the registered base pointer (e.g.
        ``buf.data_ptr()``). ``buf_offset`` is the byte offset within
        that registration that the data should land at.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        if async_supported:
            _ck(
                libmcfile.mcFileReadAsync(
                    self._handle,
                    ctypes.c_void_p(buf_base),
                    ctypes.byref(sub._size),
                    ctypes.byref(sub._file_offset),
                    ctypes.byref(sub._buf_offset),
                    ctypes.byref(sub._bytes_done),
                    ctypes.c_void_p(raw_stream),
                ),
                "mcFileReadAsync",
            )
        else:
            # Sync fallback: mcFileRead takes plain values (no bytes_done_p /
            # stream) and returns the byte count directly, so populate the
            # Submission's bytes_done ourselves and raise on a negative
            # (error) return, mirroring the async path's _ck-on-API-error.
            ret = libmcfile.mcFileRead(
                self._handle,
                ctypes.c_void_p(buf_base),
                size,
                file_offset,
                buf_offset,
            )
            if ret < 0:
                raise RuntimeError(f"mcFileRead failed (mcFile err={ret})")
            sub._bytes_done.value = ret
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Enqueue a ``mcFileWriteAsync`` on the stream."""
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        if async_supported:
            _ck(
                libmcfile.mcFileWriteAsync(
                    self._handle,
                    ctypes.c_void_p(buf_base),
                    ctypes.byref(sub._size),
                    ctypes.byref(sub._file_offset),
                    ctypes.byref(sub._buf_offset),
                    ctypes.byref(sub._bytes_done),
                    ctypes.c_void_p(raw_stream),
                ),
                "mcFileWriteAsync",
            )
        else:
            ret = libmcfile.mcFileWrite(
                self._handle,
                ctypes.c_void_p(buf_base),
                size,
                file_offset,
                buf_offset,
            )
            if ret < 0:
                raise RuntimeError(f"mcFileWrite failed (mcFile err={ret})")
            sub._bytes_done.value = ret
        return sub

    def close(self) -> None:
        """Deregister the mcFile handle and close the fd."""
        if self._fd < 0:
            return
        try:
            deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "AsyncHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
