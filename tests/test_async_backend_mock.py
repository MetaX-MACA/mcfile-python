"""
Mock-based tests for the mcfile async_backend module.

These tests use mocked mcFile library functions to verify correct API usage
without requiring actual GPU hardware or the mcFile library. They mirror the
style of ``tests/test_cufile_mock.py``.

Covered surface:
- driver lifecycle (``_ensure_driver_open`` / ``close_driver``)
- handle / buffer / stream registration
- :class:`Submission` ctypes storage
- :class:`AsyncHandle` open/close, context manager, ``read_async`` /
  ``write_async`` (both the async path and the ``async_supported=False``
  fallback)
- error propagation through ``_ck``
"""

import os
import ctypes

import pytest
import torch  # imported BEFORE patching ctypes.CDLL so that async_backend's
             # module-level ``import torch`` is a plain sys.modules lookup while
             # the CDLL patch is active (torch must not be loaded through the
             # mock CDLL).
from unittest.mock import MagicMock, patch

# Mock the library loading before importing the mcfile module. ``bindings``
# dlopens ``libmcruntime.so`` and ``libmcfile.so`` at import time; without the
# real libs present we point both CDLL calls at a single MagicMock. Because a
# MagicMock auto-creates attributes, ``bindings._declare_async_signatures``
# sees ``mcFileReadAsync`` (via ``get_lib_func``) and flips ``async_supported``
# to True, so the async code path is exercised by default; the sync fallback is
# covered by patching ``mcfile.async_backend.async_supported`` to False.
mock_libmcfile = MagicMock()

with patch("ctypes.CDLL", return_value=mock_libmcfile):
    from mcfile import async_backend as async_backend_mod
    from mcfile.async_backend import (
        AsyncHandle,
        Submission,
        register_handle,
        deregister_handle,
        register_buffer,
        deregister_buffer,
        register_stream,
        deregister_stream,
        _ensure_driver_open,
        close_driver,
    )
    from mcfile.bindings import MCfileError, MCfileHandle_t


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _cuda_tensor_mock(base=0x1000, numel=256, esize=4):
    """Build a torch.Tensor-like fake that reports as a CUDA tensor."""
    buf = MagicMock()
    buf.is_cuda = True
    buf.numel.return_value = numel
    buf.element_size.return_value = esize
    buf.data_ptr.return_value = base
    return buf


@pytest.fixture(autouse=True)
def reset_driver_state():
    """Reset the module-level ``_driver_opened`` flag between tests."""
    async_backend_mod._driver_opened = False
    yield
    async_backend_mod._driver_opened = False


@pytest.fixture
def mock_lib():
    """Fresh mock for libmcfile, patched into both bindings and async_backend.

    ``async_backend`` calls some mcFile symbols directly through its own
    ``libmcfile`` binding (register_buffer / register_stream / read_async /
    write_async) and others through ``mcfile.bindings`` wrappers
    (mcFileDriverOpen / mcFileHandleRegister / mcFileHandleDeregister), so both
    references must point at the same mock object.
    """
    new = MagicMock()
    new.mcFileDriverOpen.return_value = MCfileError(err=0)
    new.mcFileDriverClose.return_value = MCfileError(err=0)
    new.mcFileHandleRegister.return_value = MCfileError(err=0)
    new.mcFileHandleDeregister.return_value = None
    new.mcFileBufRegister.return_value = MCfileError(err=0)
    new.mcFileBufDeregister.return_value = MCfileError(err=0)
    new.mcFileStreamRegister.return_value = MCfileError(err=0)
    new.mcFileStreamDeregister.return_value = MCfileError(err=0)
    new.mcFileReadAsync.return_value = MCfileError(err=0)
    new.mcFileWriteAsync.return_value = MCfileError(err=0)
    # Sync fallback: mcFileRead / mcFileWrite return the byte count directly.
    new.mcFileRead.return_value = 1024
    new.mcFileWrite.return_value = 1024
    with patch("mcfile.bindings.libmcfile", new), patch(
        "mcfile.async_backend.libmcfile", new
    ):
        yield new


@pytest.fixture
def mock_os():
    """Mock os.open / os.close / os.posix_fallocate as used by AsyncHandle."""
    with (
        patch("mcfile.async_backend.os.open", return_value=42) as mock_open,
        patch("mcfile.async_backend.os.close") as mock_close,
        patch("mcfile.async_backend.os.posix_fallocate") as mock_fallocate,
    ):
        yield {"open": mock_open, "close": mock_close, "fallocate": mock_fallocate}


# ---------------------------------------------------------------------------
# Driver lifecycle
# ---------------------------------------------------------------------------

class TestDriverLifecycle:
    """Test _ensure_driver_open / close_driver."""

    def test_ensure_driver_open_calls_mcFileDriverOpen(self, mock_lib):
        _ensure_driver_open()
        mock_lib.mcFileDriverOpen.assert_called_once()
        assert async_backend_mod._driver_opened is True

    def test_ensure_driver_open_is_idempotent(self, mock_lib):
        _ensure_driver_open()
        _ensure_driver_open()
        mock_lib.mcFileDriverOpen.assert_called_once()

    def test_close_driver_calls_mcFileDriverClose(self, mock_lib):
        _ensure_driver_open()
        close_driver()
        mock_lib.mcFileDriverClose.assert_called_once()
        assert async_backend_mod._driver_opened is False

    def test_close_driver_noop_when_not_opened(self, mock_lib):
        close_driver()
        mock_lib.mcFileDriverClose.assert_not_called()

    def test_close_driver_resets_state_for_reopen(self, mock_lib):
        _ensure_driver_open()
        close_driver()
        _ensure_driver_open()
        assert mock_lib.mcFileDriverOpen.call_count == 2


# ---------------------------------------------------------------------------
# Handle registration
# ---------------------------------------------------------------------------

class TestHandleRegistration:
    """Test register_handle / deregister_handle."""

    def test_register_handle_opens_driver(self, mock_lib):
        register_handle(42)
        mock_lib.mcFileDriverOpen.assert_called_once()
        mock_lib.mcFileHandleRegister.assert_called_once()

    def test_register_handle_passes_fd_in_descr(self, mock_lib):
        register_handle(42)
        # bindings.mcFileHandleRegister builds MCfileDescr(type=1, fd=42) and
        # passes it next to the handle pointer.
        call = mock_lib.mcFileHandleRegister.call_args[0]
        descr = call[1]
        assert descr.type == 1
        assert descr.handle.fd == 42

    def test_register_handle_returns_handle(self, mock_lib):
        handle = register_handle(42)
        assert isinstance(handle, MCfileHandle_t)

    def test_register_handle_skips_driver_when_already_opened(self, mock_lib):
        _ensure_driver_open()
        mock_lib.mcFileDriverOpen.reset_mock()
        register_handle(42)
        mock_lib.mcFileDriverOpen.assert_not_called()
        mock_lib.mcFileHandleRegister.assert_called_once()

    def test_deregister_handle_calls_mcFileHandleDeregister(self, mock_lib):
        handle = MCfileHandle_t(12345)
        deregister_handle(handle)
        mock_lib.mcFileHandleDeregister.assert_called_once_with(handle)


# ---------------------------------------------------------------------------
# Buffer registration
# ---------------------------------------------------------------------------

class TestBufferRegistration:
    """Test register_buffer / deregister_buffer."""

    def test_register_buffer_rejects_non_cuda_tensor(self, mock_lib):
        buf = torch.empty(16, dtype=torch.uint8)  # CPU tensor
        with pytest.raises(ValueError, match="tensor must be on CUDA"):
            register_buffer(buf)
        mock_lib.mcFileBufRegister.assert_not_called()

    def test_register_buffer_calls_mcFileBufRegister(self, mock_lib):
        buf = _cuda_tensor_mock(base=0x1000, numel=256, esize=4)  # 1024 bytes
        register_buffer(buf)
        mock_lib.mcFileBufRegister.assert_called_once()
        call = mock_lib.mcFileBufRegister.call_args[0]
        assert call[0].value == 0x1000   # c_void_p(buf.data_ptr())
        assert call[1].value == 1024     # c_size_t(numel * element_size)
        assert call[2].value == 0        # c_int(0) flags

    def test_register_buffer_does_not_open_driver(self, mock_lib):
        # register_buffer talks to libmcfile directly and does not call
        # _ensure_driver_open (the handle is expected to be registered by the
        # caller beforehand).
        buf = _cuda_tensor_mock()
        register_buffer(buf)
        mock_lib.mcFileDriverOpen.assert_not_called()

    def test_deregister_buffer_calls_mcFileBufDeregister(self, mock_lib):
        buf = _cuda_tensor_mock(base=0x1000)
        deregister_buffer(buf)
        mock_lib.mcFileBufDeregister.assert_called_once()
        call = mock_lib.mcFileBufDeregister.call_args[0]
        assert call[0].value == 0x1000


# ---------------------------------------------------------------------------
# Stream registration
# ---------------------------------------------------------------------------

class TestStreamRegistration:
    """Test register_stream / deregister_stream."""

    def test_register_stream_opens_driver(self, mock_lib):
        register_stream(0x2000)
        mock_lib.mcFileDriverOpen.assert_called_once()
        mock_lib.mcFileStreamRegister.assert_called_once()

    def test_register_stream_passes_fixed_flags(self, mock_lib):
        register_stream(0x2000)
        call = mock_lib.mcFileStreamRegister.call_args[0]
        assert call[0].value == 0x2000   # c_void_p(raw_stream)
        assert call[1] == 0x7           # _STREAM_REGISTER_FLAGS

    def test_deregister_stream_calls_mcFileStreamDeregister(self, mock_lib):
        deregister_stream(0x2000)
        call = mock_lib.mcFileStreamDeregister.call_args[0]
        assert call[0].value == 0x2000


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

class TestSubmission:
    """Test the Submission ctypes-storage holder."""

    def test_initial_values(self):
        s = Submission(size=512, file_offset=100, buf_offset=200)
        assert s._size.value == 512
        assert s._file_offset.value == 100
        assert s._buf_offset.value == 200
        assert s._bytes_done.value == 0
        assert s.bytes_done == 0

    def test_bytes_done_reflects_internal_storage(self):
        # mcFile writes the actual byte count into _bytes_done at stream
        # execution time; the property must read that same field back.
        s = Submission(size=10, file_offset=0, buf_offset=0)
        s._bytes_done.value = 2048
        assert s.bytes_done == 2048

    def test_uses_slots(self):
        s = Submission(size=1, file_offset=0, buf_offset=0)
        with pytest.raises(AttributeError):
            s.not_a_slot = 1


# ---------------------------------------------------------------------------
# AsyncHandle initialization
# ---------------------------------------------------------------------------

class TestAsyncHandleInitialization:
    """Test AsyncHandle.__init__ / from_fd / fd."""

    def test_readable_uses_O_RDONLY_and_O_DIRECT(self, mock_lib, mock_os):
        h = AsyncHandle("/tmp/x.bin", writable=False)
        flags = mock_os["open"].call_args[0][1]
        # O_RDONLY == 0 on Linux, so compare against the exact expected mask
        # rather than using a bitwise AND for the read-only flag.
        assert flags == os.O_DIRECT | os.O_RDONLY
        assert not (flags & os.O_RDWR)
        assert h.fd == 42
        assert h.path == "/tmp/x.bin"
        assert h.writable is False

    def test_writable_uses_O_CREAT_and_O_RDWR(self, mock_lib, mock_os):
        h = AsyncHandle("/tmp/x.bin", writable=True)
        flags = mock_os["open"].call_args[0][1]
        assert flags & os.O_DIRECT
        assert flags & os.O_CREAT
        assert flags & os.O_RDWR
        assert h.writable is True

    def test_registers_handle_on_init(self, mock_lib, mock_os):
        AsyncHandle("/tmp/x.bin", writable=False)
        mock_lib.mcFileHandleRegister.assert_called_once()

    def test_fallocate_called_when_writable(self, mock_lib, mock_os):
        AsyncHandle("/tmp/x.bin", writable=True, fallocate_size=4096)
        mock_os["fallocate"].assert_called_once_with(42, 0, 4096)

    def test_fallocate_skipped_when_readonly(self, mock_lib, mock_os):
        AsyncHandle("/tmp/x.bin", writable=False, fallocate_size=4096)
        mock_os["fallocate"].assert_not_called()

    def test_init_closes_fd_on_register_failure(self, mock_lib, mock_os):
        mock_lib.mcFileHandleRegister.return_value = MCfileError(err=-1)
        with pytest.raises(RuntimeError, match="mcFileHandleRegister failed"):
            AsyncHandle("/tmp/x.bin", writable=False)
        mock_os["close"].assert_called_once_with(42)

    def test_from_fd_does_not_open_or_register(self, mock_lib, mock_os):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y", writable=True)
        assert h.fd == 99
        assert h.path == "/tmp/y"
        assert h.writable is True
        mock_os["open"].assert_not_called()
        mock_lib.mcFileHandleRegister.assert_not_called()

    def test_fd_property(self, mock_lib, mock_os):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=7, handle=handle, path="/tmp/y")
        assert h.fd == 7


# ---------------------------------------------------------------------------
# AsyncHandle read / write
# ---------------------------------------------------------------------------

class TestAsyncHandleReadWrite:
    """Test read_async / write_async on both async and sync-fallback paths."""

    @pytest.fixture
    def handle(self):
        return MagicMock()

    @pytest.fixture
    def async_file(self, handle):
        return AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y", writable=True)

    def test_read_async_async_path(self, mock_lib, async_file, handle):
        sub = async_file.read_async(
            buf_base=0x1000, size=512, file_offset=100, buf_offset=200,
            raw_stream=0x3000,
        )
        mock_lib.mcFileReadAsync.assert_called_once()
        mock_lib.mcFileRead.assert_not_called()
        call = mock_lib.mcFileReadAsync.call_args[0]
        assert call[0] is handle              # self._handle
        assert call[1].value == 0x1000         # c_void_p(buf_base)
        assert call[6].value == 0x3000         # c_void_p(raw_stream)
        assert isinstance(sub, Submission)
        assert sub._size.value == 512
        assert sub._file_offset.value == 100
        assert sub._buf_offset.value == 200
        assert sub._bytes_done.value == 0
        assert sub.bytes_done == 0

    def test_read_async_sync_fallback(self, mock_lib, async_file, handle):
        # The sync fallback passes plain values (no byref pointers, no
        # bytes_done_p, no stream) and populates bytes_done from the return.
        with patch("mcfile.async_backend.async_supported", False):
            sub = async_file.read_async(
                buf_base=0x1000, size=512, file_offset=100, buf_offset=200,
                raw_stream=0x3000,
            )
        mock_lib.mcFileRead.assert_called_once()
        mock_lib.mcFileReadAsync.assert_not_called()
        call = mock_lib.mcFileRead.call_args[0]
        assert call[0] is handle              # handle
        assert call[1].value == 0x1000         # c_void_p(buf_base)
        assert call[2] == 512                 # size (plain)
        assert call[3] == 100                  # file_offset (plain)
        assert call[4] == 200                  # buf_offset (plain)
        assert len(call) == 5                 # no bytes_done, no stream
        assert isinstance(sub, Submission)
        assert sub.bytes_done == 1024          # populated from mcFileRead return

    def test_write_async_async_path(self, mock_lib, async_file, handle):
        sub = async_file.write_async(
            buf_base=0x2000, size=1024, file_offset=200, buf_offset=300,
            raw_stream=0x4000,
        )
        mock_lib.mcFileWriteAsync.assert_called_once()
        mock_lib.mcFileWrite.assert_not_called()
        call = mock_lib.mcFileWriteAsync.call_args[0]
        assert call[0] is handle
        assert call[1].value == 0x2000
        assert call[6].value == 0x4000
        assert isinstance(sub, Submission)
        assert sub._size.value == 1024
        assert sub._file_offset.value == 200
        assert sub._buf_offset.value == 300

    def test_write_async_sync_fallback(self, mock_lib, async_file, handle):
        with patch("mcfile.async_backend.async_supported", False):
            sub = async_file.write_async(
                buf_base=0x2000, size=1024, file_offset=200, buf_offset=300,
                raw_stream=0x4000,
            )
        mock_lib.mcFileWrite.assert_called_once()
        mock_lib.mcFileWriteAsync.assert_not_called()
        call = mock_lib.mcFileWrite.call_args[0]
        assert call[0] is handle
        assert call[1].value == 0x2000
        assert call[2] == 1024
        assert call[3] == 200
        assert call[4] == 300
        assert len(call) == 5
        assert isinstance(sub, Submission)
        assert sub.bytes_done == 1024

    def test_read_async_returns_distinct_submissions(self, mock_lib, async_file):
        s1 = async_file.read_async(0x1000, 10, 0, 0, 0x3000)
        s2 = async_file.read_async(0x1000, 20, 0, 0, 0x3000)
        assert s1 is not s2
        assert s1._size.value == 10
        assert s2._size.value == 20
        assert mock_lib.mcFileReadAsync.call_count == 2


# ---------------------------------------------------------------------------
# AsyncHandle close / context manager
# ---------------------------------------------------------------------------

class TestAsyncHandleCloseAndContextManager:
    """Test AsyncHandle.close / __enter__ / __exit__."""

    def test_close_deregisters_and_closes_fd(self, mock_lib, mock_os):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        h.close()
        mock_lib.mcFileHandleDeregister.assert_called_once_with(handle)
        mock_os["close"].assert_called_once_with(99)

    def test_close_is_idempotent(self, mock_lib, mock_os):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        h.close()
        h.close()
        mock_os["close"].assert_called_once_with(99)

    def test_context_manager_closes_on_exit(self, mock_lib, mock_os):
        with AsyncHandle("/tmp/x.bin", writable=True) as h:
            assert h.fd == 42
            mock_lib.mcFileHandleRegister.assert_called_once()
        mock_lib.mcFileHandleDeregister.assert_called_once()
        mock_os["close"].assert_called_once_with(42)

    def test_context_manager_returns_self(self, mock_lib, mock_os):
        h = AsyncHandle("/tmp/x.bin", writable=True)
        with h as ctx:
            assert ctx is h

    def test_context_manager_closes_on_exception(self, mock_lib, mock_os):
        with pytest.raises(ValueError):
            with AsyncHandle("/tmp/x.bin", writable=True):
                raise ValueError("boom")
        mock_lib.mcFileHandleDeregister.assert_called_once()
        mock_os["close"].assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test _ck error propagation through the async_backend entry points."""

    def test_register_buffer_raises_on_lib_error(self, mock_lib):
        buf = _cuda_tensor_mock()
        mock_lib.mcFileBufRegister.return_value = MCfileError(err=-3)
        with pytest.raises(RuntimeError, match="mcFileBufRegister failed.*err=-3"):
            register_buffer(buf)

    def test_register_stream_raises_on_lib_error(self, mock_lib):
        mock_lib.mcFileStreamRegister.return_value = MCfileError(err=-5)
        with pytest.raises(RuntimeError, match="mcFileStreamRegister failed.*err=-5"):
            register_stream(0x2000)

    def test_read_async_raises_on_lib_error(self, mock_lib):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        mock_lib.mcFileReadAsync.return_value = MCfileError(err=-7)
        with pytest.raises(RuntimeError, match="mcFileReadAsync failed.*err=-7"):
            h.read_async(0x1000, 10, 0, 0, 0x3000)

    def test_write_async_raises_on_lib_error(self, mock_lib):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        mock_lib.mcFileWriteAsync.return_value = MCfileError(err=-9)
        with pytest.raises(RuntimeError, match="mcFileWriteAsync failed.*err=-9"):
            h.write_async(0x1000, 10, 0, 0, 0x3000)

    def test_read_async_sync_fallback_raises_on_negative_return(self, mock_lib):
        # The sync fallback raises when mcFileRead returns a negative byte
        # count (the sync API signals errors via a negative return).
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        mock_lib.mcFileRead.return_value = -7
        with patch("mcfile.async_backend.async_supported", False):
            with pytest.raises(RuntimeError, match="mcFileRead failed.*err=-7"):
                h.read_async(0x1000, 10, 0, 0, 0x3000)

    def test_write_async_sync_fallback_raises_on_negative_return(self, mock_lib):
        handle = MagicMock()
        h = AsyncHandle.from_fd(fd=99, handle=handle, path="/tmp/y")
        mock_lib.mcFileWrite.return_value = -9
        with patch("mcfile.async_backend.async_supported", False):
            with pytest.raises(RuntimeError, match="mcFileWrite failed.*err=-9"):
                h.write_async(0x1000, 10, 0, 0, 0x3000)
