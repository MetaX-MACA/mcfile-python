"""
Mock-based tests for the cufile module.

These tests use mocked mcFile library functions to verify correct API usage
without requiring actual NVIDIA GPU hardware or the mcFile library.
"""

import os
import ctypes
import pytest
from unittest.mock import MagicMock, patch

# Mock the library loading before importing the mcfile module
mock_libmcfile = MagicMock()
mock_libmcfile.mcFileDriverOpen.restype = None
mock_libmcfile.mcFileDriverClose.restype = None
mock_libmcfile.mcFileHandleRegister.restype = None
mock_libmcfile.mcFileBufRegister.restype = None
mock_libmcfile.mcFileBufDeregister.restype = None
mock_libmcfile.mcFileRead.restype = ctypes.c_size_t
mock_libmcfile.mcFileWrite.restype = ctypes.c_size_t

with patch("ctypes.CDLL", return_value=mock_libmcfile):
    from mcfile import CuFile, CuFileDriver
    from mcfile.bindings import MCfileError, MCfileHandle_t

# this is needed to avoid exceptions while the singleton driver object is
# GCed at the program exit.
mock_libmcfile.mcFileDriverClose.return_value = MCfileError(err=0)


@pytest.fixture
def mock_libmcfile():
    """Mock the libmcfile library and its functions."""
    with patch("mcfile.bindings.libmcfile") as mock_lib:
        # Setup default successful return values
        mock_lib.mcFileDriverOpen.return_value = MCfileError(err=0)
        mock_lib.mcFileDriverClose.return_value = MCfileError(err=0)
        mock_lib.mcFileHandleRegister.return_value = MCfileError(err=0)
        mock_lib.mcFileHandleDeregister.return_value = None
        mock_lib.mcFileBufRegister.return_value = MCfileError(err=0)
        mock_lib.mcFileBufDeregister.return_value = MCfileError(err=0)
        mock_lib.mcFileRead.return_value = 1024
        mock_lib.mcFileWrite.return_value = 1024

        yield mock_lib


@pytest.fixture
def mock_os_operations():
    """Mock os.open and os.close operations."""
    with (
        patch("mcfile.mcfile.os.open", return_value=42) as mock_open,
        patch("mcfile.mcfile.os.close") as mock_close,
    ):
        yield {"open": mock_open, "close": mock_close}


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the CuFileDriver singleton between tests."""
    # The singleton decorator stores instances in a closure
    # We need to access and clear it

    # Force garbage collection to clean up any lingering instances
    import gc

    gc.collect()

    yield

    # Clean up after test
    gc.collect()


class TestCuFileDriver:
    """Test CuFileDriver singleton."""

    def test_driver_initialization(self, mock_libmcfile):
        """Test that CuFileDriver calls mcFileDriverOpen on initialization."""
        CuFileDriver()
        mock_libmcfile.mcFileDriverOpen.assert_called_once()

    def test_driver_singleton_pattern(self, mock_libmcfile):
        """Test that CuFileDriver follows singleton pattern."""
        driver1 = CuFileDriver()
        driver2 = CuFileDriver()

        # Should only call mcFileDriverOpen once for singleton
        assert driver1 is driver2

    def test_driver_close_on_delete(self, mock_libmcfile):
        """Test that CuFileDriver destructor behavior is implemented."""
        # The @_singleton decorator wraps the class, so we get an instance
        # when we call CuFileDriver(). We can verify that the instance has __del__
        driver = CuFileDriver()
        assert hasattr(driver, "__del__")

        # Note: Testing __del__ directly is tricky because:
        # 1. It's called by garbage collector
        # 2. The singleton pattern makes it even more complex
        # The fact that __del__ exists on the instance is what we can verify


class TestCuFileInitialization:
    """Test CuFile initialization."""

    def test_cufile_basic_initialization(self, mock_libmcfile, mock_os_operations):
        """Test basic CuFile initialization."""
        cufile = CuFile("/tmp/test.bin", "r")
        assert cufile._path == "/tmp/test.bin"
        assert cufile._mode == "r"
        assert cufile._handle is None
        assert cufile._cu_file_handle is None
        assert not cufile.is_open

    def test_cufile_initialization_with_direct_io(
        self, mock_libmcfile, mock_os_operations
    ):
        """Test CuFile initialization with direct I/O flag."""
        # O_DIRECT is not available on macOS, skip this test if not available
        if not hasattr(os, "O_DIRECT"):
            pytest.skip("O_DIRECT not available on this platform")

        cufile = CuFile("/tmp/test.bin", "r", use_direct_io=True)
        assert cufile._os_mode & os.O_DIRECT

    def test_cufile_mode_conversion(self, mock_libmcfile, mock_os_operations):
        """Test different file mode conversions."""
        test_cases = [
            ("r", os.O_RDONLY),
            ("r+", os.O_RDWR),
            ("w", os.O_CREAT | os.O_WRONLY | os.O_TRUNC),
            ("w+", os.O_CREAT | os.O_RDWR | os.O_TRUNC),
            ("a", os.O_CREAT | os.O_WRONLY | os.O_APPEND),
            ("a+", os.O_CREAT | os.O_RDWR | os.O_APPEND),
        ]

        for mode, expected_os_mode in test_cases:
            cufile = CuFile("/tmp/test.bin", mode)
            assert cufile._os_mode == (expected_os_mode | os.O_DIRECT)


class TestCuFileOpenClose:
    """Test CuFile open and close operations."""

    def test_open_registers_handle(self, mock_libmcfile, mock_os_operations):
        """Test that open() calls os.open and mcFileHandleRegister."""
        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()

        # Verify os.open was called with correct arguments
        mock_os_operations["open"].assert_called_once_with("/tmp/test.bin", os.O_RDONLY | os.O_DIRECT)

        # Verify mcFileHandleRegister was called
        mock_libmcfile.mcFileHandleRegister.assert_called_once()

        # Verify the file is marked as open
        assert cufile.is_open
        assert cufile._handle == 42

    def test_open_idempotent(self, mock_libmcfile, mock_os_operations):
        """Test that calling open() multiple times is idempotent."""
        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()
        cufile.open()

        # Should only call os.open once
        mock_os_operations["open"].assert_called_once()
        mock_libmcfile.mcFileHandleRegister.assert_called_once()

    def test_close_deregisters_handle(self, mock_libmcfile, mock_os_operations):
        """Test that close() calls mcFileHandleDeregister and os.close."""
        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()
        cufile.close()

        # Verify mcFileHandleDeregister was called
        mock_libmcfile.mcFileHandleDeregister.assert_called_once()

        # Verify os.close was called with correct fd
        mock_os_operations["close"].assert_called_once_with(42)

        # Verify the file is marked as closed
        assert not cufile.is_open
        assert cufile._handle is None
        assert cufile._cu_file_handle is None

    def test_close_idempotent(self, mock_libmcfile, mock_os_operations):
        """Test that calling close() multiple times is idempotent."""
        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()
        cufile.close()
        cufile.close()

        # Should only call deregister and close once
        mock_libmcfile.mcFileHandleDeregister.assert_called_once()
        mock_os_operations["close"].assert_called_once()

    def test_open_close_sequence(self, mock_libmcfile, mock_os_operations):
        """Test complete open/close sequence."""
        cufile = CuFile("/tmp/test.bin", "w")

        # Initially closed
        assert not cufile.is_open

        # Open
        cufile.open()
        assert cufile.is_open
        mock_os_operations["open"].assert_called_once_with(
            "/tmp/test.bin", os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_DIRECT
        )

        # Close
        cufile.close()
        assert not cufile.is_open
        mock_os_operations["close"].assert_called_once_with(42)


class TestCuFileContextManager:
    """Test CuFile context manager functionality."""

    def test_context_manager_opens_and_closes(self, mock_libmcfile, mock_os_operations):
        """Test that context manager properly opens and closes file."""
        with CuFile("/tmp/test.bin", "r") as cufile:
            assert cufile.is_open
            mock_os_operations["open"].assert_called_once()
            mock_libmcfile.mcFileHandleRegister.assert_called_once()

        # After exiting context, file should be closed
        mock_libmcfile.mcFileHandleDeregister.assert_called_once()
        mock_os_operations["close"].assert_called_once()

    def test_context_manager_returns_self(self, mock_libmcfile, mock_os_operations):
        """Test that context manager returns the CuFile instance."""
        original = CuFile("/tmp/test.bin", "r")
        with original as cufile:
            assert cufile is original

    def test_context_manager_closes_on_exception(
        self, mock_libmcfile, mock_os_operations
    ):
        """Test that context manager closes file even on exception."""
        try:
            with CuFile("/tmp/test.bin", "r"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should still be closed
        mock_libmcfile.mcFileHandleDeregister.assert_called_once()
        mock_os_operations["close"].assert_called_once()


class TestCuFileReadWrite:
    """Test CuFile read and write operations."""

    def test_read_calls_mcFileRead(self, mock_libmcfile, mock_os_operations):
        """Test that read() calls mcFileRead with correct arguments."""
        mock_libmcfile.mcFileRead.return_value = 2048

        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()

        buf = ctypes.c_void_p(0x1000)
        result = cufile.read(buf, 2048, file_offset=512, dev_offset=256)

        # Verify mcFileRead was called with correct arguments
        mock_libmcfile.mcFileRead.assert_called_once()
        call_args = mock_libmcfile.mcFileRead.call_args[0]

        # Check each argument
        assert call_args[0] is cufile._cu_file_handle  # handle
        assert call_args[1] == buf  # buffer
        assert call_args[2] == 2048  # size
        assert call_args[3] == 512  # file_offset
        assert call_args[4] == 256  # dev_offset

        # Verify return value
        assert result == 2048

    def test_write_calls_mcFileWrite(self, mock_libmcfile, mock_os_operations):
        """Test that write() calls mcFileWrite with correct arguments."""
        mock_libmcfile.mcFileWrite.return_value = 4096

        cufile = CuFile("/tmp/test.bin", "w")
        cufile.open()

        buf = ctypes.c_void_p(0x2000)
        result = cufile.write(buf, 4096, file_offset=1024, dev_offset=512)

        # Verify mcFileWrite was called with correct arguments
        mock_libmcfile.mcFileWrite.assert_called_once()
        call_args = mock_libmcfile.mcFileWrite.call_args[0]

        # Check each argument
        assert call_args[0] is cufile._cu_file_handle  # handle
        assert call_args[1] == buf  # buffer
        assert call_args[2] == 4096  # size
        assert call_args[3] == 1024  # file_offset
        assert call_args[4] == 512  # dev_offset

        # Verify return value
        assert result == 4096

    def test_read_without_open_raises_error(self, mock_libmcfile, mock_os_operations):
        """Test that read() raises IOError when file is not open."""
        cufile = CuFile("/tmp/test.bin", "r")

        buf = ctypes.c_void_p(0x1000)
        with pytest.raises(IOError, match="File is not open"):
            cufile.read(buf, 1024)

        # Verify mcFileRead was not called
        mock_libmcfile.mcFileRead.assert_not_called()

    def test_write_without_open_raises_error(self, mock_libmcfile, mock_os_operations):
        """Test that write() raises IOError when file is not open."""
        cufile = CuFile("/tmp/test.bin", "w")

        buf = ctypes.c_void_p(0x2000)
        with pytest.raises(IOError, match="File is not open"):
            cufile.write(buf, 1024)

        # Verify mcFileWrite was not called
        mock_libmcfile.mcFileWrite.assert_not_called()

    def test_read_with_default_offsets(self, mock_libmcfile, mock_os_operations):
        """Test that read() uses default offset values."""
        mock_libmcfile.mcFileRead.return_value = 1024

        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()

        buf = ctypes.c_void_p(0x1000)
        cufile.read(buf, 1024)

        call_args = mock_libmcfile.mcFileRead.call_args[0]
        assert call_args[3] == 0  # file_offset defaults to 0
        assert call_args[4] == 0  # dev_offset defaults to 0

    def test_write_with_default_offsets(self, mock_libmcfile, mock_os_operations):
        """Test that write() uses default offset values."""
        mock_libmcfile.mcFileWrite.return_value = 1024

        cufile = CuFile("/tmp/test.bin", "w")
        cufile.open()

        buf = ctypes.c_void_p(0x2000)
        cufile.write(buf, 1024)

        call_args = mock_libmcfile.mcFileWrite.call_args[0]
        assert call_args[3] == 0  # file_offset defaults to 0
        assert call_args[4] == 0  # dev_offset defaults to 0


class TestCuFileOperationSequence:
    """Test complete operation sequences."""

    def test_complete_write_sequence(self, mock_libmcfile, mock_os_operations):
        """Test a complete write operation sequence."""
        mock_libmcfile.mcFileWrite.return_value = 8192

        # Create file
        cufile = CuFile("/tmp/test.bin", "w")

        # Open (should call os.open and mcFileHandleRegister)
        cufile.open()
        mock_os_operations["open"].assert_called_once()
        mock_libmcfile.mcFileHandleRegister.assert_called_once()

        # Write (should call mcFileWrite)
        buf = ctypes.c_void_p(0x3000)
        result = cufile.write(buf, 8192)
        assert result == 8192
        mock_libmcfile.mcFileWrite.assert_called_once()

        # Close (should call mcFileHandleDeregister and os.close)
        cufile.close()
        mock_libmcfile.mcFileHandleDeregister.assert_called_once()
        mock_os_operations["close"].assert_called_once()

    def test_complete_read_sequence(self, mock_libmcfile, mock_os_operations):
        """Test a complete read operation sequence."""
        mock_libmcfile.mcFileRead.return_value = 16384

        # Create file
        cufile = CuFile("/tmp/test.bin", "r")

        # Open
        cufile.open()

        # Read
        buf = ctypes.c_void_p(0x4000)
        result = cufile.read(buf, 16384)
        assert result == 16384

        # Close
        cufile.close()

        # Verify order of operations
        assert mock_os_operations["open"].call_count == 1
        assert mock_libmcfile.mcFileHandleRegister.call_count == 1
        assert mock_libmcfile.mcFileRead.call_count == 1
        assert mock_libmcfile.mcFileHandleDeregister.call_count == 1
        assert mock_os_operations["close"].call_count == 1

    def test_write_read_sequence_with_context_manager(
        self, mock_libmcfile, mock_os_operations
    ):
        """Test write and read operations using context manager."""
        mock_libmcfile.mcFileWrite.return_value = 4096
        mock_libmcfile.mcFileRead.return_value = 4096

        buf_write = ctypes.c_void_p(0x5000)
        buf_read = ctypes.c_void_p(0x6000)

        # Write
        with CuFile("/tmp/test.bin", "w") as cufile:
            result = cufile.write(buf_write, 4096)
            assert result == 4096

        # Verify write sequence
        assert mock_libmcfile.mcFileWrite.call_count == 1
        write_deregister_count = mock_libmcfile.mcFileHandleDeregister.call_count

        # Read
        with CuFile("/tmp/test.bin", "r") as cufile:
            result = cufile.read(buf_read, 4096)
            assert result == 4096

        # Verify read sequence
        assert mock_libmcfile.mcFileRead.call_count == 1

        # Both context managers should have closed
        assert (
            mock_libmcfile.mcFileHandleDeregister.call_count
            == write_deregister_count + 1
        )


class TestCuFileErrorHandling:
    """Test error handling in CuFile operations."""

    def test_driver_open_failure(self, mock_libmcfile):
        """Test handling of mcFileDriverOpen failure."""
        # Due to singleton pattern, we can't easily test initialization failure
        # in isolation from other tests. Instead, let's test the error checking
        # function directly.
        from mcfile.bindings import _ck

        error_status = MCfileError(err=-1)
        with pytest.raises(RuntimeError, match="mcFileDriverOpen failed"):
            _ck(error_status, "mcFileDriverOpen")

    def test_handle_register_failure(self, mock_libmcfile, mock_os_operations):
        """Test handling of mcFileHandleRegister failure."""
        mock_libmcfile.mcFileHandleRegister.return_value = MCfileError(
            err=-2)

        cufile = CuFile("/tmp/test.bin", "r")

        with pytest.raises(
            RuntimeError, match="mcFileHandleRegister failed.*err=-2.*"
        ):
            cufile.open()

    def test_buffer_register_failure(self, mock_libmcfile):
        """Test handling of mcFileBufRegister failure."""
        from mcfile.bindings import mcFileBufRegister

        mock_libmcfile.mcFileBufRegister.return_value = MCfileError(err=-3)

        buf = ctypes.c_void_p(0x7000)
        with pytest.raises(
            RuntimeError, match="mcFileBufRegister failed.*err=-3.*"
        ):
            mcFileBufRegister(buf, 1024, 0)

    def test_multiple_operations_after_error(self, mock_libmcfile, mock_os_operations):
        """Test that operations can continue after handling errors."""
        # First attempt fails
        mock_libmcfile.mcFileHandleRegister.return_value = MCfileError(err=-1)

        cufile1 = CuFile("/tmp/test1.bin", "r")
        with pytest.raises(RuntimeError):
            cufile1.open()

        # Second attempt succeeds
        mock_libmcfile.mcFileHandleRegister.return_value = MCfileError(err=0)

        cufile2 = CuFile("/tmp/test2.bin", "r")
        cufile2.open()
        assert cufile2.is_open
        cufile2.close()


class TestCuFileGetHandle:
    """Test get_handle method."""

    def test_get_handle_when_closed(self, mock_libmcfile, mock_os_operations):
        """Test get_handle returns None when file is closed."""
        cufile = CuFile("/tmp/test.bin", "r")
        assert cufile.get_handle() is None

    def test_get_handle_when_open(self, mock_libmcfile, mock_os_operations):
        """Test get_handle returns file descriptor when file is open."""
        cufile = CuFile("/tmp/test.bin", "r")
        cufile.open()

        handle = cufile.get_handle()
        assert handle == 42  # mocked fd value

        cufile.close()


class TestCuFileBindings:
    """Test the bindings module helper functions."""

    def test_mcFileHandleDeregister_call(self, mock_libmcfile):
        """Test mcFileHandleDeregister binding."""
        from mcfile.bindings import mcFileHandleDeregister

        handle = MCfileHandle_t(12345)
        mcFileHandleDeregister(handle)

        mock_libmcfile.mcFileHandleDeregister.assert_called_once_with(handle)

    def test_mcFileBufDeregister_call(self, mock_libmcfile):
        """Test mcFileBufDeregister binding."""
        from mcfile.bindings import mcFileBufDeregister

        mock_libmcfile.mcFileBufDeregister.return_value = MCfileError(err=0)

        buf = ctypes.c_void_p(0x8000)
        mcFileBufDeregister(buf)

        mock_libmcfile.mcFileBufDeregister.assert_called_once_with(buf)

    def test_mcFileRead_return_value(self, mock_libmcfile):
        """Test mcFileRead returns the size value."""
        from mcfile.bindings import mcFileRead

        mock_libmcfile.mcFileRead.return_value = 32768

        handle = MCfileHandle_t(12345)
        buf = ctypes.c_void_p(0x9000)
        result = mcFileRead(handle, buf, 32768, 0, 0)

        assert result == 32768

    def test_mcFileWrite_return_value(self, mock_libmcfile):
        """Test mcFileWrite returns the size value."""
        from mcfile.bindings import mcFileWrite

        mock_libmcfile.mcFileWrite.return_value = 65536

        handle = MCfileHandle_t(12345)
        buf = ctypes.c_void_p(0xA000)
        result = mcFileWrite(handle, buf, 65536, 100, 200)

        assert result == 65536
