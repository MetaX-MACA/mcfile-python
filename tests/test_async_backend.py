"""
Real-hardware tests for the mcfile async_backend module.

These tests exercise the mcFile async C API end-to-end through
``mcfile.async_backend`` and require a real GPU + the mcFile driver
(``libmcfile.so``). They mirror the style of ``tests/test_cufile.py`` but use
torch for device allocation, memset, stream handling and host copies (the
async_backend API is torch-tensor centric).

Run on a GDS-capable host, e.g.::

    TEST_CUFILE_WORK_DIR=/mnt/gds conda activate cudf_2512 \\
        python -m pytest tests/test_async_backend.py -s
"""

import os
import time

import torch

from mcfile.async_backend import (
    AsyncHandle,
    register_buffer,
    deregister_buffer,
    register_stream,
    deregister_stream,
    close_driver,
)

BUF_SIZE = int(os.environ.get("TEST_CUFILE_BUF_SIZE", 256)) * 1024 * 1024
WORK_DIR = os.environ.get("TEST_CUFILE_WORK_DIR", ".")
PATTERN_BYTE = int(os.environ.get("TEST_CUFILE_PATTERN_BYTE", 0xAB))
CUDA_DEVICE = int(os.environ.get("TEST_CUFILE_CUDA_DEVICE", 0))

file_path = os.path.join(WORK_DIR, "test_async.bin")

# --- module-level CUDA setup (mirrors test_cufile.py) ----------------------

torch.cuda.set_device(CUDA_DEVICE)
dev = torch.device(f"cuda:{CUDA_DEVICE}")
buf_w = torch.empty(BUF_SIZE, dtype=torch.uint8, device=dev)
buf_r = torch.empty(BUF_SIZE, dtype=torch.uint8, device=dev)
buf_w.fill_(PATTERN_BYTE)

# A single CUDA stream is registered once and used to batch submissions; the
# caller drains a batch with a single synchronize() (see async_backend docs).
stream = torch.cuda.Stream()
register_buffer(buf_w)
register_buffer(buf_r)
register_stream(stream.cuda_stream)


def test_async_handle_initialization():
    """Test that an AsyncHandle can be created for writing."""
    h = AsyncHandle(file_path, writable=True, fallocate_size=BUF_SIZE)
    assert isinstance(h, AsyncHandle)
    assert h.writable
    h.close()


def test_async_handle_context_manager():
    """Test that AsyncHandle works as a context manager."""
    with AsyncHandle(file_path, writable=True, fallocate_size=BUF_SIZE) as h:
        assert isinstance(h, AsyncHandle)
        assert h.writable


def test_async_write_read_roundtrip():
    """Test write_async -> sync -> read_async -> sync and verify bytes_done."""
    raw_stream = stream.cuda_stream

    begin = time.perf_counter()
    with AsyncHandle(file_path, writable=True, fallocate_size=BUF_SIZE) as h:
        begin_write = time.perf_counter()
        sub = h.write_async(
            buf_base=buf_w.data_ptr(),
            size=BUF_SIZE,
            file_offset=0,
            buf_offset=0,
            raw_stream=raw_stream,
        )
        stream.synchronize()
        write_time = time.perf_counter() - begin_write
        assert sub.bytes_done == BUF_SIZE
    dt = time.perf_counter() - begin
    print(
        f"ASYNC WRITE (w/o open/register) {sub.bytes_done / 1024 / 1024:.2f}MB "
        f"in {write_time * 1e3:.2f}ms "
        f"({sub.bytes_done / write_time / 1024 / 1024 / 1024:.2f}GB/s)"
    )
    print(
        f"FULL ASYNC WRITE {sub.bytes_done / 1024 / 1024:.2f}MB in {dt * 1e3:.2f}ms "
        f"({sub.bytes_done / dt / 1024 / 1024 / 1024:.2f}GB/s)"
    )

    begin = time.perf_counter()
    with AsyncHandle(file_path, writable=False) as h:
        begin_read = time.perf_counter()
        sub = h.read_async(
            buf_base=buf_r.data_ptr(),
            size=BUF_SIZE,
            file_offset=0,
            buf_offset=0,
            raw_stream=raw_stream,
        )
        stream.synchronize()
        read_time = time.perf_counter() - begin_read
        assert sub.bytes_done == BUF_SIZE
    dt = time.perf_counter() - begin
    print(
        f"ASYNC READ (w/o open/register) {sub.bytes_done / 1024 / 1024:.2f}MB "
        f"in {read_time * 1e3:.2f}ms "
        f"({sub.bytes_done / read_time / 1024 / 1024 / 1024:.2f}GB/s)"
    )
    print(
        f"FULL ASYNC READ {sub.bytes_done / 1024 / 1024:.2f}MB in {dt * 1e3:.2f}ms "
        f"({sub.bytes_done / dt / 1024 / 1024 / 1024:.2f}GB/s)"
    )

    # Verify the read-back data matches the written pattern.
    assert torch.equal(buf_r, buf_w)
    assert bool((buf_r == PATTERN_BYTE).all())


def test_async_batched_submissions():
    """Test that multiple submissions drain with a single synchronize()."""
    raw_stream = stream.cuda_stream
    half = BUF_SIZE // 2

    with AsyncHandle(file_path, writable=True, fallocate_size=BUF_SIZE) as h:
        subs = [
            h.write_async(
                buf_base=buf_w.data_ptr(),
                size=half,
                file_offset=0,
                buf_offset=0,
                raw_stream=raw_stream,
            ),
            h.write_async(
                buf_base=buf_w.data_ptr(),
                size=half,
                file_offset=half,
                buf_offset=half,
                raw_stream=raw_stream,
            ),
        ]
        # One sync drains the whole batch.
        stream.synchronize()
        for s in subs:
            assert s.bytes_done == half


# --- module-level teardown --------------------------------------------------

import atexit


@atexit.register
def _teardown():
    try:
        deregister_buffer(buf_w)
        deregister_buffer(buf_r)
        deregister_stream(stream.cuda_stream)
        close_driver()
    except Exception:
        pass
    try:
        os.remove(file_path)
    except OSError:
        pass
