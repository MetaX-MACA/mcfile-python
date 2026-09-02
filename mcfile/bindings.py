import ctypes

ctypes.CDLL("libmcruntime.so", ctypes.RTLD_GLOBAL)

libmcfile = ctypes.CDLL("libmcfile.so")

async_supported = False

class MCfileError(ctypes.Structure):
    _fields_ = [("err", ctypes.c_int)]


MCfileHandle_t = ctypes.c_void_p


class DescrUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("handle", ctypes.c_void_p)]


class MCfileDescr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", DescrUnion),
        ("fs_ops", ctypes.c_void_p),
    ]


def _declare_common_signatures() -> None:
    libmcfile.mcFileDriverOpen.restype = MCfileError
    libmcfile.mcFileDriverClose.restype = MCfileError
    libmcfile.mcFileHandleRegister.restype = MCfileError
    libmcfile.mcFileBufRegister.restype = MCfileError
    libmcfile.mcFileBufDeregister.restype = MCfileError
    libmcfile.mcFileRead.restype = ctypes.c_size_t
    libmcfile.mcFileWrite.restype = ctypes.c_size_t
    libmcfile.mcFileHandleRegister.argtypes = [
        ctypes.POINTER(MCfileHandle_t),
        ctypes.POINTER(MCfileDescr),
    ]
    libmcfile.mcFileHandleDeregister.argtypes = [MCfileHandle_t]
    libmcfile.mcFileBufRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libmcfile.mcFileBufDeregister.argtypes = [ctypes.c_void_p]
    libmcfile.mcFileRead.argtypes = [
        MCfileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    libmcfile.mcFileWrite.argtypes = [
        MCfileHandle_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]

def get_lib_func(lib, name):
    try:
        return getattr(lib, name)
    except AttributeError:
        return None

def _declare_async_signatures() -> None:
    """Set argtypes/restype on libmcfile symbols. Idempotent."""

    mcFileReadAsync = get_lib_func(libmcfile, "mcFileReadAsync")
    if mcFileReadAsync is None:
        return
    global async_supported
    async_supported = True
    libmcfile.mcFileReadAsync.argtypes = [
        ctypes.c_void_p,  # MCUfileHandle_t fh
        ctypes.c_void_p,  # void *bufPtr_base
        ctypes.POINTER(ctypes.c_size_t),  # size_t *size_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *file_offset_p
        ctypes.POINTER(ctypes.c_int64),  # off_t *bufPtr_offset_p
        ctypes.POINTER(ctypes.c_int64),  # ssize_t *bytes_read_p
        ctypes.c_void_p,  # MCstream stream
    ]
    libmcfile.mcFileReadAsync.restype = MCfileError

    libmcfile.mcFileWriteAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_void_p,
    ]
    libmcfile.mcFileWriteAsync.restype = MCfileError

    libmcfile.mcFileStreamRegister.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libmcfile.mcFileStreamRegister.restype = MCfileError

    libmcfile.mcFileStreamDeregister.argtypes = [ctypes.c_void_p]
    libmcfile.mcFileStreamDeregister.restype = MCfileError


# convenience
def _ck(status: MCfileError, name: str):
    if status.err != 0:
        raise RuntimeError(
            f"{name} failed (mcFile err={status.err})"
        )


def mcFileDriverOpen() -> None:
    _ck(libmcfile.mcFileDriverOpen(), "mcFileDriverOpen")


def mcFileDriverClose() -> None:
    _ck(libmcfile.mcFileDriverClose(), "mcFileDriverClose")


def mcFileHandleRegister(fd: int) -> MCfileHandle_t:
    descr = MCfileDescr(type=1, handle=DescrUnion(fd=fd))
    handle = MCfileHandle_t()
    _ck(libmcfile.mcFileHandleRegister(handle, descr), "mcFileHandleRegister")
    return handle


def mcFileHandleDeregister(handle: MCfileHandle_t) -> None:
    libmcfile.mcFileHandleDeregister(handle), "mcFileHandleDeregister"


def mcFileBufRegister(buf: ctypes.c_void_p, size: int, flags: int) -> None:
    _ck(libmcfile.mcFileBufRegister(buf, size, flags), "mcFileBufRegister")


def mcFileBufDeregister(buf: ctypes.c_void_p) -> None:
    _ck(libmcfile.mcFileBufDeregister(buf), "mcFileBufDeregister")


def mcFileRead(
    handle: MCfileHandle_t,
    buf: ctypes.c_void_p,
    size: int,
    file_offset: int,
    dev_offset: int,
) -> int:
    return libmcfile.mcFileRead(handle, buf, size, file_offset, dev_offset)


def mcFileWrite(
    handle: MCfileHandle_t,
    buf: ctypes.c_void_p,
    size: int,
    file_offset: int,
    dev_offset: int,
) -> int:
    return libmcfile.mcFileWrite(handle, buf, size, file_offset, dev_offset)

_declare_common_signatures()
_declare_async_signatures()