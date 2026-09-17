"""
mcfile-python - A basic Python wrapper for the MACA mcFile API
"""

__version__ = "0.2.1"

from .mcfile import McFile, McFileDriver
from .mcfile import McFileDriver as CuFileDriver
from .mcfile import McFile as CuFile
from mcfile import _async_backend as mcfile_async

__all__ = ["CuFile", "CuFileDriver", "McFile", "McFileDriver", "mcfile_async"]
