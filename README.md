# mcfile-python

A basic Python wrapper for the MACA mcFile API.

This repo is forked from https://github.com/yanok/cufile-python.git
## Installation

```bash
pip install mcfile-python
```

## Usage

Basic usage with the `CuFile` context manager:

```python
from mcfile import CuFile
import torch
import ctypes

# allocate an empty buffer in VRAM
t = torch.empty((1024, 1024, 16), dtype=torch.float32, device="cuda")
with CuFile("test.bin", "r") as f:
    f.read(ctypes.c_void_p(t.data_ptr()), t.nbytes)
```

Alternatively one could `import mcfile.bindings` and use it as if calling libmcfile from C++.

## Development

To set up the development environment:

1. Clone the repository:
```bash
git clone https://github.com/MetaX-MACA/mcfile-python.git
cd mcfile-python
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
```

3. Run tests:
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 