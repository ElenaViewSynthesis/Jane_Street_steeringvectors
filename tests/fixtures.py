from __future__ import annotations

import struct
import zipfile
from pathlib import Path


def _binunicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def storage_reference_pickle() -> bytes:
    """Build a protocol-2 stream resembling a PyTorch tensor storage reference."""
    return b"".join(
        [
            b"\x80\x02",  # PROTO 2
            b"(",  # MARK
            _binunicode("storage"),
            b"ctorch\nFloatStorage\n",  # GLOBAL torch.FloatStorage
            _binunicode("0"),
            _binunicode("cpu"),
            b"K\x04",  # BININT1 4
            b"t",  # TUPLE
            b"Q",  # BINPERSID
            b".",  # STOP
        ]
    )


def stack_global_pickle() -> bytes:
    """Build the smallest protocol-4 stream containing STACK_GLOBAL."""
    return b"\x80\x04\x8c\x05torch\x8c\x0cFloatStorage\x93."


def write_tiny_torch_archive(path: Path) -> Path:
    """Write a non-executable, PyTorch-shaped ZIP fixture for static tests."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("puzzle/data.pkl", storage_reference_pickle())
        archive.writestr("puzzle/data/0", b"\x00" * 16)
        archive.writestr("puzzle/byteorder", "little")
        archive.writestr("puzzle/version", "3\n")
    return path
