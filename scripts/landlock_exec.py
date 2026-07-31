from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import platform
import sys
from pathlib import Path


SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15

LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 0
LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 1

PR_SET_NO_NEW_PRIVS = 38


class RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.syscall.restype = ctypes.c_long
LIBC.prctl.restype = ctypes.c_int


class LandlockError(RuntimeError):
    """Raised when the kernel cannot enforce the requested Landlock policy."""


def _syscall(number: int, *arguments) -> int:
    result = int(LIBC.syscall(number, *arguments))
    if result < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return result


def landlock_abi_version() -> int:
    if platform.machine() not in {"x86_64", "amd64"}:
        raise LandlockError(f"Unsupported syscall architecture: {platform.machine()}")
    try:
        return _syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP}:
            raise LandlockError("Landlock is not supported by this kernel.") from exc
        raise LandlockError(f"Unable to query Landlock ABI: {exc}") from exc


def handled_filesystem_access(abi_version: int) -> int:
    rights = (
        LANDLOCK_ACCESS_FS_EXECUTE
        | LANDLOCK_ACCESS_FS_WRITE_FILE
        | LANDLOCK_ACCESS_FS_READ_FILE
        | LANDLOCK_ACCESS_FS_READ_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_FILE
        | LANDLOCK_ACCESS_FS_MAKE_CHAR
        | LANDLOCK_ACCESS_FS_MAKE_DIR
        | LANDLOCK_ACCESS_FS_MAKE_REG
        | LANDLOCK_ACCESS_FS_MAKE_SOCK
        | LANDLOCK_ACCESS_FS_MAKE_FIFO
        | LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi_version >= 2:
        rights |= LANDLOCK_ACCESS_FS_REFER
    if abi_version >= 3:
        rights |= LANDLOCK_ACCESS_FS_TRUNCATE
    if abi_version >= 5:
        rights |= LANDLOCK_ACCESS_FS_IOCTL_DEV
    return rights


def create_ruleset(abi_version: int) -> tuple[int, int]:
    filesystem_access = handled_filesystem_access(abi_version)
    network_access = 0
    if abi_version >= 4:
        network_access = LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP
    ruleset = RulesetAttr(
        handled_access_fs=filesystem_access,
        handled_access_net=network_access,
        scoped=0,
    )
    try:
        descriptor = _syscall(
            SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(ruleset),
            ctypes.sizeof(ruleset),
            ctypes.c_uint(0),
        )
    except OSError as exc:
        raise LandlockError(f"Unable to create Landlock ruleset: {exc}") from exc
    return descriptor, filesystem_access


def allowed_access_for(path: Path, writable: bool, handled_access: int) -> int:
    if path.is_dir():
        if writable:
            return handled_access
        return (
            LANDLOCK_ACCESS_FS_EXECUTE
            | LANDLOCK_ACCESS_FS_READ_FILE
            | LANDLOCK_ACCESS_FS_READ_DIR
        ) & handled_access
    access = LANDLOCK_ACCESS_FS_READ_FILE
    if os.access(path, os.X_OK):
        access |= LANDLOCK_ACCESS_FS_EXECUTE
    if writable:
        access |= LANDLOCK_ACCESS_FS_WRITE_FILE
        access |= LANDLOCK_ACCESS_FS_TRUNCATE
        access |= LANDLOCK_ACCESS_FS_IOCTL_DEV
    return access & handled_access


def add_path_rule(
    ruleset_fd: int,
    path: Path,
    *,
    writable: bool,
    handled_access: int,
) -> None:
    resolved_path = path.resolve(strict=True)
    descriptor = os.open(resolved_path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = PathBeneathAttr(
            allowed_access=allowed_access_for(resolved_path, writable, handled_access),
            parent_fd=descriptor,
        )
        _syscall(
            SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            ctypes.c_uint(0),
        )
    except OSError as exc:
        raise LandlockError(f"Unable to add Landlock rule for {resolved_path}: {exc}") from exc
    finally:
        os.close(descriptor)


def restrict_current_process(read_paths: list[Path], write_paths: list[Path]) -> int:
    abi_version = landlock_abi_version()
    if abi_version < 3:
        raise LandlockError(
            f"Landlock ABI {abi_version} is too old; ABI 3 or newer is required."
        )
    ruleset_fd, handled_access = create_ruleset(abi_version)
    try:
        for path in read_paths:
            add_path_rule(
                ruleset_fd,
                path,
                writable=False,
                handled_access=handled_access,
            )
        for path in write_paths:
            add_path_rule(
                ruleset_fd,
                path,
                writable=True,
                handled_access=handled_access,
            )
        if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error_number = ctypes.get_errno()
            raise LandlockError(
                f"Unable to set no_new_privs: {os.strerror(error_number)}"
            )
        _syscall(
            SYS_LANDLOCK_RESTRICT_SELF,
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint(0),
        )
    except OSError as exc:
        raise LandlockError(f"Unable to enforce Landlock ruleset: {exc}") from exc
    finally:
        os.close(ruleset_fd)
    return abi_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a Landlock allow-list and execute a command."
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--allow-read", action="append", type=Path, default=[])
    parser.add_argument("--allow-write", action="append", type=Path, default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probe:
        print(json.dumps({"landlock_abi": landlock_abi_version()}))
        return 0
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("No command was supplied after --.", file=sys.stderr)
        return 64
    try:
        restrict_current_process(args.allow_read, args.allow_write)
    except (LandlockError, OSError) as exc:
        print(f"Landlock policy failed closed: {exc}", file=sys.stderr)
        return 77
    os.execve(command[0], command, os.environ.copy())
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
