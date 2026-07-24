"""One secure, directory-FD / no-follow filesystem layer for Phase E project I/O.

Every Phase E read, create-only publish, and atomic replace of a project-owned file
(cassette, report, manifest, workflow, or temporary) goes through here. It composes the
``project_init`` primitives (``_open_absolute_directory``, ``_open_existing_components``,
``_read_optional_regular_at``, ``_publish_file``, ``_identity``, ``_relative_parts``) so a
symlinked parent, a swapped parent, a hard link, a FIFO/device, or a target that appears
between preflight and commit fails closed — never a path-based ``open``/``mkdir``/``rename``
that would follow a symlink out of the project. When the secure primitives are unavailable
the caller must fail closed with exit 2.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from agent_cassette.project_init import (
    ProjectInitError,
    _identity,
    _open_absolute_directory,
    _open_directory_at,
    _open_existing_components,
    _publish_file,
    _relative_parts,
    _require_secure_filesystem_primitives,
)


class SecureFilesystemError(ValueError):
    """A project path could not be traversed or written safely (fail closed → exit 2)."""


def open_root(root: Path) -> int:
    """Open the lexical project root as a no-follow directory FD, or fail closed."""
    try:
        _require_secure_filesystem_primitives()
        return _open_absolute_directory(root)
    except ProjectInitError as error:
        raise SecureFilesystemError(str(error)) from error


def read_regular(root_fd: int, relative: str) -> bytes | None:
    """Read a project-relative regular single-link file's bytes, or None if absent.

    Rejects a symlink, hard link (``st_nlink > 1``), or non-regular object anywhere on the
    path, and any parent that is not a real directory (all opened ``O_NOFOLLOW``).
    """
    parts = _relative_parts(relative)
    try:
        parent_fd, _identities = _open_existing_components(root_fd, parts[:-1])
    except ProjectInitError as error:
        raise SecureFilesystemError(str(error)) from error
    if parent_fd is None:
        return None
    try:
        result = _read_regular_single_link(parent_fd, parts[-1])
    finally:
        os.close(parent_fd)
    return None if result is None else result[0]


def _read_regular_single_link(parent_fd: int, name: str) -> tuple[bytes, tuple[int, int]] | None:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SecureFilesystemError(f"cannot inspect {name}: {error}") from error
    if not stat.S_ISREG(before.st_mode):
        raise SecureFilesystemError(f"{name} is not a regular file")
    if before.st_nlink != 1:
        raise SecureFilesystemError(f"{name} is a hard link")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise SecureFilesystemError(f"cannot safely open {name}: {error}") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(before) != _identity(opened)
        ):
            raise SecureFilesystemError(f"{name} changed while it was being opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                # identity is the fstat of the exact inode read, not the pre-read lstat
                return b"".join(chunks), _identity(opened)
            chunks.append(chunk)
    except OSError as error:
        raise SecureFilesystemError(f"cannot safely read {name}: {error}") from error
    finally:
        os.close(descriptor)


def snapshot(root_fd: int, relative: str) -> tuple[bytes, tuple[int, int]] | None:
    """Read a project-relative regular file and return (bytes, identity), or None."""
    parts = _relative_parts(relative)
    try:
        parent_fd, _identities = _open_existing_components(root_fd, parts[:-1])
    except ProjectInitError as error:
        raise SecureFilesystemError(str(error)) from error
    if parent_fd is None:
        return None
    try:
        return _read_regular_single_link(parent_fd, parts[-1])
    finally:
        os.close(parent_fd)


def identity_of(root_fd: int, relative: str) -> tuple[int, int] | None:
    """Return the (dev, ino) identity of a project-relative regular file, or None."""
    parts = _relative_parts(relative)
    try:
        parent_fd, _identities = _open_existing_components(root_fd, parts[:-1])
    except ProjectInitError as error:
        raise SecureFilesystemError(str(error)) from error
    if parent_fd is None:
        return None
    try:
        stats = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    finally:
        os.close(parent_fd)
    if not stat.S_ISREG(stats.st_mode):
        raise SecureFilesystemError(f"{relative} is not a regular file")
    return _identity(stats)


def _ensure_dir_fd(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open (creating if needed) each directory component under root, all no-follow."""
    current = os.dup(root_fd)
    try:
        for part in parts:
            try:
                child = _open_directory_at(current, part)
            except FileNotFoundError:
                try:
                    os.mkdir(part, 0o700, dir_fd=current)
                except FileExistsError:
                    pass  # a concurrent create; reopen it no-follow below
                child = _open_directory_at(current, part)  # O_DIRECTORY|O_NOFOLLOW
            os.close(current)
            current = child
        return current
    except OSError as error:
        os.close(current)
        raise SecureFilesystemError(f"unsafe project directory component: {error}") from error


def preflight_parent(root_fd: int, relative: str) -> None:
    """Open/create a project-relative file's parent chain (no-follow), raising on an
    untrusted (symlinked/swapped) parent. Used to fail closed before running a child."""
    parent_fd = _ensure_dir_fd(root_fd, _relative_parts(relative)[:-1])
    os.close(parent_fd)


def publish_create_only(root_fd: int, relative: str, content: bytes) -> None:
    """Create-only publish: never overwrites; fails closed if the final name exists."""
    parts = _relative_parts(relative)
    parent_fd = _ensure_dir_fd(root_fd, parts[:-1])
    try:
        # _publish_file returns the still-open object fd (keep_descriptor=True); close it.
        _identity_ignored, object_fd = _publish_file(parent_fd, parts[-1], content)
        os.close(object_fd)
        _fsync_dir(parent_fd)
    except (ProjectInitError, OSError) as error:
        raise SecureFilesystemError(str(error)) from error
    finally:
        os.close(parent_fd)


def atomic_replace(
    root_fd: int, relative: str, content: bytes, *, expect_identity: tuple[int, int] | None = None
) -> None:
    """Atomically (re)write a project-relative regular file through a verified dir FD.

    Rejects a symlink/hard-link/non-regular final target. When ``expect_identity`` is
    given, the current final must still have that identity immediately before the rename
    (fail closed on a concurrent swap)."""
    parts = _relative_parts(relative)
    parent_fd = _ensure_dir_fd(root_fd, parts[:-1])
    name = parts[-1]
    try:
        existing = _lstat_optional(parent_fd, name)
        if existing is not None:
            if not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1:
                raise SecureFilesystemError(f"{name} is not a regular single-link file")
            if expect_identity is not None and _identity(existing) != expect_identity:
                raise SecureFilesystemError(f"{name} changed before publish")
        elif expect_identity is not None:
            raise SecureFilesystemError(f"{name} disappeared before publish")
        temporary = f".agent-cassette-{os.urandom(8).hex()}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        try:
            descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        except OSError as error:
            raise SecureFilesystemError(f"cannot create report temporary: {error}") from error
        try:
            _write_all(descriptor, content)
            os.fsync(descriptor)
        except OSError as error:
            _unlink_quietly(parent_fd, temporary)
            raise SecureFilesystemError(f"write failed for {name}: {error}") from error
        finally:
            os.close(descriptor)
        try:
            os.rename(temporary, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        except OSError as error:
            _unlink_quietly(parent_fd, temporary)
            raise SecureFilesystemError(f"atomic publish failed for {name}: {error}") from error
        _fsync_dir(parent_fd)
    finally:
        os.close(parent_fd)


def _lstat_optional(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SecureFilesystemError(f"cannot inspect {name}: {error}") from error


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - defensive OS invariant
            raise SecureFilesystemError("short write while publishing")
        view = view[written:]


def _fsync_dir(parent_fd: int) -> None:
    try:
        os.fsync(parent_fd)
    except OSError:  # pragma: no cover - some platforms disallow directory fsync
        pass


def _unlink_quietly(parent_fd: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=parent_fd)
    except OSError:
        pass
