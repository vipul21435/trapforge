"""Canonical, byte-exact writers and readers for every file a task emits.

A task is graded by comparing bytes, so the bytes must not depend on the platform, the
process, the dict insertion order or the locale. Every file a task family produces goes
through one of these writers:

* :func:`csv_bytes`: a header plus rows of ``int`` or non-empty plain ASCII ``str`` cells,
  comma separated, ``\\n`` line endings, a final newline, no quoting (cells that would need
  quoting, and empty cells, are rejected instead, so :func:`parse_csv` reads back every file
  the writer accepts);
* :func:`json_bytes`: sorted keys, two-space indent, ASCII only, a final newline, and no
  floats (a float is not exact);
* :func:`text_bytes`: ASCII text with ``\\n`` line endings and exactly one final newline.

:func:`tree_digest` hashes a whole set of named files with SHA-256 over a framed, sorted
encoding, so two task instances are byte-identical exactly when their digests match.

>>> csv_bytes(["account", "balance"], [[3, -150], [7, 20]])
b'account,balance\\n3,-150\\n7,20\\n'
>>> parse_csv(b'account,balance\\n3,-150\\n7,20\\n')
(('account', 'balance'), (('3', '-150'), ('7', '20')))
>>> json_bytes({"b": [1, 2], "a": "x"})
b'{\\n  "a": "x",\\n  "b": [\\n    1,\\n    2\\n  ]\\n}\\n'
>>> tree_digest({"a.txt": b"hi\\n"})[:16]
'78beee6f4d52628c'
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "CanonicalError",
    "csv_bytes",
    "json_bytes",
    "parse_csv",
    "sha256_hex",
    "text_bytes",
    "tree_digest",
]

_FORBIDDEN_CELL = frozenset(',"\r\n')


class CanonicalError(ValueError):
    """Raised when a value cannot be written (or read) canonically."""


def _cell(value: object, where: str) -> str:
    if type(value) is int:
        return str(value)
    if isinstance(value, str):
        if not value:
            raise CanonicalError(f"{where}: empty cells are not allowed")
        if not value.isascii() or not value.isprintable():
            raise CanonicalError(f"{where}: cell {value!r} is not printable ASCII")
        if _FORBIDDEN_CELL.intersection(value):
            raise CanonicalError(f"{where}: cell {value!r} would need CSV quoting")
        if value != value.strip():
            raise CanonicalError(f"{where}: cell {value!r} has surrounding whitespace")
        return value
    raise CanonicalError(f"{where}: cells must be int or str, got {type(value).__name__}")


def csv_bytes(header: Sequence[str], rows: Iterable[Sequence[int | str]]) -> bytes:
    """The canonical CSV encoding of ``header`` and ``rows`` (every row as wide as the header).

    Row order is kept exactly as given: sorting is part of a task's specification, so the
    caller decides it.
    """
    if not header:
        raise CanonicalError("a CSV file needs at least one column")
    width = len(header)
    lines = [",".join(_cell(name, "header") for name in header)]
    for index, row in enumerate(rows):
        if len(row) != width:
            raise CanonicalError(f"row {index} has {len(row)} cells, the header has {width}")
        lines.append(",".join(_cell(value, f"row {index}") for value in row))
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_csv(data: bytes) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Read a file written by :func:`csv_bytes` back into its header and string rows.

    The reader is strict: it rejects non-ASCII bytes, ``\\r``, a missing final newline, blank
    lines and ragged rows, so a solver never silently reads a malformed corpus.
    """
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise CanonicalError(f"CSV is not ASCII: {error}") from error
    if "\r" in text:
        raise CanonicalError("CSV must use \\n line endings")
    if not text.endswith("\n"):
        raise CanonicalError("CSV must end with a newline")
    lines = text[:-1].split("\n")
    header = tuple(lines[0].split(","))
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        cells = tuple(line.split(","))
        if len(cells) != len(header) or not line:
            raise CanonicalError(f"line {number} does not match the {len(header)}-column header")
        rows.append(cells)
    return header, tuple(rows)


def _plain_json(value: Any, where: str) -> Any:
    """``value`` validated and rebuilt from plain ``dict``/``list`` (any Mapping is an object)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    if isinstance(value, float):
        raise CanonicalError(f"{where} is a float; canonical JSON holds exact integers only")
    if isinstance(value, Mapping):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalError(f"{where} has a non-string key {key!r}")
            plain[key] = _plain_json(item, f"{where}.{key}")
        return plain
    if isinstance(value, list | tuple):
        return [_plain_json(item, f"{where}[{index}]") for index, item in enumerate(value)]
    raise CanonicalError(f"{where} has the unsupported type {type(value).__name__}")


def json_bytes(value: Any) -> bytes:
    """The canonical JSON encoding of ``value``: sorted keys, indent 2, ASCII, final newline.

    Any :class:`~collections.abc.Mapping` (a ``MappingProxyType`` included) is written as a
    JSON object and any list or tuple as an array.
    """
    plain = _plain_json(value, "$")
    text = json.dumps(plain, sort_keys=True, indent=2, ensure_ascii=True, separators=(",", ": "))
    return (text + "\n").encode("ascii")


def text_bytes(text: str) -> bytes:
    """``text`` as ASCII with ``\\n`` line endings and exactly one final newline."""
    if not text.isascii():
        raise CanonicalError("text must be ASCII")
    if "\r" in text:
        raise CanonicalError("text must use \\n line endings")
    return (text.rstrip("\n") + "\n").encode("ascii")


def sha256_hex(data: bytes) -> str:
    """The SHA-256 digest of ``data`` as 64 lowercase hex characters."""
    return hashlib.sha256(data).hexdigest()


def _check_path(path: object) -> str:
    if not isinstance(path, str) or not path or not path.isascii():
        raise CanonicalError(f"file names must be non-empty ASCII strings, got {path!r}")
    parts = path.split("/")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in parts) or "\\" in path:
        raise CanonicalError(f"{path!r} is not a normalized relative POSIX path")
    return path


def tree_digest(files: Mapping[str, bytes]) -> str:
    """SHA-256 over a set of named files, independent of mapping order.

    Each file contributes its path, its length and its bytes, in sorted path order, so no
    two different file sets share an encoding.
    """
    digest = hashlib.sha256()
    for path in sorted(_check_path(path) for path in files):
        content = files[path]
        if not isinstance(content, bytes):
            raise CanonicalError(f"{path!r} must map to bytes")
        name = path.encode("ascii")
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()
