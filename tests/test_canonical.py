"""The canonical writers produce one byte sequence per value and reject what is not exact."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from trapforge.canonical import (
    CanonicalError,
    csv_bytes,
    json_bytes,
    parse_csv,
    sha256_hex,
    text_bytes,
    tree_digest,
)

cells = st.one_of(
    st.integers(min_value=-(10**30), max_value=10**30),
    st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=',"'),
        min_size=1,
        max_size=8,
    ),
)


@given(st.integers(1, 5).flatmap(lambda w: st.lists(st.lists(cells, min_size=w, max_size=w))))
def test_csv_round_trips_through_the_strict_reader(rows: list[list[int | str]]) -> None:
    header = [f"c{index}" for index in range(len(rows[0]))] if rows else ["c0"]
    rows = [row for row in rows if len(row) == len(header)]
    data = csv_bytes(header, rows)
    parsed_header, parsed_rows = parse_csv(data)
    assert parsed_header == tuple(header)
    assert parsed_rows == tuple(tuple(str(cell) for cell in row) for row in rows)
    assert data.endswith(b"\n")
    assert b"\r" not in data


def test_csv_layout_is_exact() -> None:
    assert csv_bytes(["a", "b"], []) == b"a,b\n"
    assert csv_bytes(["a", "b"], [(1, "x"), (-2, "y")]) == b"a,b\n1,x\n-2,y\n"


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        ([], [], "at least one column"),
        (["a"], [[1, 2]], "row 0 has 2 cells"),
        (["a"], [[1.5]], "cells must be int or str"),
        (["a"], [[True]], "cells must be int or str"),
        (["a"], [["x,y"]], "would need CSV quoting"),
        (["a"], [['say "hi"']], "would need CSV quoting"),
        (["a"], [[" x"]], "surrounding whitespace"),
        (["a"], [["caf\u00e9"]], "not printable ASCII"),
        (["a\nb"], [], "not printable ASCII"),
    ],
)
def test_csv_rejects_values_that_are_not_canonical(
    header: list[str], rows: list[list[object]], message: str
) -> None:
    with pytest.raises(CanonicalError, match=message):
        csv_bytes(header, rows)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"a\n\xff\n", "not ASCII"),
        (b"a\r\n1\r\n", "line endings"),
        (b"a\n1", "end with a newline"),
        (b"a,b\n1\n", "line 2 does not match"),
        (b"a\n\n", "line 2 does not match"),
    ],
)
def test_parse_csv_is_strict(data: bytes, message: str) -> None:
    with pytest.raises(CanonicalError, match=message):
        parse_csv(data)


def test_json_is_sorted_indented_and_integer_only() -> None:
    assert json_bytes({"z": 1, "a": [True, None, "s"]}) == (
        b'{\n  "a": [\n    true,\n    null,\n    "s"\n  ],\n  "z": 1\n}\n'
    )
    assert json_bytes({"t": (1, 2)}) == json_bytes({"t": [1, 2]})
    assert json_bytes({"u": "\u00e9"}) == b'{\n  "u": "\\u00e9"\n}\n'
    with pytest.raises(CanonicalError, match=r"\$\.a\[1\] is a float"):
        json_bytes({"a": [1, 2.0]})
    with pytest.raises(CanonicalError, match="non-string key"):
        json_bytes({1: 2})
    with pytest.raises(CanonicalError, match="unsupported type set"):
        json_bytes({"a": {1}})


def test_text_has_exactly_one_final_newline() -> None:
    assert text_bytes("hi") == b"hi\n"
    assert text_bytes("hi\n\n\n") == b"hi\n"
    with pytest.raises(CanonicalError, match="ASCII"):
        text_bytes("\u00e9")
    with pytest.raises(CanonicalError, match="line endings"):
        text_bytes("a\r\nb")


def test_sha256_matches_a_known_vector() -> None:
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_tree_digest_ignores_mapping_order_but_not_names_or_bytes() -> None:
    files = {"b/x.csv": b"1\n", "a.txt": b"2\n"}
    digest = tree_digest(files)
    assert digest == tree_digest(dict(reversed(files.items())))
    assert digest != tree_digest({"b/y.csv": b"1\n", "a.txt": b"2\n"})
    assert digest != tree_digest({"b/x.csv": b"1\n", "a.txt": b"3\n"})
    # Framing: moving bytes between a name and its content changes the digest.
    assert tree_digest({"ab": b"c"}) != tree_digest({"a": b"bc"})
    assert tree_digest({}) == sha256_hex(b"")


@pytest.mark.parametrize("path", ["", "/abs", "a//b", "a/../b", "./a", "a\\b", "\u00e9"])
def test_tree_digest_rejects_unnormalized_paths(path: str) -> None:
    with pytest.raises(CanonicalError, match=r"path|ASCII"):
        tree_digest({path: b""})


def test_tree_digest_requires_bytes() -> None:
    with pytest.raises(CanonicalError, match="must map to bytes"):
        tree_digest({"a": "text"})  # type: ignore[dict-item]
