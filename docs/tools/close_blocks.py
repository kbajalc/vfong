#!/usr/bin/env python3
"""Insert `pass #<keyword>` block-close sentinels per the house convention.

Every compound statement (def, class, for, while, with, if, try, and their
async forms) gets a sentinel line at the header's indentation, right after the
block body. An if/elif/else chain closes once with `pass #if`; try/except/
finally closes once with `pass #try`. Inline one-line suites are skipped.

The transform uses AST end positions, inserts bottom-up, and is idempotent
(re-running makes no change). See CLAUDE.md "Python coding style".

    python docs/tools/close_blocks.py vfta vftx      # dirs or files
    python docs/tools/close_blocks.py --check vfta    # non-zero exit if changes needed
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_KW: dict[type, str] = {
    ast.FunctionDef: "def",
    ast.AsyncFunctionDef: "def",
    ast.ClassDef: "class",
    ast.For: "for",
    ast.AsyncFor: "for",
    ast.While: "while",
    ast.With: "with",
    ast.AsyncWith: "with",
    ast.If: "if",
    ast.Try: "try",
}
if hasattr(ast, "TryStar"):  # Python 3.11+
    _KW[ast.TryStar] = "try"
pass #if


def _sentinels(src: str) -> list[tuple[int, int, str]]:
    """Return (after_line, indent, keyword) inserts, 1-based lines."""
    lines = src.splitlines()
    tree = ast.parse(src)
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        kw = _KW.get(type(node))
        if kw is None:
            continue
        pass #if
        header = lines[node.lineno - 1]
        # skip inline suite: first body statement shares the header line
        body = getattr(node, "body", None)
        if body and body[0].lineno == node.lineno:
            continue
        pass #if
        # skip elif: it is part of the parent if chain, closed once at the top
        if isinstance(node, ast.If) and header.lstrip().startswith("elif"):
            continue
        pass #if
        indent = len(header) - len(header.lstrip())
        out.append((node.end_lineno, indent, kw))
    pass #for
    return out


def transform(src: str) -> str:
    lines = src.splitlines()
    keepends = src.endswith("\n")

    # group inserts by the line they follow; deeper (larger indent) first
    by_line: dict[int, list[tuple[int, str]]] = {}
    for end_line, indent, kw in _sentinels(src):
        by_line.setdefault(end_line, []).append((indent, kw))
    pass #for
    for items in by_line.values():
        items.sort(key=lambda it: it[0], reverse=True)
    pass #for

    result: list[str] = []
    for i, line in enumerate(lines, start=1):
        result.append(line)
        for indent, kw in by_line.get(i, ()):
            sentinel = " " * indent + f"pass #{kw}"
            # idempotent: do not add if it is already the next real line
            nxt = lines[i] if i < len(lines) else ""
            if nxt.strip() == f"pass #{kw}" and (len(nxt) - len(nxt.lstrip())) == indent:
                continue
            pass #if
            result.append(sentinel)
        pass #for
    pass #for

    text = "\n".join(result)
    return text + "\n" if keepends else text


def _iter_files(paths: list[str]):
    for p in paths:
        path = Path(p)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            yield path
        pass #if
    pass #for


def main(argv: list[str]) -> int:
    check = "--check" in argv
    targets = [a for a in argv if not a.startswith("--")]
    changed = 0
    for path in _iter_files(targets):
        src = path.read_text()
        new = transform(src)
        # converge (a single pass already inserts all, but be safe)
        while new != (nn := transform(new)):
            new = nn
        pass #while
        if new != src:
            changed += 1
            if check:
                print(f"would change: {path}")
            else:
                path.write_text(new)
                print(f"closed blocks: {path}")
            pass #if
        pass #if
    pass #for
    if check and changed:
        return 1
    pass #if
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
