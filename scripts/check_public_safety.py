#!/usr/bin/env python3
"""Scan a tree (or single file) for content that must not be published.

Exit codes: 0 clean, 1 findings, 2 usage/configuration error (fails closed).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_DENYLIST = Path.home() / ".config" / "ai-model-evaluation" / "denylist.txt"

SKIP_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", ".ruff_cache", ".mypy_cache"}
SKIP_FILES = {"uv.lock"}
LEAKY_SUFFIX = ("tests", "fixtures", "leaky")

ALLOWED_BS_HOSTS = {"data.bs.ch", "www.bs.ch", "statistik.bs.ch"}
BS_HOST = re.compile(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\.bs\.ch\b|\.bs\.ch\b", re.I)

LINE_PATTERNS = [
    (re.compile(r"\b10\.\d+\.\d+\.\d+\b"), "private IPv4 range, class A"),
    (re.compile(r"\b192\.168\."), "private IPv4 range, class C"),
    (re.compile(r"\." + "intern" + r"\b", re.I), "internal hostname suffix (intern)"),
    (re.compile(r"\." + "local" + r"\b", re.I), "internal hostname suffix (local)"),
    (
        re.compile(r"(?i)(api[_-]?key|token|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        "possible secret assignment",
    ),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "possible Hugging Face token"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "possible API key (sk-)"),
]

ARTIFACT_NAME_PATTERNS = [
    (re.compile(r"results.*\.(csv|png)$", re.I), "result artefact"),
]
ARTIFACT_DIRS = {"predictions", "sanity_checks", "eval_results"}
AUDIO_EXT = {".wav", ".mp3", ".m4a"}


def load_denylist(path):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: cannot read denylist {path}: {e}", file=sys.stderr)
        return None
    terms = [t.strip().lower() for t in text.splitlines() if t.strip()]
    if not terms:
        print(f"error: denylist {path} is empty", file=sys.stderr)
        return None
    return terms


def is_leaky_dir(p):
    return p.parts[-3:] == LEAKY_SUFFIX


def under_leaky(path):
    parts = path.parts
    return any(parts[i : i + 3] == LEAKY_SUFFIX for i in range(len(parts) - 2))


def git_files(root):
    """Tracked + untracked-but-not-ignored files under root, or None if not in a git tree."""
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True,
        )
    except OSError:
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    r = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
    )
    if r.returncode != 0:
        raise OSError(f"git ls-files failed: {r.stderr.decode(errors='replace').strip()}")
    names = [n for n in r.stdout.decode("utf-8", errors="surrogateescape").split("\0") if n]
    return [root / n for n in names]


def iter_files(root, all_files=False, errors=None):
    root = root.resolve()
    if root.is_file():
        yield root, root.parent
        return
    inside_leaky = under_leaky(root)
    listed = None if all_files else git_files(root)
    if listed is not None:
        for p in listed:
            rel_parts = p.relative_to(root).parts
            if not (p.is_symlink() or p.exists()):
                continue  # deleted in working tree
            if any(part in SKIP_DIRS for part in rel_parts[:-1]) or p.name in SKIP_FILES:
                continue
            if under_leaky(p.relative_to(root)) and not inside_leaky:
                continue
            yield p, root
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir())
        except OSError as e:
            if errors is not None:
                errors.append(f"{d}:0: unreadable directory ({e})")
            continue
        for p in entries:
            if p.is_symlink():
                yield p, root  # reported, not followed
                continue
            if p.is_dir():
                if p.name in SKIP_DIRS:
                    continue
                if is_leaky_dir(p) and not inside_leaky:
                    continue
                stack.append(p)
            else:
                if p.name in SKIP_FILES:
                    continue
                yield p, root


def check_path(rel, terms):
    hits = []
    low = rel.as_posix().lower()
    for t in terms:
        if t in low:
            hits.append("denylisted term in path")
            break
    parts = rel.parts
    if any(part.lower() in ARTIFACT_DIRS for part in parts[:-1]):
        hits.append("result artefact directory")
    for rx, why in ARTIFACT_NAME_PATTERNS:
        if rx.search(rel.name):
            hits.append(why)
    if rel.suffix.lower() in AUDIO_EXT and not any(parts[i : i + 2] == ("tests", "fixtures") for i in range(len(parts) - 1)):
        hits.append("audio file outside tests/fixtures")
    return hits


def check_text(text, terms):
    for n, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for t in terms:
            if t in low:
                yield n, "denylisted term"
                break
        for m in BS_HOST.finditer(line):
            if m.group(0).lower() not in ALLOWED_BS_HOSTS:
                yield n, "internal canton host"
        for rx, why in LINE_PATTERNS:
            if rx.search(line):
                yield n, why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--denylist", default=str(DEFAULT_DENYLIST))
    ap.add_argument(
        "--all-files", action="store_true",
        help="walk every file, including gitignored ones (default inside a git tree: tracked + untracked-not-ignored)",
    )
    args = ap.parse_args(argv)

    terms = load_denylist(args.denylist)
    if terms is None:
        return 2
    root = Path(args.root)
    if not root.exists():
        print(f"error: root {root} does not exist", file=sys.stderr)
        return 2

    findings = []
    try:
        files = list(iter_files(root, args.all_files, findings))
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for path, base in files:
        try:
            rel = path.relative_to(base)
        except ValueError:
            rel = Path(path.name)
        shown = path.relative_to(Path.cwd().resolve()) if path.is_relative_to(Path.cwd().resolve()) else path
        if root.resolve().is_file():
            rel = Path(path.name)
        else:
            for why in check_path(rel, terms):
                findings.append(f"{shown}:0: {why}")
        if path.is_symlink():
            findings.append(f"{shown}:0: symlink not scanned")
            continue
        try:
            data = path.read_bytes()
        except OSError as e:
            findings.append(f"{shown}:0: unreadable ({e})")
            continue
        text = data.decode("utf-8", errors="replace")
        for n, why in check_text(text, terms):
            findings.append(f"{shown}:{n}: {why}")

    for f in findings:
        print(f)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
