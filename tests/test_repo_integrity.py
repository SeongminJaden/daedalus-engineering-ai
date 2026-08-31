"""Repository integrity: every source file must actually be committable.

This exists because of a real failure. `.gitignore` carried unanchored
artifact patterns like `datasets/`, which git matches at ANY depth - so the
genuine source package `surrogate/datasets/` was silently excluded. The commit
looked clean, the push succeeded, and the pushed tree could not be imported at
all.

Nothing in the normal test suite catches that, because the working tree is
complete; only a fresh clone is broken. These tests check the property
directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories that are legitimately ignored artifacts, checked from the root.
ARTIFACT_ROOTS = {".venv", ".git", "runs", "logs", "datasets", "checkpoints",
                  "outputs", "artifacts", "__pycache__", ".pytest_cache",
                  ".ruff_cache", ".mypy_cache"}


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists()


def _source_files() -> list[Path]:
    """Every .py and .yaml under the repo that is real source."""
    out = []
    for path in REPO_ROOT.rglob("*"):
        if path.is_dir() or path.suffix not in (".py", ".yaml", ".md"):
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in ARTIFACT_ROOTS for part in rel.parts):
            continue
        # data/ holds artifacts except the committed material database
        if rel.parts[0] == "data" and rel.name != "materials.yaml":
            continue
        out.append(rel)
    return out


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_no_source_file_is_gitignored():
    """The bug that motivated this file: a source package matched an artifact
    pattern and vanished from the commit."""
    files = _source_files()
    assert files, "found no source files to check"

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *[str(f) for f in files]],
        cwd=REPO_ROOT, capture_output=True, text=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these SOURCE files are excluded by .gitignore and would be missing "
        f"from a fresh clone:\n  " + "\n  ".join(ignored)
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_package_directory_has_an_init():
    """A package dir without __init__.py imports differently (or not at all)
    depending on sys.path, which is the kind of thing that only breaks for
    someone else."""
    missing = []
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in ARTIFACT_ROOTS for part in rel.parts):
            continue
        parent = path.parent
        if parent == REPO_ROOT or parent.name in ("scripts", "tests"):
            continue
        if not (parent / "__init__.py").exists():
            missing.append(str(parent.relative_to(REPO_ROOT)))
    assert not missing, f"package directories without __init__.py: {sorted(set(missing))}"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_artifact_directories_are_still_ignored():
    """The fix must not have gone too far the other way - large run artifacts
    must stay out of the repository."""
    for candidate in ("runs/x.jsonl", "runs/brain.sqlite3", "datasets/x.npz",
                      "checkpoints/m.pt", ".venv/lib/python3.10/site-packages/x.py"):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", candidate],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"{candidate} is NOT ignored but should be"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_committed_tree_can_import_every_package():
    """Import each top-level package from a checkout of HEAD, in a subprocess.

    This is the check that would have caught the original failure: the working
    tree imported fine, but the committed tree was missing a package.
    """
    import tempfile

    # Every package AND subpackage. Importing only top-level names would not
    # have caught the original bug: `surrogate/__init__.py` is a bare
    # docstring, so `import surrogate` succeeded while surrogate.datasets was
    # missing entirely.
    packages = sorted(
        ".".join(d.relative_to(REPO_ROOT).parts)
        for d in REPO_ROOT.rglob("__init__.py")
        for d in [d.parent]
        if not any(part in ARTIFACT_ROOTS for part in
                   d.relative_to(REPO_ROOT).parts)
        and d != REPO_ROOT
    )
    assert packages, "found no packages to import"

    with tempfile.TemporaryDirectory() as tmp:
        export = subprocess.run(
            ["git", "worktree", "add", "--detach", tmp, "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if export.returncode != 0:
            pytest.skip(f"could not create worktree: {export.stderr}")
        try:
            script = (
                "import sys\n"
                f"sys.path.insert(0, {tmp!r})\n"
                "import importlib\n"
                f"for name in {packages!r}:\n"
                "    importlib.import_module(name)\n"
                "print('OK')\n"
            )
            proc = subprocess.run([sys.executable, "-c", script],
                                  capture_output=True, text=True, cwd=tmp)
            assert proc.returncode == 0, (
                "the COMMITTED tree cannot be imported (the working tree can) - "
                f"a source file is missing from the commit:\n{proc.stderr}"
            )
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", tmp],
                           cwd=REPO_ROOT, capture_output=True)
