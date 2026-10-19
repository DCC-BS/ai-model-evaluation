import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_public_safety.py"


@pytest.fixture
def denylist(tmp_path):
    path = tmp_path / "denylist.txt"
    path.write_text("Testperson Muster\n", encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    return root


def run(root, denylist):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), "--denylist", str(denylist)],
        capture_output=True,
        text=True,
    )


def test_flags_denylisted_name(tree, denylist):
    f = tree / "notes.md"
    f.write_text("Interview with testperson MUSTER today\n", encoding="utf-8")
    r = run(tree, denylist)
    assert r.returncode == 1
    assert "notes.md" in r.stdout


def test_flags_denylisted_name_in_path(tree, denylist):
    (tree / "Testperson Muster").mkdir()
    (tree / "Testperson Muster" / "a.txt").write_text("x\n", encoding="utf-8")
    r = run(tree, denylist)
    assert r.returncode == 1


def test_flags_internal_host(tree, denylist):
    (tree / "cfg.txt").write_text("http://llm." + "intern.example." + "bs.ch/v1\n", encoding="utf-8")
    r = run(tree, denylist)
    assert r.returncode == 1
    assert "cfg.txt" in r.stdout


@pytest.mark.parametrize("text", ["10." + "1.2.3", "192." + "168.0.1", "box." + "local", "a." + "intern"])
def test_flags_other_internal_patterns(tree, denylist, text):
    (tree / "cfg.txt").write_text(f"host {text}\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_allows_public_bs_hosts(tree, denylist):
    (tree / "doc.md").write_text(
        "https://data.bs.ch and https://www.bs.ch and https://statistik.bs.ch and localhost\n",
        encoding="utf-8",
    )
    assert run(tree, denylist).returncode == 0


def test_flags_secret(tree, denylist):
    (tree / ".env.example").write_text("HF_TOKEN=" + "hf_" + "a" * 34 + "\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_flags_result_artifact(tree, denylist):
    (tree / "ocr_evaluation_results.csv").write_text("", encoding="utf-8")
    r = run(tree, denylist)
    assert r.returncode == 1
    assert "ocr_evaluation_results.csv" in r.stdout


def test_flags_sanity_checks_dir(tree, denylist):
    (tree / "sanity_checks").mkdir()
    (tree / "sanity_checks" / "x.txt").write_text("", encoding="utf-8")
    r = run(tree, denylist)
    assert r.returncode == 1
    assert "sanity_checks" in r.stdout


def test_flags_audio_outside_fixtures_only(tree, denylist):
    (tree / "tests" / "fixtures").mkdir(parents=True)
    (tree / "tests" / "fixtures" / "ok.wav").write_bytes(b"RIFF")
    assert run(tree, denylist).returncode == 0
    (tree / "clip.mp3").write_bytes(b"ID3")
    assert run(tree, denylist).returncode == 1


def test_missing_denylist_fails_closed(tree, tmp_path):
    (tree / "README.md").write_text("plain\n", encoding="utf-8")
    r = run(tree, tmp_path / "nope.txt")
    assert r.returncode == 2
    assert r.stderr.strip()


def test_single_file_root(tmp_path, denylist):
    f = tmp_path / "one.txt"
    f.write_text("key sk-" + "b" * 24 + "\n", encoding="utf-8")
    r = run(f, denylist)
    assert r.returncode == 1
    assert "one.txt" in r.stdout


def test_clean_tree_passes(tree, denylist):
    (tree / "README.md").write_text("# Plain readme\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 0


def test_leaky_fixtures_skipped_in_tree_scan_but_flagged_directly(tree, denylist):
    leaky = tree / "tests" / "fixtures" / "leaky"
    leaky.mkdir(parents=True)
    (leaky / "h.txt").write_text("http://llm." + "intern.example." + "bs.ch/v1\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 0
    assert run(leaky, denylist).returncode == 1


def test_empty_denylist_fails_closed(tree, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    assert run(tree, empty).returncode == 2


def test_non_utf8_denylist_fails_closed(tree, tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad\n")
    assert run(tree, bad).returncode == 2


def test_nonexistent_root_fails_closed(tmp_path, denylist):
    assert run(tmp_path / "missing", denylist).returncode == 2


def test_flags_generic_api_key(tree, denylist):
    (tree / "c.py").write_text("api_key = " + "A1b2C3d4" * 3 + "\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_flags_subdomain_of_public_host(tree, denylist):
    (tree / "c.txt").write_text("https://foo." + "data.bs.ch\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_flags_underscore_host(tree, denylist):
    (tree / "c.txt").write_text("https://my_" + "data.bs.ch\n", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_artifact_dir_case_insensitive(tree, denylist):
    (tree / "Predictions").mkdir()
    (tree / "Predictions" / "a.txt").write_text("", encoding="utf-8")
    assert run(tree, denylist).returncode == 1


def test_audio_allowed_in_nested_fixtures(tree, denylist):
    d = tree / "pkg" / "tests" / "fixtures"
    d.mkdir(parents=True)
    (d / "a.wav").write_bytes(b"RIFF")
    assert run(tree, denylist).returncode == 0


def git(tree, *args):
    subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True)


def test_git_tree_ignores_gitignored_files(tree, denylist):
    git(tree, "init", "-q")
    (tree / ".gitignore").write_text("data/\n", encoding="utf-8")
    (tree / "data").mkdir()
    (tree / "data" / "x.wav").write_bytes(b"RIFF")
    assert run(tree, denylist).returncode == 0
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(tree), "--denylist", str(denylist), "--all-files"],
        capture_output=True, text=True,
    )
    assert r.returncode == 1


def test_git_tree_flags_untracked_non_ignored(tree, denylist):
    git(tree, "init", "-q")
    (tree / ".gitignore").write_text("data/\n", encoding="utf-8")
    (tree / "clip.wav").write_bytes(b"RIFF")
    r = run(tree, denylist)
    assert r.returncode == 1
    assert "clip.wav" in r.stdout
