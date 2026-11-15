"""Pure-helper tests for scripts/worker.py (the local run-queue worker).

Only the pure helpers are tested: output-name derivation, the recreate subprocess
command, and the stderr tail. The claim/sync SQL is I/O and stays untested by design.
"""
import importlib.util
import pathlib
import sys


def _load_worker():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "worker.py"
    spec = importlib.util.spec_from_file_location("reforge_worker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()


# --- output_path_for ----------------------------------------------------------

def test_output_path_for_derives_stem_and_run_id():
    assert worker.output_path_for("reference/DISCIPLINE.mp4", 7) == "output/DISCIPLINE_run7.mp4"


def test_output_path_for_bare_filename():
    assert worker.output_path_for("grace.mp4", 12) == "output/grace_run12.mp4"


# --- recreate_cmd ---------------------------------------------------------------

def test_recreate_cmd_invokes_recreate_as_subprocess():
    cmd = worker.recreate_cmd("reference/grace.mp4", "output/grace_run3.mp4")
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/recreate.py")
    assert cmd[cmd.index("--reference") + 1] == "reference/grace.mp4"
    assert cmd[cmd.index("--out") + 1] == "output/grace_run3.mp4"


# --- parse_rerender_key -----------------------------------------------------------

def test_parse_rerender_key_extracts_reel_id():
    assert worker.parse_rerender_key("rerender:8") == 8
    assert worker.parse_rerender_key("rerender:123") == 123


def test_parse_rerender_key_none_for_normal_references():
    assert worker.parse_rerender_key("reference/DISCIPLINE.mp4") is None
    assert worker.parse_rerender_key("") is None
    assert worker.parse_rerender_key(None) is None


def test_parse_rerender_key_none_for_malformed_ids():
    assert worker.parse_rerender_key("rerender:") is None
    assert worker.parse_rerender_key("rerender:abc") is None
    assert worker.parse_rerender_key("rerender:8x") is None


# --- rerender_cmd -----------------------------------------------------------------

def test_rerender_cmd_invokes_rerender_script():
    cmd = worker.rerender_cmd(8)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/rerender_reel.py")
    assert cmd[cmd.index("--reel-id") + 1] == "8"


# --- parse_repick_key ---------------------------------------------------------------

def test_parse_repick_key_extracts_slot_id():
    assert worker.parse_repick_key("repick:71") == 71


def test_parse_repick_key_none_for_other_keys():
    assert worker.parse_repick_key("rerender:8") is None
    assert worker.parse_repick_key("reference/DISCIPLINE.mp4") is None
    assert worker.parse_repick_key("repick:") is None
    assert worker.parse_repick_key("repick:x1") is None
    assert worker.parse_repick_key(None) is None


def test_rerender_key_is_not_a_repick_key():
    assert worker.parse_rerender_key("repick:71") is None


# --- repick_cmd -----------------------------------------------------------------

def test_repick_cmd_invokes_repick_script():
    cmd = worker.repick_cmd(71)
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/repick_slot.py")
    assert cmd[cmd.index("--slot-id") + 1] == "71"


# --- error_tail -----------------------------------------------------------------

def test_error_tail_strips_and_keeps_last_chars():
    assert worker.error_tail("  boom  ") == "boom"
    assert worker.error_tail("x" * 3000, limit=100) == "x" * 100


def test_error_tail_handles_empty():
    assert worker.error_tail("") == ""
    assert worker.error_tail(None) == ""
