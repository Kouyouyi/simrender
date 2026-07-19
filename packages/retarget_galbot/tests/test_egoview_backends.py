from __future__ import annotations

import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from retarget_galbot.egoview.propainter import (
    ProPainterRunner,
    bundled_propainter_root,
)
from retarget_galbot.egoview.render import _sam2_autocast_context
from retarget_galbot.egoview.sam2_segment import (
    bundled_sam2_root,
    default_checkpoint_path,
    resolve_checkpoint,
)


def test_bundled_runtime_sources_and_licenses_exist() -> None:
    sam_root = bundled_sam2_root()
    propainter_root = bundled_propainter_root()

    assert (sam_root / "sam2" / "build_sam.py").exists()
    assert (sam_root / "sam2" / "configs" / "sam2.1").is_dir()
    assert (sam_root / "LICENSE").exists()
    assert (propainter_root / "inference_propainter.py").exists()
    assert (propainter_root / "model" / "propainter.py").exists()
    assert "non-commercial" in (propainter_root / "LICENSE").read_text()
    assert (sam_root.parent / "README.md").exists()


def test_checkpoint_resolution_precedence(tmp_path: Path, monkeypatch) -> None:
    explicit = tmp_path / "explicit.pt"
    environment = tmp_path / "environment.pt"
    explicit.touch()
    environment.touch()
    monkeypatch.setenv("SAM2_CHECKPOINT", str(environment))

    assert resolve_checkpoint(explicit) == explicit.resolve()
    assert resolve_checkpoint() == environment.resolve()
    assert default_checkpoint_path().name == "sam2.1_hiera_base_plus.pt"


def test_propainter_defaults_to_bundled_source(monkeypatch) -> None:
    monkeypatch.delenv("PROPAINTER_ROOT", raising=False)
    runner = ProPainterRunner(python_executable="/fake/python")

    assert runner.root == bundled_propainter_root().resolve()


def test_propainter_command_uses_selected_interpreter(tmp_path: Path) -> None:
    root = tmp_path / "ProPainter"
    root.mkdir()
    (root / "inference_propainter.py").touch()
    runner = ProPainterRunner(root=root, python_executable="/fake/python")
    runner.use_fp16 = False

    command = runner._command(
        tmp_path / "frames",
        tmp_path / "masks",
        tmp_path / "results",
        (2, 48, 64, 3),
    )

    assert command[:2] == ["/fake/python", str(root / "inference_propainter.py")]
    assert command[command.index("--height") + 1] == "48"
    assert command[command.index("--width") + 1] == "64"
    assert "--fp16" not in command


def test_propainter_returns_copy_for_empty_mask() -> None:
    runner = ProPainterRunner(python_executable="/fake/python")
    frames = np.arange(2 * 4 * 6 * 3, dtype=np.uint8).reshape(2, 4, 6, 3)
    masks = np.zeros((2, 4, 6), dtype=bool)

    output = runner.run(frames, masks)

    np.testing.assert_array_equal(output, frames)
    assert output is not frames


def test_cpu_sam2_autocast_is_noop() -> None:
    assert isinstance(_sam2_autocast_context("cpu"), nullcontext)


def test_hand_removal_cli_help() -> None:
    script = (
        bundled_sam2_root().parents[1]
        / "packages"
        / "retarget_galbot"
        / "scripts"
        / "run_hand_removal.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--skip_sam" in result.stdout
    assert "--skip_inpaint" in result.stdout
