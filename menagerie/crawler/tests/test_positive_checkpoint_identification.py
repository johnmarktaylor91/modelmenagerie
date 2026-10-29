"""Positive checkpoint identification: the in-child read tripwire poisons on PROOF.

BL-12(g) / AREA_WORKER opus W-2 / catalog C-25: every ``policy/checkpoint-read``
terminal in the 72-model pilot was a benign read (package metadata, ICU data,
``/etc/os-release`` probes) -- not one was a checkpoint or a weight. The category
fix is positive identification: the poisoning flag is set only when the refused
target is provably model data (:func:`_runtime_model_data_path`); every other
refused read lands in the NON-poisoning ``refused_read_probe_paths`` bucket and
raises an ordinary ``OSError`` a library probe's own ``try/except`` absorbs.

These tests are the permanent tripwire for the recurrence class: a fixture that
fails on a DIFFERENT instance of the same predicate (an unforeseen basename),
not just the six pilot paths that were each patched by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.policy import (
    ExecutionPolicy,
    PolicyViolation,
    _runtime_model_data_path,
)
from menagerie.crawler.worker_supervisor import _read_path_is_allowed


def _policy(tmp_path: Path, **kwargs: object) -> ExecutionPolicy:
    """Return one inactive policy rooted in a fresh scratch directory."""

    return ExecutionPolicy(tmp_path / "scratch", **kwargs)  # type: ignore[arg-type]


@pytest.mark.smoke
def test_unforeseen_benign_basename_refuses_without_poison(tmp_path: Path) -> None:
    """A never-before-seen small text file refuses with OSError and never poisons.

    This is the anti-allowlist fixture: the basename is deliberately one no
    allowlist has ever named, so an enumeration-shaped fix cannot pass it.
    """

    outside = tmp_path / "undeclared"
    outside.mkdir()
    probe = outside / "definitely-not-model.quux"
    probe.write_text("libraryversion=1.2.3\n", encoding="utf-8")
    with _policy(tmp_path) as observation:
        with pytest.raises(OSError):
            open(probe, "r", encoding="utf-8")
    assert observation.checkpoint_or_weight_read_attempted is False
    assert observation.checkpoint_paths == []
    assert str(probe) in observation.refused_read_probe_paths


@pytest.mark.smoke
def test_absent_file_probe_refuses_enoent_without_poison(tmp_path: Path) -> None:
    """Probing a provably absent optional file is never a policy violation.

    The single most common shape in third-party Python: ``try: open(optional)``
    inside the library's own ``except``. No bytes exist, so nothing can have
    been read; the pre-fix tripwire poisoned the attempt anyway (opus W-2
    repro against a provably absent file).
    """

    absent = tmp_path / "undeclared" / "settings.json"
    with _policy(tmp_path) as observation:
        with pytest.raises(FileNotFoundError):
            open(absent, "r", encoding="utf-8")
    assert observation.checkpoint_or_weight_read_attempted is False
    assert str(absent) in observation.refused_read_probe_paths


@pytest.mark.smoke
def test_pilot_benign_shapes_do_not_poison(tmp_path: Path) -> None:
    """The six pilot false-positive shapes all refuse without poisoning."""

    metadata_dir = tmp_path / "site-packages" / "example-1.0.dist-info"
    metadata_dir.mkdir(parents=True)
    shapes = (
        metadata_dir / "SOURCES.txt",
        metadata_dir / "installed-files.txt",
        tmp_path / "site-packages" / "astor" / "NOTAVERSION",
    )
    shapes[-1].parent.mkdir(parents=True, exist_ok=True)
    for path in shapes:
        path.write_text("plain text metadata\n", encoding="utf-8")
    with _policy(tmp_path) as observation:
        for path in shapes:
            try:
                handle = open(path, "r", encoding="utf-8")
            except OSError:
                continue
            handle.close()
    assert observation.checkpoint_or_weight_read_attempted is False
    assert observation.checkpoint_paths == []


@pytest.mark.smoke
def test_weight_format_suffix_poisons_even_when_absent(tmp_path: Path) -> None:
    """A declared weight-format suffix is positive proof: the ATTEMPT poisons."""

    weights = tmp_path / "undeclared" / "model.safetensors"
    with _policy(tmp_path) as observation:
        with pytest.raises(PolicyViolation, match="model-data read") as caught:
            open(weights, "rb")
    assert caught.value.reason_code == "checkpoint-read"
    assert observation.checkpoint_or_weight_read_attempted is True
    assert str(weights) in observation.checkpoint_paths


@pytest.mark.smoke
def test_tensor_magic_payload_poisons(tmp_path: Path) -> None:
    """An existing extensionless file bearing tensor magic is proven model data."""

    payload = tmp_path / "undeclared" / "blob"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"\x93NUMPY\x01\x00" + b"\x00" * 64)
    assert _runtime_model_data_path(payload) is True
    with _policy(tmp_path) as observation:
        with pytest.raises(PolicyViolation, match="model-data read") as caught:
            open(payload, "rb")
    assert caught.value.reason_code == "checkpoint-read"
    assert observation.checkpoint_or_weight_read_attempted is True


@pytest.mark.smoke
def test_refused_probe_paths_survive_to_dict(tmp_path: Path) -> None:
    """The non-poisoning bucket is part of the serialized policy observation."""

    absent = tmp_path / "undeclared" / "probe.txt"
    with _policy(tmp_path) as observation:
        with pytest.raises(OSError):
            open(absent, "r", encoding="utf-8")
    serialized = observation.to_dict()
    assert serialized["refused_read_probe_paths"] == [str(absent)]
    assert serialized["checkpoint_or_weight_read_attempted"] is False


@pytest.mark.smoke
def test_in_child_tripwire_never_stricter_than_parent_classifier(tmp_path: Path) -> None:
    """Anything the parent kernel-boundary classifier allows, the child allows.

    Drift where the in-child authority is STRICTER than the parent classifier is
    the platform.mac_ver class: the OS sandbox permits the read, the in-child
    hook kills the worker anyway. The corpus below holds the exact pilot
    casualties plus the shared system-file allowlist.
    """

    corpus = (
        "/etc/os-release",
        "/etc/localtime",
        "/etc/locale.alias",
        "/usr/share/locale/locale.alias",
        "/System/Library/CoreServices/SystemVersion.plist",
    )
    policy = _policy(tmp_path)
    for path_text in corpus:
        parent_allows = _read_path_is_allowed(
            path_text,
            tmp_path,
            (tmp_path / "scratch",),
            (),
            (),
        )
        if parent_allows:
            assert policy._read_path_allowed(path_text) is True, path_text
