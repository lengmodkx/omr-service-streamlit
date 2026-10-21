import os
import subprocess
import time

import pytest

from omr_service.engine.crop_cleaner import clean_expired_crops


def _set_mtime(path, timestamp):
    os.utime(path, (timestamp, timestamp))


def test_should_delete_files_older_than_retention(tmp_path):
    now = time.time()
    nested = tmp_path / "batch" / "question"
    nested.mkdir(parents=True)
    old_file = nested / "old.jpg"
    old_file.write_bytes(b"old")
    _set_mtime(old_file, now - 31 * 86400)

    result = clean_expired_crops(tmp_path, retention_days=30, now=now)

    assert not old_file.exists()
    assert not nested.exists()
    assert result.deleted_files == 1
    assert result.freed_bytes == 3


def test_should_keep_recent_files(tmp_path):
    now = time.time()
    recent_file = tmp_path / "recent.jpg"
    recent_file.write_bytes(b"recent")
    _set_mtime(recent_file, now - 86400)

    result = clean_expired_crops(tmp_path, retention_days=30, now=now)

    assert recent_file.exists()
    assert result.deleted_files == 0


def test_should_skip_when_disabled(tmp_path):
    now = time.time()
    old_file = tmp_path / "old.jpg"
    old_file.write_bytes(b"old")
    _set_mtime(old_file, now - 365 * 86400)

    result = clean_expired_crops(tmp_path, retention_days=0, now=now)

    assert old_file.exists()
    assert result.deleted_files == 0


def test_should_not_follow_symlink_escape(tmp_path):
    now = time.time()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.jpg"
    outside_file.write_bytes(b"outside")
    _set_mtime(outside_file, now - 365 * 86400)
    link = output_dir / "linked-dir"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            pytest.skip(f"当前系统不允许创建符号链接: {exc}")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside_dir)],
            check=True,
            capture_output=True,
        )

    clean_expired_crops(output_dir, retention_days=30, now=now)

    assert outside_file.exists()
    assert os.path.lexists(link)
