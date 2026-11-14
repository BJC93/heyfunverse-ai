"""Safe file writing: never silently overwrite project data.

Every tool that replaces an existing file (a story, a scene plan, an image)
goes through these helpers. They

* do nothing when the new content is identical to what is already there,
* copy the existing file into storage/backups/ before replacing it, and
* write through a temporary file so a crash cannot leave half a file behind.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


def _default_backup_dir():
    import config

    return config.BACKUP_DIR


def backup_file(path, backup_dir=None):
    """Copy ``path`` into the backup folder. Returns the backup path, or None
    if ``path`` does not exist. Never overwrites an earlier backup."""

    path = Path(path)

    if not path.is_file():
        return None

    folder = Path(backup_dir) if backup_dir else _default_backup_dir()
    folder.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = folder / f"{path.stem}.{stamp}{path.suffix}"

    counter = 1
    while candidate.exists():
        candidate = folder / f"{path.stem}.{stamp}-{counter}{path.suffix}"
        counter += 1

    shutil.copy2(path, candidate)

    return candidate


def write_text_safely(
    path,
    text,
    *,
    backup=True,
    backup_dir=None,
    encoding="utf-8"
):
    """Write ``text`` to ``path``.

    Returns {"path", "written", "backup"}. ``written`` is False when the
    existing file already holds exactly this text.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result = {"path": path, "written": False, "backup": None}

    if path.is_file():
        try:
            if path.read_text(encoding=encoding) == text:
                return result
        except UnicodeDecodeError:
            pass

        if backup:
            result["backup"] = backup_file(path, backup_dir)

    handle_fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp"
    )

    try:
        with os.fdopen(handle_fd, "w", encoding=encoding, newline="\n") as handle:
            handle.write(text)

        os.replace(temp_name, path)

    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

    result["written"] = True

    return result


def write_json_safely(path, data, *, backup=True, backup_dir=None, indent=2):
    """JSON version of :func:`write_text_safely` (UTF-8, keeps Telugu readable)."""

    text = json.dumps(data, indent=indent, ensure_ascii=False) + "\n"

    return write_text_safely(
        path, text, backup=backup, backup_dir=backup_dir
    )
