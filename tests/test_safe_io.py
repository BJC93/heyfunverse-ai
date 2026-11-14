import safe_io


def test_backup_missing_file_returns_none(tmp_path):
    assert safe_io.backup_file(tmp_path / "nope.json", tmp_path / "b") is None


def test_write_new_file_has_no_backup(tmp_path):
    result = safe_io.write_text_safely(tmp_path / "a.txt", "one", backup_dir=tmp_path / "b")
    assert result["written"] and result["backup"] is None
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one"


def test_identical_content_is_a_noop(tmp_path):
    target = tmp_path / "a.txt"
    safe_io.write_text_safely(target, "same", backup_dir=tmp_path / "b")
    result = safe_io.write_text_safely(target, "same", backup_dir=tmp_path / "b")
    assert not result["written"]
    assert not (tmp_path / "b").exists()


def test_overwrite_keeps_old_content_in_backup(tmp_path):
    target = tmp_path / "story.json"
    safe_io.write_text_safely(target, "old", backup_dir=tmp_path / "b")
    result = safe_io.write_text_safely(target, "new", backup_dir=tmp_path / "b")
    assert result["written"]
    assert result["backup"].read_text(encoding="utf-8") == "old"
    assert target.read_text(encoding="utf-8") == "new"


def test_backups_never_clobber_each_other(tmp_path):
    target = tmp_path / "s.txt"
    target.write_text("v0", encoding="utf-8")
    first = safe_io.backup_file(target, tmp_path / "b")
    second = safe_io.backup_file(target, tmp_path / "b")
    assert first != second and first.exists() and second.exists()


def test_json_keeps_telugu_readable(tmp_path):
    target = tmp_path / "s.json"
    safe_io.write_json_safely(target, {"title": "చెప్పలేని ప్రేమ"}, backup_dir=tmp_path / "b")
    assert "చెప్పలేని ప్రేమ" in target.read_text(encoding="utf-8")


def test_no_temp_files_left_behind(tmp_path):
    safe_io.write_text_safely(tmp_path / "a.txt", "x", backup_dir=tmp_path / "b")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
