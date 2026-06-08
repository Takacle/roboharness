"""Register a new robot in GMR's params.py and script argument choices.

Completely independent from skeleton_maps / body_matcher / config_gen.
Accepts only plain strings and paths — the caller assembles these from
higher-level modules.

Safety: writes are preceded by a backup, a syntax check via ``compile()``,
and an optional ``dry_run`` mode that only prints diffs.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def _find_dict_closing_line(lines: list[str], dict_var_name: str) -> int | None:
    """Return the line index of the ``}`` that closes *dict_var_name*.

    Scans forward from the line that starts ``<dict_var_name> = {``
    counting brace depth.
    """
    start_pat = re.compile(rf"^{re.escape(dict_var_name)}\s*=\s*\{{")
    start_idx: int | None = None
    for i, line in enumerate(lines):
        if start_pat.match(line.strip()):
            start_idx = i
            break
    if start_idx is None:
        return None

    depth = 0
    for i in range(start_idx, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            return i
    return None


def _find_inner_dict_bounds(
    lines: list[str], outer_name: str, sub_key: str
) -> tuple[int, int] | None:
    """Find ``(start, end)`` line indices of ``"sub_key": { ... }`` within *outer_name*.

    Returns line index of the opening ``"sub_key": {`` and the closing ``}``.
    """
    outer_close = _find_dict_closing_line(lines, outer_name)
    if outer_close is None:
        return None

    start_pat = re.compile(rf"^\s*\"{re.escape(sub_key)}\"\s*:\s*\{{")
    inner_start: int | None = None
    for i in range(len(lines)):
        if i > outer_close:
            break
        if start_pat.match(lines[i]):
            inner_start = i
            break
    if inner_start is None:
        return None

    depth = 0
    for i in range(inner_start, outer_close + 1):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            return (inner_start, i)

    return None


def _insert_entry(
    lines: list[str],
    dict_var_name: str,
    entry_line: str,
    sub_key: str | None = None,
) -> list[str] | None:
    """Insert *entry_line* into the dict, alphabetically before ``}``.

    When *sub_key* is provided, inserts into the inner dict
    ``"sub_key": { ... }`` nested within *dict_var_name*.
    """
    if sub_key is not None:
        bounds = _find_inner_dict_bounds(lines, dict_var_name, sub_key)
        if bounds is None:
            return None
        inner_start, close_idx = bounds
        start_idx: int = inner_start
    else:
        raw_close = _find_dict_closing_line(lines, dict_var_name)
        if raw_close is None:
            return None
        close_idx = raw_close
        start_pat = re.compile(rf"^{re.escape(dict_var_name)}\s*=\s*\{{")
        start_idx = -1
        for i, line in enumerate(lines):
            if start_pat.match(line.strip()):
                start_idx = i
                break
        if start_idx < 0:
            return None

    key_match = re.search(r'"([^"]+)"', entry_line)
    if key_match is None:
        return None
    new_key = key_match.group(1)

    insert_at = close_idx
    for i in range(start_idx + 1, close_idx):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        km = re.search(r'"([^"]+)"', stripped)
        if km and km.group(1) > new_key:
            insert_at = i
            break

    result = [*lines[:insert_at], entry_line + "\n", *lines[insert_at:]]
    return result


def register_in_params(
    gmr_root: Path,
    robot_name: str,
    xml_filename: str,
    base_body: str,
    cam_distance: float,
    src_formats: list[str],
    asset_subdir: str | None = None,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Add entries to GMR's ``params.py``.

    Modifies four dictionaries: ``ROBOT_XML_DICT``, ``ROBOT_BASE_DICT``,
    ``VIEWER_CAM_DISTANCE_DICT``, ``IK_CONFIG_DICT``.

    Returns a list of diff-like lines showing what would change.
    """
    params_path = gmr_root / "general_motion_retargeting" / "params.py"
    if not params_path.exists():
        raise FileNotFoundError(f"params.py not found at {params_path}")

    original = params_path.read_text()
    lines = original.splitlines(True)
    diff_lines: list[str] = []
    subdir = asset_subdir or robot_name

    entries: list[tuple[str, str, str | None]] = [
        (
            "ROBOT_XML_DICT",
            f'    "{robot_name}": ASSET_ROOT / "{subdir}" / "{xml_filename}",',
            None,
        ),
        ("ROBOT_BASE_DICT", f'    "{robot_name}": "{base_body}",', None),
        (
            "VIEWER_CAM_DISTANCE_DICT",
            f'    "{robot_name}": {cam_distance},',
            None,
        ),
    ]

    for fmt in src_formats:
        entries.append(
            (
                "IK_CONFIG_DICT",
                f'        "{robot_name}": IK_CONFIG_ROOT / "{fmt}_to_{robot_name}.json",',
                fmt,
            )
        )

    modified = lines[:]
    for dict_name, entry_line, sub_key in entries:

        def _has_active_entry(text_lines: list[str], key: str) -> bool:
            return any(f'"{key}"' in ln and not ln.lstrip().startswith("#") for ln in text_lines)

        if sub_key is not None:
            bounds = _find_inner_dict_bounds(original.splitlines(True), dict_name, sub_key)
            if bounds is not None:
                inner_start, inner_end = bounds
                inner_lines = original.splitlines(True)[inner_start : inner_end + 1]
                if _has_active_entry(inner_lines, robot_name):
                    diff_lines.append(
                        f"  SKIP {dict_name}[{sub_key}]: {robot_name} already present"
                    )
                    continue
        elif f'"{robot_name}"' in original:
            close_idx = _find_dict_closing_line(original.splitlines(True), dict_name)
            if close_idx is not None:
                start_pat = re.compile(rf"^{re.escape(dict_name)}\s*=\s*\{{")
                lines_for_check = original.splitlines(True)
                start_idx = None
                for i, line in enumerate(lines_for_check):
                    if start_pat.match(line.strip()):
                        start_idx = i
                        break
                if start_idx is not None:
                    dict_lines = lines_for_check[start_idx : close_idx + 1]
                    if _has_active_entry(dict_lines, robot_name):
                        diff_lines.append(f"  SKIP {dict_name}: {robot_name} already present")
                        continue

        new_lines = _insert_entry(modified, dict_name, entry_line, sub_key)
        if new_lines is None:
            diff_lines.append(f"  WARN: could not find {dict_name} in params.py")
            continue
        modified = new_lines
        diff_lines.append(f"  + {entry_line.strip()}")

    new_text = "".join(modified)
    try:
        compile(new_text, str(params_path), "exec")
    except SyntaxError as e:
        diff_lines.append(f"  ERROR: generated code has syntax error: {e}")
        return diff_lines

    if not dry_run and new_text != original:
        backup = params_path.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(params_path, backup)
            diff_lines.append(f"  backup: {backup}")
        params_path.write_text(new_text)
        diff_lines.append(f"  wrote: {params_path}")
    elif dry_run:
        diff_lines.append("  (dry_run — no changes written)")

    return diff_lines


def update_script_choices(
    gmr_root: Path,
    robot_name: str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Append *robot_name* to ``--robot choices=[...]`` in GMR scripts.

    Scans ``scripts/*.py`` for argparse choices containing known robot names
    and appends the new name. Returns list of modified file paths.
    """
    scripts_dir = gmr_root / "scripts"
    if not scripts_dir.is_dir():
        return [f"scripts dir not found: {scripts_dir}"]

    _CHOICES_RE = re.compile(
        r'(choices\s*=\s*\[(?:[^\]]*?)")\s*\]',
    )
    _KNOWN_ROBOT_RE = re.compile(r"unitree_g1|unitree_h1|booster_t1")
    modified: list[str] = []

    for script_path in sorted(scripts_dir.glob("*.py")):
        text = script_path.read_text()
        if "--robot" not in text:
            continue
        if not _KNOWN_ROBOT_RE.search(text):
            continue
        if f'"{robot_name}"' in text:
            modified.append(f"  SKIP {script_path.name}: already present")
            continue

        def _append_choice(m: re.Match) -> str:
            return f'{m.group(1)},\n                 "{robot_name}"]'

        new_text, count = _CHOICES_RE.subn(_append_choice, text)
        if count == 0:
            modified.append(f"  SKIP {script_path.name}: no choices pattern matched")
            continue

        try:
            compile(new_text, str(script_path), "exec")
        except SyntaxError:
            modified.append(f"  SKIP {script_path.name}: syntax check failed")
            continue

        if not dry_run:
            script_path.write_text(new_text)
        modified.append(f"  {'would update' if dry_run else 'updated'}: {script_path.name}")

    return modified
