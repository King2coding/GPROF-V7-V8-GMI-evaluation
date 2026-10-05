#!/usr/bin/env python3
"""Safely inventory or delete final GMI matchup files outside 2021--2024."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import stat
import sys
from pathlib import Path


DEFAULT_MATCHUP_DIR = Path(
    "/scratch/kkumah/GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_matchups/matchups/GMI"
)
DEFAULT_REPORT_DIR = DEFAULT_MATCHUP_DIR.parents[1] / "logs"
START_DATE = dt.date(2021, 1, 1)
END_DATE = dt.date(2024, 12, 31)
FINAL_NAME = re.compile(
    r"GMI_GPROF_V7_V8_MRMS_RAQI_ERA5_MERRA2_AutoSnow_(\d{6})_(\d{8})\.nc$"
)


def parse_final_name(name: str) -> dt.date:
    match = FINAL_NAME.fullmatch(name)
    if not match:
        raise ValueError("filename does not match the authoritative final GMI matchup pattern")
    return dt.datetime.strptime(match.group(2), "%Y%m%d").date()


def inventory(directory: Path) -> dict:
    categories = {
        "inside_retained_period": [],
        "before_start_date": [],
        "after_end_date": [],
        "malformed_or_unparseable": [],
        "unsafe_symbolic_links": [],
        "stale_partial_files": [],
    }
    total_final = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            path = str(directory / entry.name)
            if entry.name.endswith(".partial"):
                info = entry.stat(follow_symlinks=False)
                categories["stale_partial_files"].append({"path": path, "bytes": info.st_size})
                continue
            if not entry.name.endswith(".nc"):
                continue
            total_final += 1
            if entry.is_symlink():
                categories["unsafe_symbolic_links"].append({"path": path, "reason": "symbolic link"})
                continue
            if not entry.is_file(follow_symlinks=False):
                categories["malformed_or_unparseable"].append(
                    {"path": path, "reason": "not a regular file"}
                )
                continue
            size = entry.stat(follow_symlinks=False).st_size
            try:
                acquisition_date = parse_final_name(entry.name)
            except ValueError as exc:
                categories["malformed_or_unparseable"].append(
                    {"path": path, "bytes": size, "reason": str(exc)}
                )
                continue
            item = {"path": path, "bytes": size, "acquisition_date": acquisition_date.isoformat()}
            if acquisition_date < START_DATE:
                categories["before_start_date"].append(item)
            elif acquisition_date > END_DATE:
                categories["after_end_date"].append(item)
            else:
                categories["inside_retained_period"].append(item)

    for values in categories.values():
        values.sort(key=lambda item: item["path"])
    proposed = categories["before_start_date"] + categories["after_end_date"]
    proposed.sort(key=lambda item: item["path"])
    return {
        "matchup_directory": str(directory),
        "retained_start_date_inclusive": START_DATE.isoformat(),
        "retained_end_date_inclusive": END_DATE.isoformat(),
        "total_final_matchup_files_inspected": total_final,
        "inside_retained_period_count": len(categories["inside_retained_period"]),
        "before_start_date_count": len(categories["before_start_date"]),
        "after_end_date_count": len(categories["after_end_date"]),
        "malformed_or_unparseable_count": len(categories["malformed_or_unparseable"]),
        "unsafe_symbolic_link_count": len(categories["unsafe_symbolic_links"]),
        "stale_partial_count": len(categories["stale_partial_files"]),
        "proposed_deletion_count": len(proposed),
        "proposed_bytes_released": sum(item["bytes"] for item in proposed),
        "proposed_deletion_paths": [item["path"] for item in proposed],
        **categories,
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-dir", type=Path, default=DEFAULT_MATCHUP_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--delete", action="store_true", help="delete confirmed out-of-period files")
    args = parser.parse_args()

    directory = args.matchup_dir
    if directory.is_symlink() or not directory.is_dir():
        parser.error(f"matchup directory is missing, not a directory, or is a symbolic link: {directory}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or DEFAULT_REPORT_DIR / f"gmi_study_period_cleanup_{stamp}.json"
    report = inventory(directory)
    unsafe = report["malformed_or_unparseable_count"] + report["unsafe_symbolic_link_count"]
    report.update({
        "mode": "delete" if args.delete else "dry-run",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "deleted_paths": [],
        "deletion_errors": [],
    })

    print(json.dumps({key: report[key] for key in (
        "mode", "total_final_matchup_files_inspected", "inside_retained_period_count",
        "before_start_date_count", "after_end_date_count", "malformed_or_unparseable_count",
        "unsafe_symbolic_link_count", "stale_partial_count", "proposed_deletion_count",
        "proposed_bytes_released",
    )}, indent=2))
    for path in report["proposed_deletion_paths"]:
        print(f"{'DELETE_CANDIDATE' if args.delete else 'WOULD_DELETE'} {path}")
    for item in report["stale_partial_files"]:
        print(f"PARTIAL_REPORTED_ONLY {item['path']}")

    if args.delete and unsafe:
        print("Refusing all deletion because unsafe or unclassifiable .nc entries exist.", file=sys.stderr)
    elif args.delete:
        for item in report["before_start_date"] + report["after_end_date"]:
            path = Path(item["path"])
            try:
                current = path.lstat()
                if not stat.S_ISREG(current.st_mode) or current.st_size != item["bytes"]:
                    raise RuntimeError("file type or size changed after inventory")
                if START_DATE <= parse_final_name(path.name) <= END_DATE:
                    raise RuntimeError("file is no longer classified outside the retained period")
                path.unlink()
                report["deleted_paths"].append(str(path))
                print(f"DELETED {path}")
            except Exception as exc:
                report["deletion_errors"].append({"path": str(path), "error": str(exc)})
                print(f"DELETE_FAILED {path}: {exc}", file=sys.stderr)

    write_report(report, report_path)
    print(f"REPORT {report_path}")
    if unsafe or report["deletion_errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
