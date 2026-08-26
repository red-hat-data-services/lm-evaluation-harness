#!/usr/bin/env python3
"""Remove pure-Python packages from secondary requirements to avoid hermeto hash conflicts.

Hermeto (Konflux prefetch) puts all pip entries into a single flat output directory.
When two pip entries (e.g. CUDA and CPU indexes) both include a pure-Python package,
the wheel filename is identical (py3-none-any.whl) but the hashes differ because
AIPCC rebuilds everything per index. Hermeto deletes the file on hash mismatch.

Fix: strip pure-Python packages from the secondary (CPU) requirements. The primary
(CUDA) entry downloads them, and they're available to all arches from the shared
output directory. Binary packages have arch-specific filenames and don't conflict.

Detection heuristic: packages with exactly 1 hash in both files are pure-Python
(single wheel). Packages with >1 hash in either file have multiple arch-specific
wheels (binary).

Usage:
    python scripts/dedup_prefetch_requirements.py \\
        --primary requirements/cuda \\
        --secondary requirements/cpu \\
        [--requirements-files requirements.txt requirements-build.txt]
"""
import argparse
import re
import sys
from pathlib import Path


def parse_requirement_blocks(text: str) -> dict[str, dict]:
    """Parse a requirements file into {package_name: {block, hash_count}}."""
    packages = {}
    header_lines = []
    current_block = []
    current_name = None

    for line in text.splitlines(keepends=True):
        if re.match(r"^[a-zA-Z0-9_.-]+==[^\s]+", line):
            if current_name:
                packages[current_name] = {
                    "block": "".join(current_block),
                    "hash_count": len(re.findall(r"--hash=", "".join(current_block))),
                }
            current_name = re.match(r"^([a-zA-Z0-9_.-]+)==", line).group(1).lower()
            current_block = [line]
        elif current_name:
            current_block.append(line)
        else:
            header_lines.append(line)

    if current_name:
        packages[current_name] = {
            "block": "".join(current_block),
            "hash_count": len(re.findall(r"--hash=", "".join(current_block))),
        }

    return {"header": "".join(header_lines), "packages": packages}


def dedup_file(primary_path: Path, secondary_path: Path) -> tuple[int, int]:
    """Remove pure-Python packages from secondary that also exist in primary.

    Returns (kept, removed) counts.
    """
    primary = parse_requirement_blocks(primary_path.read_text())
    secondary = parse_requirement_blocks(secondary_path.read_text())

    kept_blocks = []
    removed = 0

    for name in sorted(secondary["packages"]):
        pkg = secondary["packages"][name]
        primary_pkg = primary["packages"].get(name)

        is_pure_python = (
            primary_pkg is not None
            and pkg["hash_count"] == 1
            and primary_pkg["hash_count"] == 1
        )

        if is_pure_python:
            removed += 1
        else:
            kept_blocks.append(pkg["block"])

    output = secondary["header"] + "".join(kept_blocks)
    secondary_path.write_text(output)
    return len(kept_blocks), removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--primary", required=True, type=Path, help="Directory with primary (CUDA) requirements")
    parser.add_argument("--secondary", required=True, type=Path, help="Directory with secondary (CPU) requirements")
    parser.add_argument(
        "--requirements-files",
        nargs="+",
        default=["requirements.txt", "requirements-build.txt"],
        help="Requirements filenames to process (default: requirements.txt requirements-build.txt)",
    )
    args = parser.parse_args()

    total_kept = 0
    total_removed = 0

    for filename in args.requirements_files:
        primary_path = args.primary / filename
        secondary_path = args.secondary / filename

        if not primary_path.exists():
            print(f"SKIP {filename}: primary file not found at {primary_path}", file=sys.stderr)
            continue
        if not secondary_path.exists():
            print(f"SKIP {filename}: secondary file not found at {secondary_path}", file=sys.stderr)
            continue

        kept, removed = dedup_file(primary_path, secondary_path)
        total_kept += kept
        total_removed += removed
        print(f"{filename}: kept {kept} binary packages, removed {removed} pure-Python packages")

    print(f"\nTotal: kept {total_kept}, removed {total_removed}")

    if total_removed == 0:
        print("Nothing to deduplicate — files may already be deduped or have no pure-Python overlap.")


if __name__ == "__main__":
    main()
