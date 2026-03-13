#!/usr/bin/env python3
"""
Script to check if two files contain the same data in different orders.
Compares files line by line, ignoring the order of lines.
"""

import argparse
from pathlib import Path
from typing import Set, Tuple


def read_file_lines(file_path: Path) -> Tuple[Set[str], int]:
    """
    Read all lines from a file and return them as a set along with the total count.

    Args:
        file_path: Path to the file to read

    Returns:
        Tuple of (set of lines, total line count)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n\r') for line in f]

    return set(lines), len(lines)


def compare_files(file1_path: Path, file2_path: Path, verbose: bool = True) -> bool:
    """
    Compare two files to check if they contain the same data in different orders.

    Args:
        file1_path: Path to the first file
        file2_path: Path to the second file
        verbose: If True, print detailed comparison information

    Returns:
        True if files contain the same data, False otherwise
    """
    # Read both files
    print(f"Reading {file1_path}...")
    lines1, count1 = read_file_lines(file1_path)

    print(f"Reading {file2_path}...")
    lines2, count2 = read_file_lines(file2_path)

    # Print basic stats
    print(f"\n{file1_path.name}: {count1} total lines, {len(lines1)} unique lines")
    print(f"{file2_path.name}: {count2} total lines, {len(lines2)} unique lines")

    # Check if sets are equal
    are_same = lines1 == lines2

    if are_same:
        print(f"\n✓ Files contain the SAME data (possibly in different order)")
        if count1 != len(lines1):
            print(f"  Note: File 1 has {count1 - len(lines1)} duplicate lines")
        if count2 != len(lines2):
            print(f"  Note: File 2 has {count2 - len(lines2)} duplicate lines")
    else:
        print(f"\n✗ Files contain DIFFERENT data")

        if verbose:
            # Find differences
            only_in_file1 = lines1 - lines2
            only_in_file2 = lines2 - lines1

            if only_in_file1:
                print(f"\nLines only in {file1_path.name} ({len(only_in_file1)} lines):")
                for line in sorted(only_in_file1)[:10]:  # Show first 10
                    print(f"  {line[:100]}...")  # Truncate long lines
                if len(only_in_file1) > 10:
                    print(f"  ... and {len(only_in_file1) - 10} more")

            if only_in_file2:
                print(f"\nLines only in {file2_path.name} ({len(only_in_file2)} lines):")
                for line in sorted(only_in_file2)[:10]:  # Show first 10
                    print(f"  {line[:100]}...")  # Truncate long lines
                if len(only_in_file2) > 10:
                    print(f"  ... and {len(only_in_file2) - 10} more")

    return are_same


def main():
    parser = argparse.ArgumentParser(
        description="Check if two files contain the same data in different orders"
    )
    parser.add_argument(
        "file1",
        type=str,
        help="Path to the first file (relative to data/datasets/)"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Path to the second file (relative to data/datasets/)"
    )

    args = parser.parse_args()

    # Prepend data/datasets to file paths
    base_path = Path("data/datasets")
    file1_path = base_path / args.file1
    file2_path = base_path / args.file2

    # Check if files exist
    if not file1_path.exists():
        print(f"Error: File not found: {file1_path}")
        return 1

    if not file2_path.exists():
        print(f"Error: File not found: {file2_path}")
        return 1

    # Compare files (verbose is always True by default)
    try:
        are_same = compare_files(file1_path, file2_path)
        return 0 if are_same else 1
    except Exception as e:
        print(f"Error comparing files: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
