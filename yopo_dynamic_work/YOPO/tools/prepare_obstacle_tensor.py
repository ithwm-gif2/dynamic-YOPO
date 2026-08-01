#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML


def convert_episode(episode_path: Path, force: bool = False) -> None:
    yaml = YAML(typ="safe")
    with open(episode_path / "metadata.yaml", "r", encoding="utf-8") as file:
        metadata = yaml.load(file)

    frames = int(metadata["frames"])
    obstacle_count = int(metadata["obstacle_count"])
    expected_records = frames * obstacle_count
    output_path = episode_path / "obstacles.bin"
    expected_bytes = expected_records * 6 * np.dtype(np.float32).itemsize
    if output_path.exists() and output_path.stat().st_size == expected_bytes and not force:
        print(f"skip {episode_path.name}: obstacle tensor already valid")
        return

    temporary_path = episode_path / "obstacles.bin.tmp"
    input_path = episode_path / "obstacles.csv"
    record_count = 0
    buffer = []
    chunk_records = 100_000

    with open(input_path, "r", encoding="utf-8", newline="") as source, open(
        temporary_path, "wb"
    ) as target:
        reader = csv.DictReader(source)
        required = {"cx", "cy", "cz", "size_x", "size_y", "height"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unsupported obstacle CSV schema: {input_path}")

        for row in reader:
            buffer.append(
                (
                    float(row["cx"]),
                    float(row["cy"]),
                    float(row["cz"]),
                    0.5 * float(row["size_x"]),
                    0.5 * float(row["size_y"]),
                    0.5 * float(row["height"]),
                )
            )
            if len(buffer) >= chunk_records:
                np.asarray(buffer, dtype=np.float32).tofile(target)
                record_count += len(buffer)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=np.float32).tofile(target)
            record_count += len(buffer)

    if record_count != expected_records:
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"{episode_path}: {record_count} records, expected {expected_records}"
        )
    os.replace(temporary_path, output_path)
    print(
        f"converted {episode_path.name}: {record_count} records, "
        f"{output_path.stat().st_size / (1024 ** 2):.1f} MiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert obstacle CSV labels to a float32 memory-mapped tensor."
    )
    parser.add_argument("roots", nargs="+", help="Dataset root directories")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for root_text in args.roots:
        root = Path(root_text).expanduser().resolve()
        episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
        if not episodes:
            raise FileNotFoundError(f"No episode_* folders found in {root}")
        for episode in episodes:
            convert_episode(episode, force=args.force)


if __name__ == "__main__":
    main()
