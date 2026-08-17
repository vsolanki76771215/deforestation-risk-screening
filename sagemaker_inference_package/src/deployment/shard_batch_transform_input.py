#!/usr/bin/env python3
"""Split a prepared JSON Lines file into independently schedulable S3 objects."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--records-per-shard", type=int, default=500)
    args = parser.parse_args()
    if args.records_per_shard < 1:
        raise ValueError("--records-per-shard must be at least 1")
    source, destination = Path(args.input_jsonl), Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    total = shard_count = 0
    handle = None
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if total % args.records_per_shard == 0:
                    if handle:
                        handle.close()
                    shard_count += 1
                    handle = (destination / f"part-{shard_count:05d}.jsonl").open("w", encoding="utf-8", newline="\n")
                handle.write(line.rstrip("\r\n") + "\n")
                total += 1
    finally:
        if handle:
            handle.close()
    if total == 0:
        raise ValueError("Input file contains no JSON Lines records")
    print(f"Sharded {total} records into {shard_count} files: {destination}")


if __name__ == "__main__":
    main()
