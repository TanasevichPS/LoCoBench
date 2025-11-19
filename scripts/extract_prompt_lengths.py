#!/usr/bin/env python3
"""
Extract prompt/response lengths from the latest generation log.

By default the script scans
    /srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/clean/LoCoBench/logs
for files matching `generation_*.log`, picks the most recently created
file, and emits a JSONL summary containing the prompt/response lengths.

Usage:
    python scripts/extract_prompt_lengths.py \
        --log-dir /path/to/logs \
        --output logs/prompt_length_summary.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_LOG_DIR = Path("/srv/nfs/VESO/home/polina/trsh/mcp/LoCoBench/clean/LoCoBench/logs")
DEFAULT_OUTPUT = Path("logs/prompt_length_summary.jsonl")

PROMPT_RE = re.compile(r"Prompt length:\s*(\d+)\s*chars")
RESPONSE_RE = re.compile(r"OpenAI response length:\s*(\d+)\s*chars")
TIMESTAMP_RE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")


@dataclass
class LengthEntry:
    timestamp: str
    prompt_length: Optional[int]
    response_length: Optional[int]

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "prompt_length": self.prompt_length,
                "response_length": self.response_length,
                "prompt_length_display": (
                    f"Prompt length: {self.prompt_length} chars" if self.prompt_length is not None else None
                ),
                "response_length_display": (
                    f"OpenAI response length: {self.response_length} chars"
                    if self.response_length is not None
                    else None
                ),
            },
            ensure_ascii=False,
        )


def find_latest_log(log_dir: Path) -> Path:
    log_files = sorted(log_dir.glob("generation_*.log"))
    if not log_files:
        raise FileNotFoundError(f"No generation_*.log files found in {log_dir}")
    return max(log_files, key=lambda p: p.stat().st_mtime)


def parse_log(file_path: Path) -> List[LengthEntry]:
    entries: List[LengthEntry] = []
    pending_prompt: Optional[LengthEntry] = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip("\n")
            ts_match = TIMESTAMP_RE.match(line)
            timestamp = ts_match.group(1) if ts_match else ""

            prompt_match = PROMPT_RE.search(line)
            if prompt_match:
                pending_prompt = LengthEntry(timestamp=timestamp, prompt_length=int(prompt_match.group(1)), response_length=None)
                continue

            response_match = RESPONSE_RE.search(line)
            if response_match:
                response_length = int(response_match.group(1))
                if pending_prompt:
                    pending_prompt.response_length = response_length
                    entries.append(pending_prompt)
                    pending_prompt = None
                else:
                    entries.append(
                        LengthEntry(timestamp=timestamp, prompt_length=None, response_length=response_length)
                    )

    if pending_prompt:
        entries.append(pending_prompt)

    return entries


def write_output(entries: List[LengthEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        for entry in entries:
            out_file.write(entry.to_json() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract prompt/response lengths from generation logs.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory containing generation logs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL file for extracted lengths")
    args = parser.parse_args()

    latest_log = find_latest_log(args.log_dir)
    entries = parse_log(latest_log)
    write_output(entries, args.output)

    print(f"Parsed {len(entries)} prompt/response entries from {latest_log}")
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
