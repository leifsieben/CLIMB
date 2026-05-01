#!/usr/bin/env python3
"""Expand an experiment spec into a resolved run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_manifest import generate_manifest, load_spec, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a resolved CLIMB experiment manifest")
    parser.add_argument("--spec", required=True, help="Path to experiment spec YAML")
    parser.add_argument("--output", required=True, help="Output manifest JSON")
    parser.add_argument("--dry_run", action="store_true", help="Print counts and do not write output")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    manifest = generate_manifest(spec, spec_path=args.spec)

    if args.dry_run:
        print(json.dumps(manifest["summary"], indent=2))
        return

    write_manifest(manifest, args.output)
    print(f"Wrote manifest to {args.output}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
