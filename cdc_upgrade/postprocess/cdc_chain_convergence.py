#!/usr/bin/env python3
"""Combine multiple CDC chain roots and report convergence diagnostics."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def discover_text_chain(path_root: Path) -> Path:
    direct = path_root.with_suffix(".1.txt")
    if direct.exists():
        return direct
    candidate = path_root.parent / f"{path_root.name}.1.txt"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find first chain text file for root '{path_root}'.")


def load_chain_lengths(paths: list[Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        out[str(path)] = max(len(lines) - 1, 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CDC chain roots and report Gelman-Rubin convergence.")
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--combined-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--ignore-rows", type=float, default=0.0)
    args = parser.parse_args()

    text_files = [discover_text_chain(root) for root in args.roots]
    args.combined_root.parent.mkdir(parents=True, exist_ok=True)

    for index, src in enumerate(text_files, start=1):
        dst = args.combined_root.parent / f"{args.combined_root.name}.{index}.txt"
        shutil.copyfile(src, dst)
    first_root = args.roots[0]
    for suffix in (".updated.yaml", ".input.yaml"):
        meta = first_root.parent / f"{first_root.name}{suffix}"
        if meta.exists():
            shutil.copyfile(meta, args.combined_root.parent / f"{args.combined_root.name}{suffix}")

    try:
        from getdist import loadMCSamples
    except ImportError as exc:
        raise RuntimeError("getdist is required for convergence reporting.") from exc

    settings = {"ignore_rows": float(args.ignore_rows)} if args.ignore_rows else None
    samples = loadMCSamples(str(args.combined_root), no_cache=True, settings=settings)
    rminus1 = float(samples.getGelmanRubin())
    eigenvalues = [float(x) for x in samples.getGelmanRubinEigenvalues()]
    chain_lengths = load_chain_lengths(text_files)
    summary = {
        "combined_root": str(args.combined_root),
        "source_roots": [str(root) for root in args.roots],
        "source_chain_lengths": chain_lengths,
        "nchains": len(text_files),
        "total_rows": int(sum(chain_lengths.values())),
        "ignore_rows": float(args.ignore_rows),
        "rminus1": rminus1,
        "gelman_rubin_eigenvalues": eigenvalues,
    }

    target = args.output_json or (args.combined_root.parent / f"{args.combined_root.name}_convergence.json")
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Combined root : {args.combined_root}")
    print(f"Chains        : {len(text_files)}")
    print(f"R-1           : {rminus1:.6f}")
    print(f"Summary JSON  : {target}")


if __name__ == "__main__":
    main()
