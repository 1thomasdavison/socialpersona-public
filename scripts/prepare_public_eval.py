#!/usr/bin/env python3
"""Prepare strictly public inputs for the maintained package CLIs.

This does not reconstruct private text, images, timestamps, or paper results.
User-level caption summaries stay separate: there is no reliable image-to-post
join in the source export, so this helper never invents one.
"""
import argparse
from pathlib import Path
import shutil

from user_profile_pipeline.release_privacy import verify_data


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=Path("results/public_inputs"))
    args = parser.parse_args()
    issues = verify_data(args.release_dir)
    if issues:
        raise SystemExit("Invalid public release: " + "; ".join(issues[:5]))
    output = args.output_dir.resolve()
    if output == args.release_dir.resolve() or (args.release_dir.resolve() / "data") in output.parents:
        raise SystemExit("Generated evaluation inputs must not overwrite released data")
    gold = output / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    for folder in sorted((args.release_dir / "data/users").iterdir()):
        shutil.copyfile(folder / "gold_profile.json", gold / (folder.name + ".json"))
    print("Prepared 100 generalized profiles for the maintained evaluation CLI.")


if __name__ == "__main__":
    main()
