#!/usr/bin/env python3
"""Prepare generalized public profiles for the evaluation CLI.

Visual topic summaries remain at the user level because the public data omits
links between images and individual posts.
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
    release_dir = args.release_dir.resolve()
    output = args.output_dir.resolve()
    data_dir = release_dir / "data"
    if output in (release_dir, data_dir) or data_dir in output.parents:
        raise SystemExit("Generated evaluation inputs must not overwrite released data")
    issues = verify_data(release_dir)
    if issues:
        raise SystemExit("Invalid public release: " + "; ".join(issues[:5]))
    gold = output / "gold"
    gold.mkdir(parents=True, exist_ok=True)
    for folder in sorted((data_dir / "users").iterdir()):
        shutil.copyfile(folder / "gold_profile.json", gold / (folder.name + ".json"))
    print("Prepared 100 generalized profiles for the maintained evaluation CLI.")


if __name__ == "__main__":
    main()
