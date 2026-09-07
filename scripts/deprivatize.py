#!/usr/bin/env python3
"""Build a privacy-first public derivative with explicit source/output paths.

The former hash/NER export is not a publication boundary. See PRIVACY.md in the
public repository for the intentionally lossy fixed-vocabulary representation.
"""
from user_profile_pipeline.release_privacy import main

if __name__ == "__main__":
    main()
