# SocialPersona

Public code and a **privacy-derived research subset of 100 users** for studying
interest profiling and personalized dialogue from multimodal social-media context.

**This is an intentionally lossy derivative, not the inputs used to obtain the
paper's scores.** Verbatim posts, original image captions, original media,
precise timestamps, source identifiers, and per-user source counts are absent.
See [privacy treatment](PRIVACY.md), [data documentation](data/README.md), and the
machine-readable [audit](data/privacy_audit.json).

## Public data

Each randomly reassigned participant has four coarse timeline topic summaries,
a generalized interest profile across seven domains, and six user-level visual
topic summaries. Topic descriptions come exclusively from a fixed vocabulary;
no source free text is copied. Empty topics are explicitly marked as withheld.

```text
data/
  users/participant_001/
    posts.jsonl          # Four chronological-rank phase summaries, not posts
    gold_profile.json    # Generalized source annotations, not revalidated gold
    image_captions.json  # Six user-level topic summaries, not image captions
  vocabulary.json
  privacy_audit.json
  manifest.json          # SHA-256 of public files only
src/user_profile_pipeline/
scripts/
tests/
configs/*.example
```

The fixed four-row representation avoids publishing each person's original post
count. Phase numbers preserve coarse order only; they are neither dates nor
equal-duration intervals. User-level visual summaries have no post-level join.
The file names retain the dataset interface but their semantics have changed.

## Install and validate

Use Python 3.10 or newer. Run commands from this repository's root.

```bash
python -m pip install -e .
python -B scripts/verify_release.py --require-git
python -B -m unittest discover -s tests -v
```

## Offline evaluation smoke test

```bash
python -B scripts/prepare_public_eval.py
python -B -m user_profile_pipeline.benchmark.profile_eval \
  --gold-export-dir results/public_inputs/gold \
  --data-test-root data/users \
  --output-dir results/public_smoke \
  --models mock_oracle \
  --anchor-match-model exact_match \
  --visual-mode wo_text_image \
  --profile-input-mode text_only \
  --profile-method direct
```

The oracle copies generalized reference labels to test loading and scoring
with offline normalized exact-label matching;
it is not a measured model baseline and requires no model API calls. Experiments
on this derivative must report their own scores and its `privacy-derived-v1`
version. Caption summaries are provided for separate research use; this smoke
test uses text summaries only.

The maintained CLIs are
`user_profile_pipeline.benchmark.profile_eval`,
`user_profile_pipeline.personalized_dialogue.runner`, and
`user_profile_pipeline.cli`. Use `--help` for their options. Registered model
providers and environment-variable names are in
`src/user_profile_pipeline/benchmark/profile_eval/specs.py`; some registrations
use third-party API providers. Configure the intended endpoint before running
paid calls. Configuration examples contain environment-variable names, never
credentials. Historical standalone runners are not included.

## Scope and access

Use this subset for interface development and exploratory aggregate research.
It is unsuitable for validating detailed interest recovery, temporal distances,
image understanding, or reproducing the paper's numerical results. The
generalization is deterministic lexical extraction, not human-validated
paraphrasing: negation, ambiguity, non-English content and omitted details can
reduce utility. The original human validation does not transfer to these altered
labels and inputs.

The paper describes the controlled benchmark separately. Qualified researchers
may request controlled access from **qkzhang@ir.hit.edu.cn**. Public source
availability is not evidence of participants' consent. We make no new consent,
ethics-board approval, or zero-risk anonymity claim for this release.

This repository begins with a clean initial history. It excludes prior snapshots,
private identity mappings, per-user model outputs and caches. Manuscript PDFs
with source-derived example images are distributed separately and are not part
of this data package.

## Responsible use and rights

The intended use is aggregate research evaluation. Do not attempt account
identification, cross-dataset linkage, surveillance, targeting, or consequential
decisions about individuals. Report privacy concerns and removal requests to
**qkzhang@ir.hit.edu.cn**, using only the public participant identifier and file
path when possible; do not post suspected identities in public issues.

Code: [MIT](LICENSE). Author-created dataset annotations and the generalized
derivative: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), consistent
with the project's existing release designation. This designation does not grant
rights to excluded third-party source posts or images. The responsible-use
statement describes intended use and does not modify the licenses.
