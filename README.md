# SocialPersona

Code and public data for **SocialPersona: From Social-Media Evidence to
Personalized Recommendations**.

Qinkai Zhang, Yanyan Zhao, Xin Lu, Yulin Hu, Pengtao Han, and Bing Qin

Harbin Institute of Technology

## About the paper

People reveal their interests through what they post, photograph, and return to
over time. SocialPersona studies whether models can infer these interests from
social-media timelines and use them to give useful personalized recommendations.
It distinguishes stable interests from recent ones and connects two tasks:

| Task | What it evaluates |
| --- | --- |
| Profile construction | Recover active domains and specific interests, with supporting evidence and stable/recent categories. |
| Personalized response generation | Use the user's interests in recommendations, evaluated for interest coverage, concreteness, and fluency. |

The full benchmark contains **171 users and 2,597 human-verified interest tags**
across seven domains: sports and outdoor activities, entertainment, gaming, food
and drink, travel and city exploration, photography and creation, and pets.
Timelines cover up to two years and 200 posts per user.

The paper evaluates six models on a fixed 100-user subset using text, image
captions, and timestamps. Models recover broad domains more reliably than
specific interests; the best Interest F1 is **0.414**. In controlled recommendation
experiments, human-annotated profiles improve interest coverage over predicted
profiles by **0.41–1.16 points** on a 0–5 scale. Predicted profiles have mixed
results compared with the original timelines, highlighting the value of retaining
specific interests and the details that support them.

## Public dataset

This repository includes a **100-user subset**. To protect participants' privacy,
the public data underwent strict de-identification: identifying details and
original media were removed, and posts, captions, and interest labels were
generalized. These changes reduce the available detail, so evaluation results
on the public version will differ from those reported in the paper.

Each participant has four ordered timeline summaries, an interest profile across
seven domains, and six user-level visual topic summaries. The release contains
400 timeline summaries, 600 visual summaries, and 1,841 generalized interest
entries. It supports exploring the data format and evaluation pipeline.

```text
data/
  users/participant_001/
    posts.jsonl          # Four timeline topic summaries
    gold_profile.json    # Generalized interest annotations and evidence links
    image_captions.json  # User-level visual topics from six caption models
  vocabulary.json       # Topic vocabulary
  privacy_audit.json    # Aggregate privacy-processing statistics
  manifest.json         # Public data checksums
src/user_profile_pipeline/
scripts/
tests/
configs/*.example
```

Timeline phases preserve coarse order; dates and durations have been removed.
Visual summaries aggregate topics at the user level. Generalized interest labels
retain their source categories and have not undergone a separate human review.
See the [data guide](data/README.md) for field definitions and
[privacy documentation](PRIVACY.md) for the processing procedure.

For controlled research access to the full benchmark, contact
**qkzhang@ir.hit.edu.cn**. When reporting experiments with the public subset,
identify the data version as `privacy-derived-v1`.

## Quick start

Use Python 3.10 or newer. Clone the repository and install it from the repository
root, preferably in a virtual environment:

```bash
git clone https://github.com/1thomasdavison/socialpersona-public.git
cd socialpersona-public
python -m pip install -e .
```

Prepare the public profiles and run an offline loading and scoring check:

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

In PowerShell, put the evaluation command on one line or use backticks for line
continuation. This check runs locally without API calls. `mock_oracle` copies
reference labels into predictions, and `exact_match` scores normalized label
matches. The resulting scores check the pipeline; model comparisons require
predictions from the models being evaluated. Outputs are saved under
`results/public_smoke/`.

## Evaluation code

The profiling code implements three methods: **direct** inference from a timeline,
**hierarchical** summarization followed by aggregation, and **extractive** evidence
selection followed by profile generation. The response evaluation code supports
generation and judging with different user-context settings.

| Entry point | Purpose |
| --- | --- |
| `python -m user_profile_pipeline.benchmark.profile_eval` | Profile inference and scoring |
| `python -m user_profile_pipeline.personalized_dialogue.runner` | Personalized response generation and evaluation |
| `python -m user_profile_pipeline.cli` | Annotation and profile construction pipeline |

Use `--help` with any entry point for its arguments. Model registrations,
endpoints, and API-key environment-variable names are listed in
[`specs.py`](src/user_profile_pipeline/benchmark/profile_eval/specs.py).
Some registrations use third-party providers; select the endpoint you intend
to use and supply credentials through environment variables. Configuration
examples are in [`configs/`](configs/).

Run the offline tests with:

```bash
python -B -m unittest discover -s tests -v
```

For release checks, run `python -B scripts/verify_release.py --require-git` in a
clean checkout. It checks data schemas, evidence links, checksums, package files,
and Git history. Local environments and generated results should be kept outside
that checkout during this check.

## License and contact

Code is released under the [MIT License](LICENSE). Author-created annotations
and the generalized public data are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The dataset is intended for aggregate research. Please respect participants'
privacy and avoid attempts to identify or target individuals. Send research
questions, privacy concerns, or removal requests to **qkzhang@ir.hit.edu.cn**;
for data-specific requests, include the public participant ID and file path.
