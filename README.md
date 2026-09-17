# SocialPersona

Code and public data for **SocialPersona: Benchmarking Personalized Profiling and
Response with Multimodal Social-Media Context**.

[arXiv:2606.26654](https://arxiv.org/abs/2606.26654) ·
[Paper PDF](https://arxiv.org/pdf/2606.26654)

Qinkai Zhang, Yanyan Zhao, Xin Lu, Yulin Hu, Pengtao Han, and Bing Qin

Harbin Institute of Technology

## About the paper

People reveal their interests through what they post, photograph, and return to
over time. SocialPersona studies whether models can infer these interests from
social-media timelines and use them to give useful personalized recommendations.
It distinguishes stable interests from recent ones and connects two tasks:

| Task | What it evaluates |
| --- | --- |
| Profile Inference | Recover active domains and specific interests, with supporting evidence and stable/recent categories. |
| Personalized Dialogue | Use the user's interests in recommendations, evaluated for interest coverage, concreteness, and fluency. |

The full benchmark contains **171 users and 2,597 human-verified interest tags**
across seven domains: sports and outdoor activities, entertainment, gaming, food
and drink, travel and city exploration, photography and creation, and pets.
Timelines cover up to two years and 200 posts per user.

Experiments with six models show that broad interest domains are easier to
recover than specific and recent interests. Text and images provide complementary
signals, while using inferred profiles effectively in personalized dialogue
remains a challenge. The [paper](https://arxiv.org/abs/2606.26654) describes the
evaluation setup and results.

## Public dataset

The `targeted-deidentified-v2` release contains **100 users**,
**17,511 posts**, **85,488 image descriptions** for 14,248 image entries, and
**2,836 interest annotations** across seven domains.

De-identification replaces identifying words and selected private details while
preserving individual records and specific interests. Games, works, public
artists, brands and teams remain where they describe interests. The same rules
apply to post text, image descriptions and interest labels. Existing masked
names are not guessed or restored.

```text
data/
  users/participant_001/
    posts.jsonl          # Individual posts with relative days
    gold_profile.json    # Specific interest labels and local evidence links
    image_captions.json  # Individual descriptions from six caption models
  privacy_audit.json     # Aggregate transformation statistics
  manifest.json          # Data checksums
src/user_profile_pipeline/
scripts/
tests/
configs/*.example
```

Dates become relative days; source IDs, original images, contact details and
media fingerprints are excluded. Evidence IDs and row indices are rebuilt
consistently. Captions retain their image-level grouping; the earlier export did
not preserve a reliable caption-to-post association, so they remain separate
from the timeline input in the offline example.

This version retains substantially more detail than the four-summary
`privacy-derived-v1` release. Processing still changes the inputs, so new
experiments should report their own results and the data version. The paper's
reported scores come from the controlled benchmark.

See the [data guide](data/README.md) and [privacy treatment](PRIVACY.md).
For controlled research access, contact **qkzhang@ir.hit.edu.cn**.

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
  --max-posts 0 \
  --anchor-match-model exact_match \
  --visual-mode wo_text_image \
  --profile-input-mode text_only \
  --profile-method direct
```

In PowerShell, put the evaluation command on one line or use backticks for line
continuation. This check runs locally without API calls. `mock_oracle` copies
reference evidence from the full timeline; `--max-posts 0` keeps all posts so
those evidence indices remain valid. It copies
reference labels into predictions, and `exact_match` scores normalized label
matches. The resulting scores check the pipeline; model comparisons require
predictions from the models being evaluated. Outputs are saved under
`results/public_smoke/`.

## Evaluation code

Current profile extraction and scoring use `uncapped_interest_tags_v1`: supported
interest tags have no count limit in Direct, Hierarchical, or Extractive. Direct
uses one shared prompt; the other methods retain their own inputs and schemas.
Predictions and metrics record `profile_eval_protocol_version`, and resuming a run
requires the current version. The paper's existing scores predate this code revision.

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

## Citation

```bibtex
@misc{zhang2026socialpersona,
  title = {SocialPersona: Benchmarking Personalized Profiling and Response with Multimodal Social-Media Context},
  author = {Qinkai Zhang and Yanyan Zhao and Xin Lu and Yulin Hu and Pengtao Han and Bing Qin},
  year = {2026},
  eprint = {2606.26654},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2606.26654}
}
```

## License and contact

Code is released under the [MIT License](LICENSE). Author-created annotations
and the de-identified public data are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The dataset is intended for aggregate research. Please respect participants'
privacy and avoid attempts to identify or target individuals. Send research
questions, privacy concerns, or removal requests to **qkzhang@ir.hit.edu.cn**;
for data-specific requests, include the public participant ID and file path.
