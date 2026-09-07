# Privacy-derived subset

100 users, 400 aggregate text summaries, 600 user-level visual topic summaries,
seven profile domains per user, and 1,841 generalized interest entries.
This is a lossy derivative, not the original benchmark or a verbatim-text corpus.

## Files

- `users/participant_NNN/posts.jsonl`: four rows ordered by `relative_phase`.
  `post_id` identifies an aggregate phase, not a social-media post; `topics` is
  a fixed-vocabulary list and `text` is a deterministic template. `media` is empty.
- `users/participant_NNN/gold_profile.json`: generalized annotations with source
  active/inactive statuses and long/short/negative bucket membership. Evidence
  indices are zero-based references to the four summary rows; the corresponding
  IDs must match. The filename is retained for tooling and does not imply that
  the transformed annotations have been human revalidated.
- `users/participant_NNN/image_captions.json`: `model_summaries` for six named
  source caption models. These summarize topics across each user's captions;
  they are not descriptions of individual images and have no phase association.
- `vocabulary.json`: permitted generic labels, including broad domain fallbacks.
- `privacy_audit.json`: aggregate transformation counts and limitations.
- `manifest.json`: hashes of public data files, with no original media hashes.

Phases are quartiles of chronological post rank, computed privately, with no
absolute timestamps, elapsed durations, original ordering within a phase, or
per-user source counts. The long/short annotation distinction comes from the
source profiles, not these phases. An empty topic list is rendered as
`Generalized text topics: content withheld.` or its visual counterpart; it is
not evidence that the source contains no activity.

There is no model-API requirement to inspect this subset. See the repository
[README](../README.md) for an offline loading/scoring smoke test and
[PRIVACY.md](../PRIVACY.md) for the full treatment and residual risks.
