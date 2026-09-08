# Public dataset

Version: `privacy-derived-v1`.

The public subset contains 100 users, 400 timeline topic summaries, 600 user-level
visual topic summaries, seven profile domains per user, and 1,841 generalized
interest entries. Strict de-identification and content generalization protect
participants' privacy while retaining a common format for research. The reduced
detail means that evaluations on this version will differ from the paper's
original experiments.

## Files

| File | Contents |
| --- | --- |
| `users/participant_NNN/posts.jsonl` | Four timeline summaries, ordered by `relative_phase`. Each row contains a local `post_id`, fixed-vocabulary `topics`, template-generated `text`, and an empty `media` list. |
| `users/participant_NNN/gold_profile.json` | Generalized interest annotations, source domain statuses, and `long_term_interests`, `short_term_interests`, and `negative_interests` categories. Evidence links refer to the four timeline summaries. |
| `users/participant_NNN/image_captions.json` | `model_summaries` from six caption models, each aggregating visual topics across the user's timeline. |
| `vocabulary.json` | Permitted topic labels and broad domain labels used when more specific topics are withheld. |
| `privacy_audit.json` | Aggregate counts from privacy processing. |
| `manifest.json` | SHA-256 checksums of the public data files. |

## Reading the records

Each timeline phase groups a quarter of the user's posts in chronological order.
The public records preserve this coarse sequence. Dates, elapsed durations,
within-phase ordering, and original post counts have been removed. A `post_id`
therefore identifies a summary phase. An `evidence_post_indices` value is a
zero-based index into the four rows, and `evidence_post_ids` provides the matching
phase IDs.

Interest categories retain the source annotations: `long_term_interests`
corresponds to stable interests, and `short_term_interests` to recent interests.
Labels have been generalized and merged within each category. The categories
were assigned from the source timelines, and the transformed labels have not
undergone a separate human review. The `gold_profile.json` filename keeps the
existing evaluation interface.

Visual summaries are grouped by caption model at the user level. Individual
images and their captions have been removed, along with links to specific posts
or timeline phases. The README's offline example uses the timeline text summaries.

An empty topic list is displayed as `Generalized text topics: content withheld.`
or its visual equivalent. This means the processing retained no eligible topics
for that summary; the source may still contain activity.

See the repository [README](../README.md) for the paper overview and an offline
evaluation example, and [PRIVACY.md](../PRIVACY.md) for the full processing details
and privacy considerations.
