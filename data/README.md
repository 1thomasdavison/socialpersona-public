# Public dataset

Version: `targeted-deidentified-v2`.

The dataset contains 100 participants, 17,511 individual posts, 14,248 image
entries with 85,488 model descriptions, and 2,836 interest annotations. Targeted
replacements preserve specific interests and the surrounding text. Ten local
place labels were de-identified; all annotation entries and source buckets remain.

## Files and fields

| File | Contents |
| --- | --- |
| `users/participant_NNN/posts.jsonl` | Individual records with `user_id`, local `post_id`, `relative_day`, `post_type`, `language`, `text`, and an empty `media` list. |
| `users/participant_NNN/gold_profile.json` | Seven domains with source status and `long_term_interests`, `short_term_interests`, and `negative_interests`. Each entry has a label and matching evidence IDs/indices. |
| `users/participant_NNN/image_captions.json` | Individual image groups under `captions`, each with six `model_captions` summaries. `post_association` is explicitly `unavailable`. |
| `privacy_audit.json` | Aggregate processing counts. |
| `manifest.json` | SHA-256 checksums of data files and this guide. |

`relative_day` is the number of days since the participant's earliest source
post. Rows retain the source export's order, which may include a pinned post and
reverse chronology. Sort on `relative_day` for chronological views; keep the
original row indices when resolving evidence. Calendar dates and original IDs
are not used as public metadata.

`evidence_post_indices` contains zero-based positions in `posts.jsonl`.
`evidence_post_ids` names those same records. Where the earlier export's indices
disagreed with its IDs, the IDs determined the rebuilt references.

Long-term and short-term correspond to stable and recent source annotations.
Privacy processing does not reassess their membership. Masked standalone local
place labels become `exploring local area N`, with distinct source places kept
separate within a participant. Label changes have not been re-annotated as new
human-validated gold.

The image descriptions retain image-level grouping and the original model keys.
The earlier export did not preserve a reliable image-to-post association; no
join is reconstructed from sequence or image counts. Original media and media
fingerprints are excluded. The offline example in the main README uses posts.

This version restores detail from the earlier entity-masked local export. It
retains existing placeholders rather than guessing removed names. Public works,
games, brands, teams and performers remain when they describe interests. Results
obtained with these processed inputs should report this version and their own
scores. See [PRIVACY.md](../PRIVACY.md) for the treatment and remaining limits.
