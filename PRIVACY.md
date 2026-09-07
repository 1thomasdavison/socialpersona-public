# Public-release privacy treatment

Version: `privacy-derived-v1`. Prepared on 2026-09-07.

## Scope and threat model

The source export contains 100 users, 17,511 posts, 2,836 interest annotations,
14,248 image entries and 85,488 model captions. A surface replacement of account
IDs was insufficient: 821 posts retained mentions, 2,098 retained hashtags, all
17,511 retained dates, and 1,785 media entries retained nonempty content hashes.
Searchable wording, image descriptions, exact record counts and evidence graphs
also allow linkage even after named entities are masked.

This public derivative applies data minimization locally. Source text and
captions are never sent to an external language model for this transformation.
All exported prose is reconstructed from literal templates and a reviewed,
non-sensitive topic vocabulary. No unknown source phrase has a pass-through
fallback.

## Treatment by field

| Source information | Public treatment |
| --- | --- |
| Account names, handles and former anonymous IDs | Randomly reassign 100 participant IDs; retain the assignment privately outside this repository. |
| Original post IDs, reply/quote links and deterministic ID hashes | Remove; create local phase IDs with no source-derived hash. |
| Verbatim post bodies, spelling, quotes and unusual phrases | Extract only approved generic topics; emit fixed-template summaries. All remaining text is omitted. |
| Mentions, hashtags, URLs and email addresses | Remove before lexical extraction. No contact strings are exported from participant data. |
| Names, affiliations, named places, venues, titles, brands and OCR | Never copy any extracted source span. These are outside the output vocabulary. |
| Health, politics, religion, demographics and other sensitive attributes | No output categories for these attributes; no new sensitive inference. |
| Dates, time zones, observation spans and collection metadata | Omit. Sort locally and group by four chronological-rank phases without dates or durations. |
| Each user's original number of posts and image counts | Omit; emit exactly four text summaries and six model summaries per user. |
| Raw images, thumbnails, EXIF, visual fingerprints and media hashes | Omit all files and identifiers. |
| Image-level model captions | Aggregate each model's captions to a user-level generic topic summary; omit individual descriptions and image-level ordering. |
| Precise and rare interest labels | Map within the original domain to generic vocabulary, or its broad domain fallback. Retain a topic only if detected for at least five source users. |
| Long-term, short-term and negative buckets | Keep source bucket membership but generalize and merge labels within each bucket. Buckets are not recomputed from the public phases. |
| Evidence IDs and indices | Rebuild to public phase records; deduplicate links. Original IDs are authoritative if old indices disagree. Never invent a caption-to-post association. |
| Metadata, raw predictions, dialogue outputs and caches | Exclude entirely. |
| Git history and binary manuscript illustrations | Start with a clean history and exclude old snapshots and source-derived illustration files. |

Topical extraction selects up to five frequent eligible topics per summary and
prints them alphabetically, without their frequencies. It does not establish
that the user likes each topic: a mention, negation, or visual object can trigger
the same lexical feature. Names containing ordinary topic words can yield a
generic topic but cannot be copied into the released text.

The source had 729 interest entries whose ID-based and index-based evidence
pointed to different sets of phases. All referenced source IDs resolved. The
derivative uses the explicit IDs, rebuilds indices and checks their equality.
It does not change the controlled benchmark's original annotation files.

## Validation

`scripts/verify_release.py` validates all user files, strict field allowlists,
fixed templates, vocabulary membership, four-phase shape, six-model shape,
cross-file IDs, evidence references and public-file checksums. It also rejects
private mappings, generated artifacts, media files, credential patterns and
internal filesystem paths in the package and available Git history.

Regression tests inject names, addresses, links, hashes and arbitrary fields;
confirm suppression of a topic seen for fewer than five users; exercise
multilingual/no-match handling and Unicode JSONL; and check mismatched evidence.
The audit contains aggregate counts only, never source snippets or mappings.

## Privacy and scientific limitations

The five-user vocabulary threshold is **not** k-anonymity of participant records.
No differential-privacy, complete anonymity, or immunity to membership/linkage
attacks is claimed. Joint combinations of interests and retained coarse patterns
may still be distinctive, especially to someone who already has source data.
Earlier downloads or third-party copies cannot be recalled by this release.

The derivative contains 400 text-phase summaries, 600 caption-model summaries,
and 1,841 generalized interest entries. Original per-user domain statuses are
retained. Full text, visual detail, time gaps, fine labels and evidence frequency
are deliberately lost. Generalized labels are not newly validated gold, and
the paper's scores must not be attributed to this derivative. The existing
controlled inputs and reported results have not been changed by this release.

For a privacy concern or removal request, contact **qkzhang@ir.hit.edu.cn** with
the public participant ID. Do not publish inferred identities or raw content.
