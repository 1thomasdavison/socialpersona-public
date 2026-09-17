# Public data privacy treatment

Version: `targeted-deidentified-v2`.

## Content-preserving treatment

This version starts from the earlier entity-masked, record-level export. It
preserves individual posts, image descriptions and specific interest labels.
It replaces remaining identifying spans and uses reviewed local rewrites.
Public artists, athletes, fictional characters, works, games, brands and teams
can remain as interest objects. Existing masked names are not guessed or restored.

The previous `privacy-derived-v1` package reduced each participant to four
timeline summaries and six visual topic summaries. Version 2 restores the
record-level representation from the earlier local export.

| Field | Treatment |
| --- | --- |
| Participant IDs | Keep the existing randomly assigned public IDs. Identity mappings remain private. |
| Post IDs | Use participant-local sequential IDs; discard source IDs and deterministic hashes. |
| Text and image descriptions | Keep wording and detail around identifying spans. Normalize Unicode and remove account/contact patterns. |
| Private names and identifiers | Mask residual ordinary names, personal account names, identifying OCR, contacts and confirmed identifying spans. |
| Public interest objects | Preserve named works, performers, games, brands, teams and public destinations when they describe interests. |
| Personal affiliations and precise locations | Replace identifying school/workplace links, home/routine locations and identifying local labels. Broad travel destinations can remain. |
| Sensitive personal disclosures | Apply contextual edits to explicit private disclosures and clinical results. Generic activities, feelings and interests are retained. |
| Interest annotations | Apply the same privacy rules to labels. Keep domain status, stable/recent/negative membership and one entry per source annotation. No fixed vocabulary, frequency threshold or bucket-level merging is used. |
| Evidence | Resolve source post IDs and rebuild indices into the released rows. Reject unresolved references. |
| Time | Replace structured dates with days since the participant's first observed post. Preserve intervals and row order. Mask identified calendar dates in prose. |
| Post type and language | Retain the source's non-identifying type and language code. |
| Images and captions | Exclude original images, URLs, media hashes and file paths. Keep individual descriptions from six caption models. |
| Caption-to-post association | The earlier export did not retain a reliable association. This version marks it unavailable; no association is invented. |
| Other metadata and generated outputs | Exclude collection metadata, private mappings, raw model responses, review logs and evaluation caches. |

## Review and validation

Local pattern checks run before semantic review. With the dataset owner's
authorization, previously scrubbed prose was reviewed through ChatAnywhere.
Models proposed exact-span edits; a second review screened for excessive
redaction, followed by editorial checks of accepted changes and ambiguous names.
Nonliteral edits and edits to existing placeholders were quarantined. Requests,
responses, source fingerprints, edit decisions and mappings remain private.

The builder is `user_profile_pipeline.targeted_privacy`. It takes explicit
private rule and mapping files and refuses to overwrite an existing destination.
The optional review scripts require `tiktoken`, private working directories and
a `CHATANYWHERE_API_KEY`. Ordinary dataset loading and offline evaluation do not
call them. Cost caps and request hashes bound calls and prevent silently reusing
unrelated reviews.

`scripts/verify_release.py` checks versioned schemas, record counts, participant
IDs, evidence references, direct identifier patterns and data checksums. It also
scans the package and available Git history for credentials, private paths and
excluded artifacts. Offline tests cover preserving public interests and negation,
keyword replacement, unchanged source files, Unicode JSONL, evidence repair,
relative-time handling and protection of already-masked text.

## Interpretation

De-identification reduces direct identity disclosure; it does not guarantee
anonymity. Retained wording, interests, record counts and relative timing may
still support linkage. This version preserves more research detail than version
1 while providing a different privacy–utility tradeoff.

The earlier entity masking removed some useful names and occasionally
misclassified ordinary words. Those losses are not repaired by guessing. Changed
labels have not been re-annotated as a new human-validated gold standard, and no
claim is made that this version reproduces the paper's numerical results.

Report concerns or removal requests to **qkzhang@ir.hit.edu.cn**, using a public
participant ID and file path. Do not publish inferred identities or source text.
