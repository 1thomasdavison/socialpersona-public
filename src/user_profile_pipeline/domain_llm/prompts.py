from __future__ import annotations


DOMAIN_EVIDENCE_FIRST_SYSTEM_PROMPT = """You are performing evidence-first interest summarization for one domain.

Goal:
- Produce natural-language interest labels and short descriptions suitable for downstream LLM personalization benchmarking.
- Use canonical tags only as evidence anchors, not as the main surface form.

You will receive:
- domain name and definition
- profiling target / generation goal metadata
- observation-window metadata
- canonical tag clusters (including evidence examples)
- representative posts
- optional chunk summaries

Rules:
- Use only provided evidence.
- Output human-readable labels (2-6 words), not raw canonical ids.
- Labels should usually add user-facing detail beyond any single canonical tag. Synthesize repeated patterns from tag clusters, evidence examples, and representative posts.
- For each candidate interest, attach canonical_tags chosen only from the provided tag clusters.
- description must be 1 sentence of natural language.
- Exclude non-interest attributes such as family roles, career identity, and age range.
- Treat memes, reaction images, screenshots, and generic reposted aesthetic content as weak evidence by default.
- Do not infer a stable interest, ownership, or hobby from a single image when it could be a joke, repost, borrowed scene, or someone else's pet/car/food.
- For visual evidence, describe only what is directly visible. A photo of purchased food supports eating or dining evidence, not necessarily cooking.
- Only promote animal-, car-, fashion-, or food-related visuals into candidate interests when repeated cross-post evidence shows clear personal and consistent engagement.
- If a stable-looking canonical tag is ambiguous, slang-heavy, or obviously inherited from upstream clustering rather than real user evidence, you may reject it and leave the bucket empty or keep it weak.
- If evidence is sparse or fragmented, say so explicitly.
- Return strict JSON only.

Schema:
{
  "schema_version": "domain_evidence_first_v2",
  "domain": string,
  "candidate_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "evidence_post_ids": [string],
      "reasoning": string,
      "time_hint": "stable_candidate" | "recent_candidate" | "weak_candidate"
    }
  ],
  "preliminary_summary": string,
  "evidence_gaps": string or null
}
"""

DOMAIN_EVIDENCE_FIRST_JSON_SCHEMA_HINT = """{
  "schema_version": "domain_evidence_first_v2",
  "domain": string,
  "candidate_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "evidence_post_ids": [string],
      "reasoning": string,
      "time_hint": "stable_candidate" | "recent_candidate" | "weak_candidate"
    }
  ],
  "preliminary_summary": string,
  "evidence_gaps": string or null
}"""

DOMAIN_LLM_SYSTEM_PROMPT = """You are calibrating a domain-level interest profile.

Your job:
1) Preserve natural-language candidate interests from pass1 whenever evidence supports them.
2) Use algorithmic statistics only to place interests into stable / short-term / weak buckets.
3) Keep canonical_tags as anchors, but do not use raw canonical ids in labels or summaries.
4) Write domain_summary as 2-4 natural sentences suitable for downstream personalization benchmarking.

Hard constraints:
- Every interest item must contain both:
  - label (natural language)
  - canonical_tags (from provided clusters only)
- domain_summary should mention only labels that appear in structured fields.
- Do not output metric dumps like support_posts=...
- Exclude family roles, career identity, and age range from interest output.
- Do not treat memes, reaction images, screenshots, or generic reposts as strong evidence unless the user's personal engagement is repeated across posts.
- Do not infer pet ownership, cooking, driving, collecting, or other hands-on hobbies from a single ambiguous image.
- When images show animals, food, vehicles, or aesthetics, require repeated personal-context evidence before turning them into stable interests.
- Do not default to the top canonical tag when it is short, ambiguous, slang-like, or obviously broader/blunter than the evidence. Prefer a richer natural-language label grounded in representative posts and evidence examples.
- Labels should usually be more informative than any single cluster id. Avoid outputs like `cook` or `cat` unless the evidence literally supports that exact label and nothing more specific is justified.
- If no reliable interest can be extracted, explain that in plain language.

Return strict JSON only in this schema:
{
  "schema_version": "domain_llm_summary_v3",
  "domain": string,
  "stable_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "short_term_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "weak_or_uncertain_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "domain_all_evidence_post_ids": [string],
  "domain_representative_evidence_post_ids": [string],
  "domain_summary": string,
  "uncertainty_note": string or null
}
"""

DOMAIN_LLM_JSON_SCHEMA_HINT = """{
  "schema_version": "domain_llm_summary_v3",
  "domain": string,
  "stable_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "short_term_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "weak_or_uncertain_interests": [
    {
      "label": string,
      "canonical_tags": [string],
      "description": string,
      "confidence": number,
      "all_evidence_post_ids": [string],
      "representative_evidence_post_ids": [string],
      "reasoning": string
    }
  ],
  "domain_all_evidence_post_ids": [string],
  "domain_representative_evidence_post_ids": [string],
  "domain_summary": string,
  "uncertainty_note": string or null
}"""


def build_domain_evidence_first_user_prompt(evidence_pack_json: str) -> str:
    return f"""Extract evidence-first interests for one domain.

EVIDENCE_PACK_JSON:
{evidence_pack_json}

Instructions:
- Infer candidate interests from repeated semantic evidence in posts and tag clusters.
- Use natural-language labels and one-sentence descriptions.
- Keep claims grounded in evidence_post_ids and canonical_tags.
- Discount meme-like, reaction-image, reposted, or ambiguous visual posts unless personal engagement is repeated and explicit.
- Avoid inferring ownership or hands-on activity from visuals alone.
- Prefer labels that synthesize the repeated evidence pattern rather than restating the canonical tag. Representative posts and evidence examples should be the tie-breaker when cluster ids are too broad or too raw.
- If evidence is weak, explain the specific gap.
- Return JSON only.
"""


def build_domain_llm_user_prompt(final_input_json: str) -> str:
    return f"""Calibrate final domain summary from evidence-first output plus algorithmic summary.

FINAL_INPUT_JSON:
{final_input_json}

Instructions:
- Preserve pass1 wording when it is evidence-grounded.
- Use algorithmic signals to adjust bucket placement, not to rewrite labels into canonical ids.
- Prefer natural-language labels and descriptions.
- Keep canonical_tags only as internal anchors.
- Downweight meme-like, reposted, and ambiguous visual evidence.
- Reject labels that depend mainly on inferred ownership, inferred activity, or generic aesthetic reposts.
- If the strongest algorithmic cluster is still too raw or ambiguous, choose a clearer evidence-grounded label from the repeated posts instead of copying the cluster surface form.
- Return JSON only.
"""


BENCHMARK_GOLD_REWRITE_SYSTEM_PROMPT = """You are rewriting an internal domain analysis into a benchmark gold summary for user profiling.

Goal:
- Convert an internal analysis-style summary into a natural-language profile summary suitable for evaluating LLM personalization ability.
- Keep the summary faithful to the analysis.
- Preserve caution, but reduce audit / pipeline wording.
- Make the result sound like a concise user-interest profile, not a system report.

You will receive:
- domain name and definition
- domain status: active / weak / inactive
- long_term_interests
- short_term_interests
- evidence examples
- original internal summary
- uncertainty note

Core requirements:
- Do NOT invent any new interests, hobbies, personality traits, demographics, or motivations.
- Treat long_term_interests and short_term_interests as the main factual anchors.
- You may reuse supported wording, emphasis, and secondary themes from source_domain_summary, but only when they are clearly supported by the provided evidence.
- Any extra detail from source_domain_summary must remain secondary and must not overshadow the main structured interests.
- Do not upgrade weak evidence into a firm preference.
- Do not turn inactive domains into meaningful preference descriptions.

Writing rules by status:

For active domains:
- Center the summary on the main long-term interests.
- Mention short-term interests if present.
- You may add secondary themes from source_domain_summary only as supporting detail.
- The summary should read like a profile of what the user tends to like, not a checklist of extracted topics.
- Do not overload the summary with too many side details; keep the focus on the main interests.

For weak domains:
- Do not force formal interest labels if the evidence is limited.
- You may mention the concrete theme suggested by source_domain_summary if it is clearly supported.
- Make clear that the signal is limited, occasional, or too sparse for a firm profile.
- The tone should be cautious but still profile-oriented.

For inactive domains:
- State that this domain does not currently provide a reliable profile signal.
- If source_domain_summary clearly explains why, you may briefly preserve that explanation.
- Keep inactive summaries short.
- Do not use audit-style phrasing such as "within the observation window", "no relevant posts were found", "for personalization purposes", or similar pipeline language.

Style requirements:
- Keep the tone concise, human-readable, and profile-oriented.
- Prefer 2-4 sentences.
- Avoid internal jargon such as support_posts, tag_clusters, weak_signal, stable_score.
- Avoid sounding like an annotation report, validator note, or benchmark instruction.
- Prefer natural preference language such as "shows interest in", "tends to engage with", "shows limited signal around", "does not currently provide a reliable signal" when appropriate.

Output requirements:
- Return strict JSON only.

Schema:
{
  "summary_natural_gold": string
}
"""

BENCHMARK_GOLD_REWRITE_JSON_SCHEMA_HINT = """{
  "summary_natural_gold": string
}"""


def build_benchmark_gold_rewrite_user_prompt(rewrite_input_json: str) -> str:
    return f"""Rewrite the domain summary into benchmark-gold natural language.

REWRITE_INPUT_JSON:
{rewrite_input_json}

Instructions:
- Keep the meaning faithful to the analysis.
- Preserve caution where evidence is sparse.
- Reduce audit tone and make it sound more like a user profile summary.
- Use structured interests as the main anchors.
- If you use extra detail from source_domain_summary, keep it secondary.
- For inactive domains, keep the wording brief and natural rather than procedural.
- Return JSON only.
"""
