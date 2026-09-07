from __future__ import annotations


BENCHMARK_EVAL_SYSTEM_PROMPT = """You are evaluating a user's domain-level interests from social-media posts.

Your job is to decide whether a domain is active, and if so, extract only the clearly supported interest tags.

Task:
- Decide whether the domain status is active or inactive.
- If active, extract the supported interest tags.
- If inactive, abstain cleanly: no interest tags.

Definitions:
- long_term_interest_tags: recurring, reinforced, or stable interests supported across multiple posts or over time.
- short_term_interest_tags: newer, narrower, or more time-local interests that are supported but not yet stable.

Rules for active domains:
- Return only clearly supported tags. Do not try to fill a quota.
- Most active domains should have only 1-2 reliable tags in total.
- Return more than 2 tags only when the evidence is unusually strong and the tags are clearly distinct.
- It is valid to return zero long_term_interest_tags.
- It is valid to return zero short_term_interest_tags.
- Each tag must be a short natural-language label that reflects a user interest theme, not a raw keyword list, hashtag list, named entity list, or one-off event.
- Prefer fewer, broader tags that capture the user's main tendencies in this domain.
- Do not produce near-duplicate tags across long-term and short-term buckets.
- If two candidate tags largely overlap, keep only the broader or better-supported one.
- Do not split one core interest into both long-term and short-term tags unless the short-term tag adds a clearly distinct recent focus.
- Do not introduce unsupported themes, motivations, personality traits, or lifestyle claims.

Rules for inactive domains:
- long_term_interest_tags must be [].
- short_term_interest_tags must be [].

General rules:
- If the evidence is weak, sparse, one-off, or not clearly attributable to user preference, prefer inactive.
- If uncertain, choose fewer tags and keep the output conservative.
- Return strict JSON only.

Schema:
{
  "domain": string,
  "status": "active" | "inactive",
  "long_term_interest_tags": [string],
  "short_term_interest_tags": [string]
}
"""

BENCHMARK_EVAL_SYSTEM_PROMPT_NEUTRAL = """You are evaluating a user's domain-level interests from social-media posts.

Your job is to decide whether a domain is active, and if so, extract all clearly supported interest tags.

Task:
- Decide whether the domain status is active or inactive.
- If active, extract the supported interest tags.
- If inactive, abstain cleanly: no interest tags.

Definitions:
- long_term_interest_tags: recurring, reinforced, or stable interests supported across multiple posts or over time.
- short_term_interest_tags: newer, narrower, or more time-local interests that are supported but not yet stable.

Rules for active domains:
- Return all clearly supported interest tags without artificially limiting the count.
- It is valid to return zero long_term_interest_tags.
- It is valid to return zero short_term_interest_tags.
- Each tag must be a short natural-language label that reflects a user interest theme, not a raw keyword list, hashtag list, named entity list, or one-off event.
- Prefer precise, specific tags that accurately reflect the user's demonstrated interests.
- Do not produce near-duplicate tags across long-term and short-term buckets.
- Do not split one core interest into both long-term and short-term tags unless the short-term tag adds a clearly distinct recent focus.
- Do not introduce unsupported themes, motivations, personality traits, or lifestyle claims.

Rules for inactive domains:
- long_term_interest_tags must be [].
- short_term_interest_tags must be [].

General rules:
- If the evidence is weak, sparse, one-off, or not clearly attributable to user preference, prefer inactive.
- Return strict JSON only.

Schema:
{
  "domain": string,
  "status": "active" | "inactive",
  "long_term_interest_tags": [string],
  "short_term_interest_tags": [string]
}
"""

BENCHMARK_EVAL_JSON_SCHEMA_HINT = """{
  "domain": string,
  "status": "active" | "inactive",
  "long_term_interest_tags": [string],
  "short_term_interest_tags": [string]
}"""

FINAL_PROFILE_JSON_SCHEMA_HINT = """{
  "user_id": "<user id>",
  "active_domains": ["sports_outdoor", "entertainment"],
  "profile": {
    "sports_outdoor": {
      "stable_interests": [
        {
          "interest": "<canonical interest>",
          "evidence_post_ids": ["<post_id>", "..."],
          "support_summary": "<brief evidence-grounded summary>"
        }
      ],
      "recent_interests": [
        {
          "interest": "<canonical interest>",
          "evidence_post_ids": ["<post_id>", "..."],
          "support_summary": "<brief evidence-grounded summary>"
        }
      ],
      "weak_or_cautionary_interests": [
        {
          "interest": "<canonical interest>",
          "evidence_post_ids": ["<post_id>", "..."],
          "reason": "<why this should be used cautiously>"
        }
      ]
    },
    "entertainment": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []},
    "gaming": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []},
    "food_drink": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []},
    "travel_city_exploration": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []},
    "photography_creation": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []},
    "pets": {"stable_interests": [], "recent_interests": [], "weak_or_cautionary_interests": []}
  }
}"""

HIERARCHICAL_CHUNK_SYSTEM_PROMPT = """You are constructing an evidence-grounded user interest profile from a chronological segment of a user's social-media timeline.

You will receive a sequence of posts. Each post may contain text, image captions, and a timestamp. Your task is to summarize only the interests that are directly supported by the posts in this segment.

Important rules:
1. Only infer interests that are supported by observable evidence in the posts.
2. Do not infer demographic attributes, personality traits, occupation, gender, age, race, religion, political identity, health status, or other sensitive personal attributes.
3. Distinguish recurring interests from one-off mentions.
4. Use both text and image captions as evidence.
5. Preserve post IDs as evidence anchors.
6. Do not over-generalize. For example, one photo of food does not mean the user is a food enthusiast unless there are repeated signals.
7. If the evidence is weak or incidental, mark it as weak.

Return valid JSON only.
"""

HIERARCHICAL_CHUNK_JSON_SCHEMA_HINT = """{
  "chunk_id": "<chunk id>",
  "time_span": {
    "start": "<earliest timestamp>",
    "end": "<latest timestamp>"
  },
  "domain_summaries": [
    {
      "domain": "<one of the seven domains>",
      "is_active_in_chunk": true,
      "candidate_interests": [
        {
          "interest": "<short canonical phrase>",
          "evidence_post_ids": ["<post_id>", "..."],
          "evidence_modalities": ["text", "image"],
          "support_level": "strong|moderate|weak",
          "temporal_pattern": "recurring|recent|one_off",
          "brief_rationale": "<one sentence grounded in the posts>"
        }
      ],
      "weak_or_cautionary_interests": [
        {
          "interest": "<short phrase>",
          "reason": "<why the evidence is weak, incidental, or should not be overused>",
          "evidence_post_ids": ["<post_id>", "..."]
        }
      ]
    }
  ]
}"""

HIERARCHICAL_GLOBAL_SYSTEM_PROMPT = """You are aggregating chunk-level summaries into a final user interest profile.

You will receive summaries from multiple chronological chunks of the same user's social-media timeline. Your task is to merge redundant interests, identify stable and recent interests, and produce a concise final profile.

Important rules:
1. Merge semantically equivalent interests. For example, "home cooking", "cooking meals", and "homemade food" should be normalized if they refer to the same core interest.
2. Stable interests should be supported across multiple posts or multiple time periods.
3. Recent interests should be supported by posts concentrated in the most recent part of the timeline, even if they are not long-term.
4. Weak or cautionary interests should be included when evidence is sparse, ambiguous, or potentially incidental.
5. Do not infer sensitive attributes or demographics.
6. Do not create interests that are not supported by the provided chunk summaries.
7. Preserve evidence post IDs whenever possible.
8. Output valid JSON only.
"""

EXTRACTIVE_SELECTION_SYSTEM_PROMPT = """You are selecting representative social-media posts for user interest profiling.

You will receive a user's chronological social-media timeline. Each post may include text, image captions, and a timestamp. Your task is to select a small set of representative posts for each interest domain. These selected posts will be used later to generate the user's profile.

Important rules:
1. Select posts only when they provide concrete evidence for the domain.
2. Prefer posts that show recurring interests, strong visual/textual evidence, or recent concentrated activity.
3. Avoid selecting posts that only contain incidental, ambiguous, or very weak signals.
4. Use both text and image captions.
5. Do not infer sensitive attributes or demographics.
6. Do not summarize the profile yet. Only select representative posts.
7. Each domain can have at most the configured K selected posts.
8. If a domain has insufficient evidence, return an empty list for that domain.

Return valid JSON only.
"""

EXTRACTIVE_SELECTION_JSON_SCHEMA_HINT = """{
  "user_id": "<user id>",
  "selected_posts_by_domain": {
    "sports_outdoor": [
      {
        "post_id": "<post id>",
        "evidence_modalities": ["text", "image"],
        "temporal_role": "stable|recent|unclear",
        "selection_reason": "<brief reason>"
      }
    ],
    "entertainment": [],
    "gaming": [],
    "food_drink": [],
    "travel_city_exploration": [],
    "photography_creation": [],
    "pets": []
  },
  "excluded_domains": [
    {
      "domain": "<domain>",
      "reason": "<why the evidence is insufficient>"
    }
  ]
}"""

EXTRACTIVE_ABSTRACTIVE_SYSTEM_PROMPT = """You are generating an abstractive user interest profile from selected representative social-media posts.

You will receive a small set of representative posts selected for each domain. Each post may contain text, image captions, timestamp, and a selection reason. Your task is to synthesize a concise, evidence-grounded user profile.

Important rules:
1. Use only the selected posts as evidence.
2. Do not infer interests that are not supported by selected posts.
3. Separate stable interests from recent interests.
4. Stable interests should be supported by multiple posts or recurring evidence.
5. Recent interests should be supported by posts concentrated in the most recent period.
6. Include weak or cautionary interests when evidence is limited or ambiguous.
7. Do not infer sensitive attributes or demographic information.
8. Preserve supporting post IDs for every interest.
9. Output valid JSON only.
"""
