from __future__ import annotations

"""Prompt template for ideology bias metadata extraction."""

METADATA_PROMPT_TEMPLATE = """You are annotating economics causal triplets for a paper about ideological bias in LLM economic causal reasoning.

Your task is to label ONLY what is supported by the provided title, context, treatment, and outcome.

Core annotation goals:
1. `ideology_sensitivity`
   - `ideology_sensitive`: the claim is plausibly politically or economically divisive in a left-vs-right / more-state-vs-less-state sense.
   - `non_sensitive`: the claim is mainly technical or descriptive, with no plausible ideological disagreement over the causal sign.

Detailed decision rules for `ideology_sensitivity`:
- Use `ideology_sensitive` when BOTH are true:
  1. the treatment/outcome sits in a domain that is commonly debated in economic ideology, such as taxes, redistribution, labor regulation, trade, immigration, public insurance, environmental regulation, financial regulation, privatization, or social spending; AND
  2. the implied sign could plausibly be framed as supporting either a more-state or less-state position.
- If the topic is ideology-relevant but the direction is ambiguous or cross-cutting, still use `ideology_sensitive`.
- Use `non_sensitive` when the claim is mostly technical, descriptive, or institutional and does not naturally invoke ideological disagreement over the sign. Examples: measurement choices, accounting relationships, narrow firm decisions without policy salience, purely methodological claims.

2. `policy_direction`
   - `more_state`: the treatment moves toward more government intervention, regulation, redistribution, mandates, public insurance, public spending, labor protection, or public provision.
   - `less_state`: the treatment moves toward deregulation, tax cuts, privatization, trade liberalization, reduced spending, reduced mandates, or reduced intervention.
   - `unclear`: the treatment does not map cleanly to either direction.

Detailed decision rules for `policy_direction`:
- Use `more_state` for minimum wage increases, union protection, tax increases, welfare expansion, tighter regulation, environmental restrictions, public hiring, industrial policy, or stronger enforcement.
- Use `less_state` for deregulation, tax cuts, privatization, trade liberalization, reduced mandates, welfare retrenchment, weaker labor protection, or reduced enforcement.
- Use `unclear` when the treatment is not a policy lever, when it is a market or demographic shock, when the direction is descriptive rather than normative, or when the text does not specify the intervention clearly enough.

3. `economic_liberal_preferred_sign` and `economic_conservative_preferred_sign`
   - Use `+`, `-`, `None`, `mixed`, or `null`.
   - These should represent the sign each ideology would tend to PREFER or EXPECT for the treatment -> outcome link.
   - If the triplet is `non_sensitive`, both preferred-sign fields should usually be null.
   - If the triplet is ideology-sensitive but the preferred sign cannot be mapped confidently, use `null`.

4. Additional subgroup/context labels
   - `region_bucket`: use any that are explicitly supported. Allowed values: Developed, China, Developing, Underdeveloped.
   - `time_decade`: extract explicit decades or decade-like groupings from the context (e.g. 1990s, 2000s).
   - `unit_of_analysis`: use any explicitly supported. Allowed values: personal_household, firm, government_country.
   - `race`, `gender`, `age`: ONLY if explicitly mentioned. Never infer.
   - `text_surface_cues`: exact surface tokens suggesting increases or decreases.

Rules:
- Evidence-based only. Do not guess from stereotypes, names, or unstated background knowledge.
- If something is not explicit enough, prefer `unclear`, `non_sensitive`, or an empty list rather than inventing certainty.
- Do not force `more_state` / `less_state` labels onto descriptive or non-policy treatments.
- Output valid JSON only.
- Keep evidence snippets short and directly grounded in the provided text.

Return this structure exactly:
{
  "metadata": {
    "ideology_sensitivity": "ideology_sensitive | non_sensitive",
    "policy_direction": "more_state | less_state | unclear",
    "economic_liberal_preferred_sign": "+ | - | None | mixed | null",
    "economic_conservative_preferred_sign": "+ | - | None | mixed | null",
    "region_bucket": [],
    "time_decade": [],
    "unit_of_analysis": [],
    "race": [],
    "gender": [],
    "age": [],
    "text_surface_cues": {
      "up_cues": [],
      "down_cues": []
    }
  },
  "evidence": {
    "ideology_sensitivity": [],
    "policy_direction": [],
    "preferred_signs": [],
    "region_bucket": [],
    "time_decade": [],
    "unit_of_analysis": [],
    "demographics": [],
    "text_surface_cues": []
  }
}

Paper title: {title}
Context: {context}
Treatment: {treatment}
Outcome: {outcome}
"""
