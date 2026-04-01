from __future__ import annotations

MAIN_RESULTS_PROMPT = """# Role
You are an expert economist.

# Task
Given a context and a treatment-outcome pair, predict the most likely sign of the causal effect.

# Sign Definitions
- '+': The treatment increases the outcome.
- '-': The treatment decreases the outcome.
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Context
{context}

# Treatment-Outcome Pair
Treatment: {treatment}
Outcome: {outcome}

# Output
Return exactly one of these labels:
+
-
None
mixed
Do not output any explanation, punctuation, or extra words.

Label:
"""

