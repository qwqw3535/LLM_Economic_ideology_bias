from __future__ import annotations

ICL_EXPERIMENT_PROMPT = """# Role
You are an expert economist.

# Task
You are given reference examples related to the target treatment-outcome pair.
Predict the most likely sign for the target pair in the target context.

# Sign Definitions
- '+': The treatment increases the outcome.
- '-': The treatment decreases the outcome.
- 'None': No statistically significant effect.
- 'mixed': The effect varies across subgroups or specifications.

# Reference Examples
{examples}

# Target
Treatment: {treatment}
Outcome: {outcome}
Context: {context_new}

# Output Format
Respond with exactly one label and nothing else:
+ , - , None , or mixed

Answer:
"""

