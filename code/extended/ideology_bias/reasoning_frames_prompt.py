"""Prompt template for reasoning-frame annotation."""

REASONING_FRAMES_PROMPT_TEMPLATE = """You are labeling economic reasoning frames in model explanations.

Controlled vocabulary:
- efficiency
- incentives
- market_distortion
- productivity
- redistribution
- insurance
- externalities
- fiscal_burden
- state_capacity
- other

Task:
- Read the reasoning text.
- Choose exactly one `primary_frame`.
- Choose up to three `secondary_frames`.
- Use ONLY the controlled vocabulary.
- If the text is generic or not clearly captured by the vocabulary, use `other`.

Rules:
- Focus on the reasoning, not the correctness of the prediction.
- Do not invent frames that are not present in the text.
- `secondary_frames` may be empty.
- Output JSON only.

Return:
{
  "reasoning_frames": {
    "primary_frame": "...",
    "secondary_frames": [],
    "justification": "short evidence-based explanation"
  }
}

Task: {task_name}
Model family: {family}
Model: {model}
Reasoning:
{reasoning}
"""

