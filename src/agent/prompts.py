"""Wording guardrails for FabPilot investigation summaries.

The summary is assembled deterministically in ``nodes.py`` (there is no LLM
call), but FabPilot keeps five categories strictly separate so a reader can tell
model evidence apart from interpretation:

- ``observed_signals``      -- direct outputs from models, SHAP, or input data
- ``model_outputs``         -- prediction labels, probabilities, classes, scores
- ``hypotheses``            -- possible explanations that need engineer validation
- ``recommended_checks``    -- practical next steps for review
- ``human_review_required`` -- flag raised on low confidence or missing evidence

A hypothesis must never be promoted into an observed signal, and no summary
should claim a confirmed physical root cause.
"""

# Appended to every generated summary to keep the framing investigation-oriented.
SAFE_INTERPRETATION_NOTE = (
    "These are model signals and investigation hypotheses, not confirmed physical "
    "root causes. Validate with process engineering before acting."
)
