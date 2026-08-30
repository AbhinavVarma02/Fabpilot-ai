# Limitations

FabPilot AI is a public-dataset MVP for semiconductor yield investigation support.

- SECOM features are anonymized process-sensor variables. They should not be interpreted as direct physical root causes.
- SHAP-style explanations describe model behavior for a sample. They do not prove causality.
- WM-811K is a public benchmark wafer-map dataset and does not represent company-specific fab data.
- The current wafer CNN is trained on a compact balanced sample for a working MVP baseline.
- The LangGraph workflow organizes structured evidence and hypotheses. It does not make model predictions directly.
- Investigation summaries are generated deterministically (no external LLM call), so they only restate and organize the structured model evidence rather than reasoning beyond it.
- Human validation is required before acting on any model output or investigation hypothesis.
