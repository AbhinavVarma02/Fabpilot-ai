# FabPilot AI

FabPilot AI is a decision-support demo for semiconductor yield investigation. It brings process-sensor risk, wafer-map classification, local feature explanations, and human-review routing into one Streamlit workflow.

**Live Demo:** [https://huggingface.co/spaces/AbhinavVarma/fabpilot-ai](https://huggingface.co/spaces/AbhinavVarma/fabpilot-ai)

The project uses public SECOM and WM-811K data. It is designed to organize model evidence and investigation hypotheses, not to claim confirmed physical root causes.

## What it does

- Estimates pass or failure risk from anonymized SECOM process-sensor features.
- Classifies WM-811K wafer maps into nine defect-pattern classes.
- Shows the local sensor-feature contributions behind each yield prediction.
- Routes uncertain wafer predictions to human review using confidence and top-two class margin.
- Produces a structured investigation summary that keeps model outputs separate from hypotheses and recommended checks.

## How it works

1. A saved scikit-learn pipeline handles SECOM preprocessing and yield-risk inference.
2. A local linear SHAP-style calculation explains the saved Logistic Regression model in log-odds space.
3. A compact PyTorch CNN classifies a 64 by 64 wafer map.
4. A LangGraph workflow runs the models in a controlled sequence, applies review rules, and assembles the evidence.
5. Streamlit presents the results and lets the user download the investigation summary.

Everything runs from local artifacts and bundled demo samples. The app does not call an external LLM or require an API key.

## Human-review design

FabPilot AI does not treat every prediction as equally reliable. A wafer prediction is accepted automatically only when:

- softmax confidence is at least `0.40`, and
- the margin between the top two classes is at least `0.10`.

Lower-certainty cases are marked for human review. Missing sensor or wafer evidence also triggers review. These are demonstration rules for selective prediction, not calibrated production thresholds.

## Reproducible wafer evaluation

The included evaluation script uses the same model, demo sample, inference path, and review rule as the application:

```bash
python scripts/evaluate_wafer_review.py
```

Current results on the 180-map public demo sample:

| Measure | Result |
| --- | ---: |
| Overall benchmark accuracy | 57.8% (104/180) |
| Automatically accepted | 85 maps |
| Accuracy on accepted subset | 88.2% (75/85) |
| Routed to human review | 95 maps |

The 88.2% result applies only to the accepted subset. It is not the overall CNN accuracy. The generated audit is available in [`reports/wm811k_review_audit.md`](reports/wm811k_review_audit.md).

## Model results

SECOM yield model on its held-out test split:

| Metric | Result |
| --- | ---: |
| Failure precision | 0.098 |
| Failure recall | 0.190 |
| Failure F1 | 0.129 |
| ROC-AUC | 0.623 |
| PR-AUC | 0.122 |

WM-811K CNN baseline on its held-out balanced-sample test split:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.545 |
| Macro F1 | 0.522 |
| Macro recall | 0.565 |

These are baseline portfolio results, not production performance claims.

## Project structure

```text
app/                    Streamlit application and UI
artifacts/              Small deployment-ready model artifacts
data/demo/              Compact public inference samples
reports/                Public audit and dashboard screenshot
scripts/                Reproducible evaluation script
src/agent/              LangGraph state, nodes, and review workflow
src/data/               Data loading and preprocessing helpers
src/explainability/     Local feature-explanation logic
src/models/             Yield and wafer model definitions
src/utils/              Evaluation utilities
tests/                  Review-rule test
```

Raw and processed datasets are intentionally excluded from the repository.

## Run locally

FabPilot AI was developed with Python 3.13 on Windows. The Docker image uses Python 3.11.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies and start the dashboard:

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`, choose a sensor row and wafer sample, then select **Run investigation**.

## Run with Docker

```bash
docker build -t fabpilot-ai .
docker run --rm -p 7860:7860 fabpilot-ai
```

Open `http://localhost:7860`.

## Included runtime assets

The dashboard uses four compact public assets:

- `artifacts/secom_yield_model.joblib`
- `artifacts/wafer_cnn.pt`
- `data/demo/secom_demo_features.csv`
- `data/demo/wm811k_demo_sample.npz`

Git LFS tracks the binary model, NPZ, and image files. Raw SECOM and WM-811K datasets are not included.

## Limitations

- SECOM features are anonymized, so explanations identify sensor variables rather than physical process causes.
- SHAP-style contributions explain model behavior, not causality.
- WM-811K is a public benchmark and does not represent company-specific fab data.
- The wafer CNN is a compact baseline trained on a balanced sample, not the full dataset.
- The SECOM model has weak failure-class performance and should not be used for autonomous decisions.
- Human validation is required before acting on any output or hypothesis.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the concise safety statement.

## License

This project is available under the [MIT License](LICENSE).
