"""Presentation layer for the FabPilot AI dashboard.

Pure UI only: theme CSS, HTML card/panel rendering, and matplotlib styling.
This module contains no model, agent, SHAP, or preprocessing logic -- it just
renders the structured evidence produced by the LangGraph workflow.
"""

from __future__ import annotations

import html as _html
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from src.agent.nodes import (
    WAFER_REVIEW_CONFIDENCE_THRESHOLD,
    WAFER_REVIEW_MARGIN_THRESHOLD,
)

# --- Palette -----------------------------------------------------------------
INK = "#0F172A"
MUTED = "#64748B"
BRAND = "#2563EB"
TEAL = "#0D9488"
VIOLET = "#7C3AED"
OK = "#16A34A"
BAD = "#DC2626"
WARN = "#D97706"

FAIL_THRESHOLD = 0.50    # display-only mirror of the yield model decision rule


CSS = """
<style>
:root { --fp-ink:#0F172A; --fp-muted:#64748B; --fp-border:#E2E8F0; }
.block-container { max-width: 1200px; padding-top: 1.1rem; padding-bottom: 3rem; }
[data-testid="stHeader"] { background: transparent; }
html, body, [class*="css"] { font-feature-settings: "cv11", "ss01"; }

.stButton > button {
  border-radius: 10px; font-weight: 700; border: none; padding: .55rem 1rem;
  background: linear-gradient(135deg,#2563EB,#1D4ED8); color: #fff;
  box-shadow: 0 8px 18px -8px rgba(37,99,235,.65);
}
.stButton > button:hover { filter: brightness(1.06); box-shadow: 0 10px 22px -8px rgba(37,99,235,.75); }

/* Hero */
.fp-hero {
  position: relative; overflow: hidden; border-radius: 20px;
  padding: 2.3rem 2.5rem; margin: 0 0 1.3rem;
  background:
    radial-gradient(1100px 380px at 12% -30%, rgba(56,189,248,.35), transparent 60%),
    linear-gradient(135deg,#0B1220 0%,#132a52 46%,#1D4ED8 100%);
  color: #fff; box-shadow: 0 20px 42px -20px rgba(29,78,216,.6);
}
.fp-hero-kicker { font-size: .72rem; font-weight: 800; letter-spacing: .16em; color: #7DD3FC; }
.fp-hero-title { font-size: 2.25rem; font-weight: 800; margin: .35rem 0 .55rem; letter-spacing: -.01em; }
.fp-hero-sub { color: #CBD5E1; max-width: 660px; font-size: 1rem; line-height: 1.55; margin: 0; }
.fp-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.05rem; }
.fp-badge {
  font-size: .78rem; font-weight: 600; color: #E2E8F0;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.20);
  border-radius: 999px; padding: .3rem .72rem;
}
.fp-hero-note { margin-top: 1rem; font-size: .8rem; color: #93C5FD; font-weight: 600; }

/* Metric cards */
.fp-metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin: 4px 0 8px; }
@media (max-width: 860px) { .fp-metrics { grid-template-columns: repeat(2,1fr); } }
.fp-metric {
  position: relative; background: #fff; border: 1px solid var(--fp-border);
  border-radius: 16px; padding: 15px 16px 14px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.fp-metric::before {
  content: ""; position: absolute; left: 0; top: 15px; bottom: 15px; width: 4px;
  border-radius: 0 4px 4px 0; background: var(--accent,#2563EB);
}
.fp-metric-label { font-size: .71rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #64748B; }
.fp-metric-value { font-size: 1.55rem; font-weight: 800; color: #0F172A; margin: .18rem 0 .06rem; line-height: 1.15; }
.fp-metric-sub { font-size: .77rem; color: #94A3B8; }
.fp-bar { height: 7px; border-radius: 6px; background: #EEF2F7; margin-top: .62rem; overflow: hidden; }
.fp-bar-fill { height: 100%; border-radius: 6px; }
.fp-metric-foot { font-size: .74rem; color: #64748B; margin-top: .42rem; font-weight: 600; }
.fp-metric-note { color: #64748B; font-size: .82rem; line-height: 1.5; margin: .35rem 0 1rem; }

/* Status banner */
.fp-status {
  display: flex; align-items: center; gap: .75rem; border: 1px solid;
  border-radius: 14px; padding: .8rem 1.1rem; margin: .4rem 0 1rem; font-weight: 700;
}
.fp-status--warn { background: #FEF3C7; border-color: #FDE68A; color: #92400E; }
.fp-status--ok { background: #DCFCE7; border-color: #BBF7D0; color: #166534; }
.fp-status-meta { margin-left: auto; font-weight: 600; font-size: .82rem; opacity: .85; }

/* Evidence panels */
.fp-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 6px; }
@media (max-width: 860px) { .fp-grid2 { grid-template-columns: 1fr; } }
.fp-panel {
  background: #fff; border: 1px solid var(--fp-border); border-left: 4px solid var(--accent,#2563EB);
  border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
}
.fp-panel-head { display: flex; align-items: center; gap: .5rem; margin-bottom: .5rem; }
.fp-panel-title { font-weight: 700; color: #0F172A; font-size: .97rem; }
.fp-panel-count { margin-left: auto; font-size: .72rem; font-weight: 700; color: #475569; background: #F1F5F9; border-radius: 999px; padding: .1rem .55rem; }
.fp-list { margin: 0; padding-left: 1.05rem; }
.fp-list li { color: #334155; font-size: .9rem; line-height: 1.5; margin: .22rem 0; }
.fp-empty-list { color: #94A3B8; font-style: italic; font-size: .86rem; }

/* Human review banner */
.fp-review { display: flex; gap: .9rem; align-items: flex-start; border: 1px solid; border-radius: 16px; padding: 1rem 1.2rem; margin-top: 14px; }
.fp-review--req { background: #FFFBEB; border-color: #FDE68A; }
.fp-review--ok { background: #F0FDF4; border-color: #BBF7D0; }
.fp-review-title { font-weight: 800; color: #0F172A; font-size: 1.02rem; }
.fp-review-sub { color: #475569; font-size: .88rem; margin-top: .15rem; line-height: 1.5; }

/* Tables, tags, code, notes */
code { background: #F1F5F9; border-radius: 6px; padding: .05rem .34rem; font-size: .82em; color: #1E293B; }
.fp-table { width: 100%; border-collapse: collapse; margin-top: .4rem; font-size: .86rem; }
.fp-table th { text-align: left; color: #64748B; font-weight: 700; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; padding: .3rem .4rem; border-bottom: 1px solid #E2E8F0; }
.fp-table td { padding: .4rem .4rem; border-bottom: 1px solid #F1F5F9; color: #334155; }
.fp-num { font-variant-numeric: tabular-nums; font-weight: 700; }
.fp-tag { font-size: .71rem; font-weight: 700; border-radius: 999px; padding: .12rem .55rem; white-space: nowrap; }
.fp-tag--bad { background: #FEE2E2; color: #B91C1C; }
.fp-tag--ok { background: #DCFCE7; color: #15803D; }
.fp-note { color: #64748B; font-size: .82rem; line-height: 1.55; border-top: 1px dashed #E2E8F0; margin-top: 1rem; padding-top: .8rem; }

/* Empty state */
.fp-empty { background: #fff; border: 1px dashed #CBD5E1; border-radius: 18px; padding: 1.7rem 1.9rem; height: 100%; }
.fp-kicker { display: inline-block; font-size: .7rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; color: #2563EB; background: #EFF6FF; border-radius: 999px; padding: .2rem .6rem; margin-bottom: .6rem; }
.fp-empty h3 { margin: .1rem 0 .35rem; color: #0F172A; font-size: 1.25rem; font-weight: 800; }
.fp-empty p { color: #64748B; margin: 0 0 1.1rem; font-size: .94rem; max-width: 640px; }
.fp-steps { display: grid; gap: .55rem; }
.fp-step { display: flex; align-items: center; gap: .7rem; color: #334155; font-size: .92rem; }
.fp-step b { flex: none; width: 1.55rem; height: 1.55rem; border-radius: 50%; background: #EFF6FF; color: #1D4ED8; font-weight: 800; font-size: .8rem; display: inline-flex; align-items: center; justify-content: center; }

/* Missing artifacts */
.fp-missing { background: #FEF2F2; border: 1px solid #FECACA; border-radius: 16px; padding: 1.3rem 1.5rem; }
.fp-missing h3 { color: #991B1B; margin: 0 0 .4rem; font-size: 1.1rem; font-weight: 800; }
.fp-missing p { color: #7F1D1D; font-size: .9rem; margin: 0 0 .6rem; }
.fp-missing code { background: #FEE2E2; color: #991B1B; }

/* Sidebar brand */
.fp-brand { display: flex; align-items: center; gap: .6rem; padding: .2rem 0 .3rem; }
.fp-brand-mark { width: 2.1rem; height: 2.1rem; border-radius: 10px; background: linear-gradient(135deg,#2563EB,#0EA5E9); color: #fff; font-weight: 800; display: inline-flex; align-items: center; justify-content: center; box-shadow: 0 6px 14px -6px rgba(37,99,235,.6); }
.fp-brand-name { font-weight: 800; color: #0F172A; font-size: 1.05rem; line-height: 1; }
.fp-brand-tag { font-size: .72rem; color: #64748B; }
.fp-side-note { font-size: .76rem; color: #94A3B8; line-height: 1.5; }
.fp-side-head { font-size: .72rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #64748B; margin: .2rem 0 .1rem; }
</style>
"""


def _md(html_block: str) -> None:
    """Render trusted HTML, stripping per-line indentation so Streamlit's
    markdown parser never mistakes an indented tag for a code block."""

    cleaned = "\n".join(line.strip() for line in html_block.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def _inline(text: str) -> str:
    """Escape a plain evidence string and render `backtick` spans as <code>."""

    safe = _html.escape(str(text))
    parts = safe.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def format_probability(value: float) -> str:
    score = float(value)
    if score < 0.001:
        return "<0.1%"
    if score > 0.999:
        return ">99.9%"
    return f"{score * 100:.1f}%"


def inject_css() -> None:
    _md(CSS)


def hero() -> None:
    _md(
        """
        <div class="fp-hero">
          <div class="fp-hero-kicker">APPLIED AI · SEMICONDUCTOR YIELD</div>
          <div class="fp-hero-title">FabPilot AI</div>
          <p class="fp-hero-sub">An agentic diagnosis copilot that turns process-sensor rows and wafer maps into
          structured, explainable yield-risk evidence, with a human in the loop for every uncertain case.</p>
          <div class="fp-badges">
            <span class="fp-badge">SECOM yield model</span>
            <span class="fp-badge">WM-811K wafer CNN</span>
            <span class="fp-badge">SHAP explanations</span>
            <span class="fp-badge">LangGraph workflow</span>
            <span class="fp-badge">Runs offline · no API key</span>
          </div>
          <div class="fp-hero-note">MVP decision-support demo · organizes model evidence and hypotheses · not a confirmed root-cause tool</div>
        </div>
        """
    )


def sidebar_brand() -> None:
    _md(
        """
        <div class="fp-brand">
          <div class="fp-brand-mark">Fp</div>
          <div>
            <div class="fp-brand-name">FabPilot AI</div>
            <div class="fp-brand-tag">Yield Diagnosis Copilot</div>
          </div>
        </div>
        """
    )


def sidebar_heading(text: str) -> None:
    _md(f'<div class="fp-side-head">{_html.escape(text)}</div>')


def sidebar_footer() -> None:
    _md(
        """
        <div class="fp-side-note">
        FabPilot organizes model evidence for review. It does not confirm physical root causes,
        and human validation is required before acting. Predictions come from trained model
        artifacts: the workflow only routes tools and assembles the summary.
        </div>
        """
    )


def missing_artifacts(missing: list[str]) -> None:
    items = "".join(f"<li><code>{_html.escape(m)}</code></li>" for m in missing)
    _md(
        f"""
        <div class="fp-missing">
          <h3>Local artifacts not found</h3>
          <p>The dashboard needs the trained models and the processed wafer sample. The following are missing:</p>
          <ul class="fp-list">{items}</ul>
          <p>Run the modeling notebooks (<code>02</code>, <code>04</code>, <code>05</code>) to regenerate them, then reload this page.</p>
        </div>
        """
    )


def empty_state() -> None:
    _md(
        """
        <div class="fp-empty">
          <span class="fp-kicker">Ready</span>
          <h3>Run a sample investigation</h3>
          <p>Pick a SECOM sensor row and a wafer-map sample in the control panel, then run the workflow.
          The graph combines model evidence into a single, reviewable diagnosis.</p>
          <div class="fp-steps">
            <div class="fp-step"><b>1</b> Yield-risk model scores the SECOM sensor row</div>
            <div class="fp-step"><b>2</b> SHAP explains the top sensor-feature contributions</div>
            <div class="fp-step"><b>3</b> CNN classifies the wafer-map defect pattern</div>
            <div class="fp-step"><b>4</b> Confidence check flags low-certainty cases</div>
            <div class="fp-step"><b>5</b> Structured summary + human-review decision</div>
          </div>
        </div>
        """
    )


def status_banner(state: dict[str, Any], secom_idx: int, wafer_idx: int) -> None:
    required = bool(state.get("human_review_required"))
    cls = "fp-status--warn" if required else "fp-status--ok"
    label = (
        "Investigation complete: human review recommended"
        if required
        else "Investigation complete: cleared, no low-confidence flag"
    )
    _md(
        f"""
        <div class="fp-status {cls}">
          <span>{label}</span>
          <span class="fp-status-meta">SECOM row {secom_idx} · Wafer sample {wafer_idx}</span>
        </div>
        """
    )


def _metric(label, value, sub, accent, *, bar=None, bar_color=None, foot=None) -> str:
    bar_html = ""
    if bar is not None:
        width = max(0.0, min(1.0, float(bar))) * 100
        bar_html = (
            f'<div class="fp-bar"><div class="fp-bar-fill" '
            f'style="width:{width:.0f}%;background:{bar_color or accent}"></div></div>'
        )
    foot_html = f'<div class="fp-metric-foot">{foot}</div>' if foot else ""
    return (
        f'<div class="fp-metric" style="--accent:{accent}">'
        f'<div class="fp-metric-label">{label}</div>'
        f'<div class="fp-metric-value">{_html.escape(str(value))}</div>'
        f'<div class="fp-metric-sub">{sub}</div>{bar_html}{foot_html}</div>'
    )


def metric_cards(state: dict[str, Any]) -> None:
    pred = state.get("yield_prediction") or "n/a"
    risk = float(state.get("yield_risk_score") or 0.0)
    yconf = float(state.get("yield_confidence") or 0.0)
    dclass = state.get("defect_class") or "n/a"
    dconf = float(state.get("defect_confidence") or 0.0)
    dmargin = float(state.get("defect_top2_margin") or 0.0)

    pred_accent = BAD if pred == "fail" else OK if pred == "pass" else MUTED
    risk_color = BAD if risk >= FAIL_THRESHOLD else OK
    wafer_clear = (
        dconf >= WAFER_REVIEW_CONFIDENCE_THRESHOLD
        and dmargin >= WAFER_REVIEW_MARGIN_THRESHOLD
    )
    dconf_color = OK if wafer_clear else WARN
    cards = "".join(
        [
            _metric("Yield Prediction", pred.upper(), "SECOM yield-risk model", pred_accent,
                    bar=yconf, bar_color=pred_accent, foot=f"decision confidence score {format_probability(yconf)}"),
            _metric("Estimated Fail Risk", format_probability(risk), "SECOM fail-class model score", risk_color,
                    bar=risk, bar_color=risk_color, foot=f"fail threshold {format_probability(FAIL_THRESHOLD)}"),
            _metric("Defect Pattern", dclass, "WM-811K wafer CNN", BRAND, foot="predicted spatial class"),
            _metric(
                "Softmax Confidence",
                format_probability(dconf),
                "wafer CNN model score",
                dconf_color,
                bar=dconf,
                bar_color=dconf_color,
                foot=(
                    f"MVP review below {format_probability(WAFER_REVIEW_CONFIDENCE_THRESHOLD)} "
                    f"or margin below {format_probability(WAFER_REVIEW_MARGIN_THRESHOLD)}"
                ),
            ),
        ]
    )
    _md(f'<div class="fp-metrics">{cards}</div>')
    _md(
        '<div class="fp-metric-note">Scores are model outputs for decision support, not calibrated production probabilities. '
        'The wafer confidence threshold is an MVP decision-support threshold based on deployed demo model behavior, '
        'not a production calibration standard. Low-confidence and small-margin cases are routed to human review.</div>'
    )


def _format_model_outputs(model_outputs: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for out in model_outputs:
        tool = out.get("tool", "tool")
        if tool == "yield_prediction_model":
            lines.append(
                f"Yield model: <b>{_html.escape(str(out.get('prediction')))}</b>, "
                f"model-estimated fail risk {format_probability(float(out.get('yield_risk_score', 0)))}, "
                f"decision confidence score {format_probability(float(out.get('confidence', 0)))}"
            )
        elif tool == "shap_explanation":
            lines.append(f"SHAP explainer: {len(out.get('top_features', []))} top sensor-feature contributions")
        elif tool == "wafer_defect_classifier":
            lines.append(
                f"Wafer CNN: <b>{_html.escape(str(out.get('defect_class')))}</b>, "
                f"softmax confidence {format_probability(float(out.get('confidence', 0)))}, "
                f"top-2 margin {format_probability(float(out.get('top2_margin', 0)))}"
            )
        else:
            lines.append(_html.escape(str(tool)))
    return lines


def _panel(title: str, accent: str, items: list[str], *, raw: bool = False) -> str:
    if items:
        body = '<ul class="fp-list">' + "".join(
            f"<li>{it if raw else _inline(it)}</li>" for it in items
        ) + "</ul>"
    else:
        body = '<div class="fp-empty-list">None recorded for this sample.</div>'
    return (
        f'<div class="fp-panel" style="--accent:{accent}">'
        f'<div class="fp-panel-head"><span class="fp-panel-title">{title}</span>'
        f'<span class="fp-panel-count">{len(items)}</span></div>{body}</div>'
    )


def evidence_sections(state: dict[str, Any]) -> None:
    observed = state.get("observed_signals") or []
    models = _format_model_outputs(state.get("model_outputs") or [])
    hypotheses = state.get("hypotheses") or []
    # De-dupe checks the same way the summary node does (display parity, not logic).
    checks = list(dict.fromkeys(state.get("recommended_checks") or []))

    grid = (
        '<div class="fp-grid2">'
        + _panel("Observed Signals", BRAND, observed)
        + _panel("Model Outputs", TEAL, models, raw=True)
        + _panel("Investigation Hypotheses", VIOLET, hypotheses)
        + _panel("Recommended Checks", OK, checks)
        + "</div>"
    )
    _md(grid)


def human_review(state: dict[str, Any]) -> None:
    required = bool(state.get("human_review_required"))
    if required:
        _md(
            f"""
            <div class="fp-review fp-review--req">
              <div>
                <div class="fp-review-title">Human Review Required</div>
                <div class="fp-review-sub">A model confidence check, wafer top-2 margin check, or evidence-availability check
                requested review. Route this sample to a process engineer before acting on it.</div>
              </div>
            </div>
            """
        )
    else:
        _md(
            f"""
            <div class="fp-review fp-review--ok">
              <div>
                <div class="fp-review-title">No Low-Confidence Flag</div>
                <div class="fp-review-sub">Both evidence sources were available, the SECOM confidence check passed, and the
                wafer result met the MVP confidence and margin checks. Human validation is still recommended before any action.</div>
              </div>
            </div>
            """
        )


def shap_table(shap_features: list[dict[str, Any]]) -> None:
    rows = ""
    for feat in shap_features:
        value = float(feat.get("shap_value", 0.0))
        toward_fail = feat.get("direction") == "pushes_toward_failure"
        tag = (
            '<span class="fp-tag fp-tag--bad">toward failure</span>'
            if toward_fail
            else '<span class="fp-tag fp-tag--ok">toward pass</span>'
        )
        rows += (
            f'<tr><td><code>{_html.escape(str(feat.get("feature")))}</code></td>'
            f'<td class="fp-num">{value:+.3f}</td><td>{tag}</td></tr>'
        )
    _md(
        '<table class="fp-table"><thead><tr>'
        '<th>Anonymized SECOM sensor feature</th><th>SHAP value</th><th>Contribution</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )
    _md(
        '<div class="fp-note">SECOM feature names are anonymized in the public dataset. In a real fab deployment, these IDs would map to tool, chamber, recipe, and sensor metadata.</div>'
    )


def safety_note() -> None:
    _md(
        """
        <div class="fp-note">
        <b>Safety note.</b> SHAP values explain model behavior for a single sample; they do not prove physical
        causality. SECOM features are anonymized sensor signals and WM-811K is a public benchmark dataset: this
        is an MVP decision-support demo, not a confirmed root-cause or production inspection tool.
        </div>
        """
    )


def _clean_axes(ax) -> None:
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors="#475569", labelsize=8, length=0)


def shap_figure(shap_features: list[dict[str, Any]]):
    """Direction-coloured horizontal SHAP contribution chart (presentation only)."""

    ordered = sorted(shap_features, key=lambda f: float(f.get("shap_value", 0.0)))
    names = [str(f.get("feature")) for f in ordered]
    values = [float(f.get("shap_value", 0.0)) for f in ordered]
    colors = [BAD if v > 0 else OK for v in values]

    fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=140)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.barh(names, values, color=colors, height=0.6)
    ax.axvline(0, color="#94A3B8", linewidth=1)
    ax.grid(axis="x", color="#EEF2F7", linewidth=1)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.set_xlabel("SHAP value   (right: toward failure · left: toward pass)", fontsize=8, color="#64748B")
    fig.tight_layout()
    return fig


def wafer_figure(image):
    """Styled wafer-map preview (presentation only)."""

    fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=140)
    fig.patch.set_alpha(0)
    ax.imshow(image, cmap="viridis", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")
    fig.tight_layout(pad=0.2)
    return fig
