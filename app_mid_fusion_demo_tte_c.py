#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig7 Mid-fusion interactive demo (browse-only, Grad-CAM hi-res _c).

Based on app_mid_fusion_demo_tte_b.py.
- Tables for all 42 test cases
- Dose/CAM volumes for the 3 manuscript Grad-CAM cases only
- Manuscript panels are the exact figs_summary PNG exports
- Interactive slices are bilinear-upsampled to 512 px
Usage:
  streamlit run app_mid_fusion_demo_tte_c.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "demo_assets_c"
VOLUMES = ASSETS / "volumes"
PANELS = ASSETS / "panels"
DISPLAY_PX = 512

COLORS = {
    "Clinical-only": "#000000",
    "Dose-only": "#999999",
    "Early": "#0072B2",
    "Mid": "#009E73",
    "Late": "#D55E00",
}

GENDER_EN = {"女": "Female", "男": "Male"}
CAM_KEYS = {
    "Early": "cam_early",
    "Mid": "cam_mid",
    "Late": "cam_late",
}


@st.cache_data(show_spinner=False)
def load_assets():
    import json

    meta = json.loads((ASSETS / "demo_meta_tte_c.json").read_text(encoding="utf-8"))
    roster = pd.read_csv(ASSETS / "demo_case_roster_tte_a.csv")
    wide = pd.read_csv(ASSETS / "demo_case_probabilities_wide_tte_a.csv")
    labels = pd.read_csv(ASSETS / "demo_reclass_labels_tte_a.csv")
    curves = pd.read_parquet(ASSETS / "demo_survival_curves_tte_a.parquet")
    volume_meta = pd.read_csv(ASSETS / "demo_volume_meta_tte_c.csv")
    blurbs = json.loads((ASSETS / "demo_case_blurbs_tte_c.json").read_text(encoding="utf-8"))
    for frame in (roster, wide, labels, curves, volume_meta):
        frame["case_id"] = frame["case_id"].astype(str)
    return meta, roster, wide, labels, curves, volume_meta, blurbs


@st.cache_data(show_spinner=False)
def load_volume(case_id: str):
    packed = np.load(VOLUMES / case_id / "dose_cam.npz")
    dose = np.asarray(packed["dose"], float)
    cams = {}
    for label, key in CAM_KEYS.items():
        if key in packed.files:
            cams[label] = np.asarray(packed[key], float)
        elif label == "Mid" and "cam" in packed.files:
            cams[label] = np.asarray(packed["cam"], float)
    if "Mid" not in cams and "cam" in packed.files:
        cams["Mid"] = np.asarray(packed["cam"], float)
    return dose, cams


def risk_label(p: float, pt: float) -> str:
    return "High risk" if p >= pt else "Low risk"


def display_gender(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    s = str(value)
    return GENDER_EN.get(s, s)


def upsample_gray(arr: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    img = Image.fromarray(np.clip(arr, 0, 1).astype(np.float32), mode="F")
    out = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32)


def upsample_rgb(arr: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")
    out = img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0


def normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def adaptive01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo + 1e-8:
        lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def overlay_slice(dose: np.ndarray, cam: np.ndarray, z: int, alpha: float = 0.60) -> np.ndarray:
    z = int(np.clip(z, 0, dose.shape[0] - 1))
    d_norm = adaptive01(dose[z])
    c_norm = np.clip(cam[z], 0, 1)
    # jet-like RGB for CAM to align with manuscript panels
    heat = np.zeros(d_norm.shape + (3,), dtype=np.float32)
    # piecewise jet approximation
    t = c_norm
    heat[..., 0] = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    heat[..., 1] = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    heat[..., 2] = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    gray = np.stack([d_norm, d_norm, d_norm], axis=-1)
    mask = c_norm[..., None]
    out = (1.0 - alpha * mask) * gray + (alpha * mask) * heat
    return upsample_rgb(np.clip(out, 0, 1))


def survival_figure(curves: pd.DataFrame, case_id: str, models: list[str]) -> go.Figure:
    fig = go.Figure()
    sub = curves.loc[curves["case_id"].eq(case_id)]
    for m in models:
        part = sub.loc[sub["model"].eq(m)].sort_values("time_months")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["time_months"],
                y=part["S"],
                mode="lines",
                name=m,
                line=dict(color=COLORS.get(m, "#333"), width=2.2 if m == "Mid" else 1.5),
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis_title="Follow-up time (months)",
        yaxis_title="Predicted survival S(t)",
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return fig


def dumbbell_figure(p_clin: float, p_mid: float, pt: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vline(x=pt, line_dash="dash", line_color="#94A3B8", annotation_text=f"pt={pt:.2f}")
    fig.add_trace(
        go.Scatter(
            x=[p_clin, p_mid],
            y=["Clinical → Mid", "Clinical → Mid"],
            mode="lines",
            line=dict(color="#94A3B8", width=2),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_clin],
            y=["Clinical → Mid"],
            mode="markers",
            name="Clinical-only",
            marker=dict(size=12, color="white", line=dict(color="#000", width=2)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_mid],
            y=["Clinical → Mid"],
            mode="markers",
            name="Mid",
            marker=dict(size=12, color=COLORS["Mid"]),
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(range=[-0.02, 1.02], title="24-month event probability p(24m)"),
        yaxis=dict(showticklabels=False),
        template="plotly_white",
        legend=dict(orientation="h", y=1.2),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Mid-fusion Grad-CAM Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    meta, roster, wide, labels, curves, volume_meta, blurbs = load_assets()

    st.title("Mid-fusion Case-level Demo")
    st.caption(meta.get("disclaimer", ""))
    st.markdown(
        f"Frozen test set **n={meta['n_test']}** ({meta['n_events']} events). "
        f"Mid λ={meta['lam']}, c={meta['scale']}; "
        f"primary decision time point {meta['primary_month']} months. "
        "Grad-CAM volumes are limited to the three manuscript cases "
        "(Figure 5C, Figure S3C1, Figure S3C2)."
    )

    with st.sidebar:
        st.header("Filters")
        pt = st.slider(
            "Decision threshold pt",
            min_value=0.05,
            max_value=0.50,
            value=float(meta.get("primary_threshold", 0.20)),
            step=0.01,
        )
        event_opt = st.selectbox("Outcome", ["All", "Events only", "Censored only"])
        reclass_opt = st.selectbox(
            "Reclassification vs Clinical-only",
            ["All", "upward", "downward", "concordant_low", "concordant_high"],
        )
        showcase_only = st.checkbox("Grad-CAM manuscript cases only", value=True)
        models_curve = st.multiselect(
            "Survival curve models",
            ["Clinical-only", "Dose-only", "Early", "Mid", "Late"],
            default=["Clinical-only", "Mid"],
        )

    view = roster.copy()
    view["live_clin_high"] = (view["p24_Clinical-only"] >= pt).astype(int)
    view["live_mid_high"] = (view["p24_Mid"] >= pt).astype(int)

    def live_label(row):
        if row["live_clin_high"] == 0 and row["live_mid_high"] == 1:
            return "upward"
        if row["live_clin_high"] == 1 and row["live_mid_high"] == 0:
            return "downward"
        if row["live_clin_high"] == 0:
            return "concordant_low"
        return "concordant_high"

    view["live_reclass"] = view.apply(live_label, axis=1)

    if event_opt == "Events only":
        view = view.loc[view["OS_event"] == 1]
    elif event_opt == "Censored only":
        view = view.loc[view["OS_event"] == 0]
    if reclass_opt != "All":
        view = view.loc[view["live_reclass"] == reclass_opt]
    if showcase_only:
        view = view.loc[view["is_showcase"] == 1]

    st.subheader("Test-set case list")
    show_cols = [
        "case_id",
        "OS_event",
        "OS_time_days",
        "age",
        "TNM",
        "dose_group",
        "p24_Clinical-only",
        "p24_Mid",
        "live_reclass",
        "is_showcase",
        "showcase_roles",
    ]
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(
        view[show_cols].rename(
            columns={
                "p24_Clinical-only": "p24_Clinical",
                "live_reclass": "reclass@pt",
                "is_showcase": "gradcam_case",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    case_ids = view["case_id"].tolist()
    if not case_ids:
        st.warning("No cases match the current filters.")
        return

    default_idx = 0
    preferred = ["040807-3014", "049259-1986", "061675-1986"]
    for pref in preferred:
        if pref in case_ids:
            default_idx = case_ids.index(pref)
            break

    case_id = st.selectbox("Select case", case_ids, index=default_idx)
    row = view.loc[view["case_id"].eq(case_id)].iloc[0]
    w = wide.loc[wide["case_id"].eq(case_id)].iloc[0]

    st.subheader(f"Case detail · {case_id}")
    c1, c2, c3, c4 = st.columns(4)
    p_clin = float(w["p24_Clinical-only"])
    p_mid = float(w["p24_Mid"])
    c1.metric("Clinical p(24m)", f"{p_clin:.3f}", risk_label(p_clin, pt))
    c2.metric("Mid p(24m)", f"{p_mid:.3f}", risk_label(p_mid, pt))
    c3.metric("Δp (Mid−Clinical)", f"{p_mid - p_clin:+.3f}")
    c4.metric(
        "Follow-up",
        f"{int(row['OS_time_days'])} days",
        "Event" if int(row["OS_event"]) == 1 else "Censored",
    )

    info_cols = st.columns(3)
    info_cols[0].write(
        {
            "Age": row.get("age"),
            "Sex": display_gender(row.get("gender")),
            "TNM": row.get("TNM"),
            "KPS": row.get("karnofsky"),
        }
    )
    info_cols[1].write(
        {
            "Dose group": row.get("dose_group"),
            "Total dose (Gy)": row.get("total_dose_gy"),
            "Technique": row.get("radiotherapy_technique"),
        }
    )
    info_cols[2].write(
        {
            "Concurrent chemo intensity": row.get("concurrent_chemo_intensity"),
            "Pre-RT chemo": row.get("preRT_chemo_any"),
            "Showcase role": row.get("showcase_roles") or "—",
        }
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown("**Predicted survival curves**")
        st.plotly_chart(survival_figure(curves, case_id, models_curve or ["Mid"]), use_container_width=True)
    with right:
        st.markdown("**24-month probability dumbbell (Clinical → Mid)**")
        st.plotly_chart(dumbbell_figure(p_clin, p_mid, pt), use_container_width=True)
        st.markdown(
            f"Stratification at current threshold: **Clinical {risk_label(p_clin, pt)}** → "
            f"**Mid {risk_label(p_mid, pt)}** ({row['live_reclass']})"
        )

    st.subheader("Manuscript Grad-CAM panel and interactive slices")
    vol_ids = set(volume_meta["case_id"])
    has_vol = case_id in vol_ids and (VOLUMES / case_id / "dose_cam.npz").is_file()
    if not has_vol:
        st.info(
            "Dose/CAM volumes are packaged for the three manuscript Grad-CAM cases only "
            "(Figure 5C, Figure S3C1, Figure S3C2). Enable “Grad-CAM manuscript cases only”, "
            "or select one of those cases."
        )
        return

    meta_row = volume_meta.loc[volume_meta["case_id"].eq(case_id)].iloc[0]
    blurb = blurbs.get(case_id) or str(meta_row.get("blurb") or "")
    if blurb:
        st.write(blurb)

    figure_name = str(meta_row.get("manuscript_figure") or "")
    panel_path = VOLUMES / case_id / "manuscript_panel.png"
    if not panel_path.is_file() and figure_name:
        panel_path = PANELS / f"{figure_name}.png"
    if panel_path.is_file():
        st.markdown(f"**Manuscript panel · {figure_name}**")
        st.image(str(panel_path), use_container_width=True)
        st.caption(
            "Exact high-resolution panel exported for the manuscript "
            f"({figure_name}; same pixels as figs_summary)."
        )

    dose, cams = load_volume(case_id)
    default_z = int(meta_row["mid_cam_peak_z"])
    z = st.slider("Axial slice z", 0, int(dose.shape[0] - 1), default_z)
    alpha = st.slider("CAM overlay opacity", 0.0, 0.9, 0.60, 0.05)
    cam_choice = st.selectbox("Grad-CAM model", ["Mid", "Early", "Late"], index=0)
    cam = cams.get(cam_choice, cams["Mid"])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Dose (grayscale, {DISPLAY_PX} px) · z={z}**")
        st.image(upsample_gray(adaptive01(dose[z])), clamp=True, use_container_width=True)
    with col_b:
        st.markdown(f"**Dose + {cam_choice} Grad-CAM ({DISPLAY_PX} px) · z={z}**")
        st.image(overlay_slice(dose, cam, z, alpha=alpha), clamp=True, use_container_width=True)
    st.caption(
        f"Default anchor slice mid_cam_peak_z={default_z}; dose_peak_z={meta_row.get('dose_peak_z')}. "
        f"Native volume grid is 64×64×64 (float32); interactive views are bilinear-upsampled to "
        f"{DISPLAY_PX}×{DISPLAY_PX} to match manuscript panel sharpness. "
        "Grad-CAM is precomputed (not online inference)."
    )


if __name__ == "__main__":
    main()
