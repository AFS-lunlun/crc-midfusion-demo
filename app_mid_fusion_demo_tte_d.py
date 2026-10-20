#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig7 Mid-fusion demo (_d).

Test-set browser (n=42) plus three Grad-CAM cases with high-resolution panels.
Usage:
  streamlit run app_mid_fusion_demo_tte_d.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "demo_assets_d"
VOLUMES = ASSETS / "volumes"
DISPLAY_PX = 512
FONT = "Arial"

COLORS = {
    "Clinical-only": "#000000",
    "Dose-only": "#999999",
    "Early": "#0072B2",
    "Mid": "#009E73",
    "Late": "#D55E00",
}
CAM_KEYS = {"Early": "cam_early", "Mid": "cam_mid", "Late": "cam_late"}
GENDER_EN = {"女": "Female", "男": "Male"}

GRADCAM_CASES = [
    {"key": "case1", "title": "Case 1", "tag": "Long-censored", "case_id": "040807-3014"},
    {"key": "case2", "title": "Case 2", "tag": "Death", "case_id": "049259-1986"},
    {"key": "case3", "title": "Case 3", "tag": "Early-censored", "case_id": "061675-1986"},
]
GRADCAM_IDS = {c["case_id"] for c in GRADCAM_CASES}


@st.cache_data(show_spinner=False)
def load_assets():
    import json

    meta = json.loads((ASSETS / "demo_meta_tte_d.json").read_text(encoding="utf-8"))
    roster = pd.read_csv(ASSETS / "demo_case_roster_tte_a.csv", dtype={"case_id": str})
    wide = pd.read_csv(ASSETS / "demo_case_probabilities_wide_tte_a.csv", dtype={"case_id": str})
    curves = pd.read_parquet(ASSETS / "demo_survival_curves_tte_a.parquet")
    volume_meta = pd.read_csv(ASSETS / "demo_volume_meta_tte_d.csv", dtype={"case_id": str})
    curves["case_id"] = curves["case_id"].astype(str)
    return meta, roster, wide, curves, volume_meta


@st.cache_data(show_spinner=False)
def load_volume(case_id: str):
    packed = np.load(VOLUMES / case_id / "dose_cam.npz")
    dose = np.asarray(packed["dose"], float)
    cams = {
        label: np.asarray(packed[key], float)
        for label, key in CAM_KEYS.items()
        if key in packed.files
    }
    if "Mid" not in cams and "cam" in packed.files:
        cams["Mid"] = np.asarray(packed["cam"], float)
    return dose, cams


def risk_label(p: float, pt: float) -> str:
    return "High" if p >= pt else "Low"


def live_reclass(p_clin: float, p_mid: float, pt: float) -> str:
    clin_high = p_clin >= pt
    mid_high = p_mid >= pt
    if not clin_high and mid_high:
        return "upward"
    if clin_high and not mid_high:
        return "downward"
    return "concordant_high" if clin_high else "concordant_low"


def display_gender(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return GENDER_EN.get(str(value), str(value))


def fmt_tnm(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    text = str(value).strip()
    return text if text else "—"


def adaptive01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo + 1e-8:
        lo, hi = float(arr.min()), float(arr.max())
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def upsample_gray(arr: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    img = Image.fromarray(np.clip(arr, 0, 1).astype(np.float32), mode="F")
    return np.asarray(img.resize((size, size), resample=Image.BILINEAR), dtype=np.float32)


def upsample_rgb(arr: np.ndarray, size: int = DISPLAY_PX) -> np.ndarray:
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(u8, mode="RGB")
    return np.asarray(img.resize((size, size), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def overlay_slice(dose: np.ndarray, cam: np.ndarray, z: int, alpha: float = 0.60) -> np.ndarray:
    z = int(np.clip(z, 0, dose.shape[0] - 1))
    d_norm = adaptive01(dose[z])
    t = np.clip(cam[z], 0, 1)
    heat = np.zeros(d_norm.shape + (3,), dtype=np.float32)
    heat[..., 0] = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    heat[..., 1] = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    heat[..., 2] = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    gray = np.stack([d_norm, d_norm, d_norm], axis=-1)
    out = (1.0 - alpha * t[..., None]) * gray + (alpha * t[..., None]) * heat
    return upsample_rgb(np.clip(out, 0, 1))


def apply_font(fig: go.Figure) -> go.Figure:
    fig.update_layout(font=dict(family=FONT, size=12, color="#243746"))
    return fig


def survival_figure(curves: pd.DataFrame, case_id: str) -> go.Figure:
    fig = go.Figure()
    sub = curves.loc[curves["case_id"].eq(case_id)]
    for model in ("Clinical-only", "Dose-only", "Early", "Mid", "Late"):
        part = sub.loc[sub["model"].eq(model)].sort_values("time_months")
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["time_months"],
                y=part["S"],
                mode="lines",
                name=model,
                line=dict(color=COLORS.get(model, "#333"), width=2.4 if model == "Mid" else 1.5),
            )
        )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=10, t=10, b=40),
        xaxis_title="Months",
        yaxis_title="S(t)",
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return apply_font(fig)


def dumbbell_figure(p_clin: float, p_mid: float, pt: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vline(x=pt, line_dash="dash", line_color="#94A3B8")
    fig.add_trace(
        go.Scatter(
            x=[p_clin, p_mid],
            y=["", ""],
            mode="lines",
            line=dict(color="#94A3B8", width=2),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_clin],
            y=[""],
            mode="markers",
            name="Clinical-only",
            marker=dict(size=12, color="white", line=dict(color="#000", width=2)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p_mid],
            y=[""],
            mode="markers",
            name="Mid",
            marker=dict(size=12, color=COLORS["Mid"]),
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(range=[-0.02, 1.02], title="p(24m)"),
        yaxis=dict(showticklabels=False),
        template="plotly_white",
        legend=dict(orientation="h", y=1.25),
    )
    return apply_font(fig)


def panel_path(case_id: str) -> Path:
    for name in ("panel.png", "manuscript_panel.png"):
        path = VOLUMES / case_id / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing panel for {case_id}")


def main() -> None:
    st.set_page_config(
        page_title="Mid-fusion Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stSelectbox,
        .stSlider, .stButton, .stDataFrame, .stMetric, div, p, span, label {{
            font-family: {FONT}, "Helvetica Neue", Helvetica, sans-serif !important;
        }}
        .case-card {{
            border: 1px solid #DCE4EA;
            border-radius: 10px;
            padding: 0.8rem 0.9rem;
            min-height: 120px;
            background: #F8FAFC;
            font-family: {FONT}, sans-serif;
        }}
        .case-card.active {{
            border: 1.5px solid #009E73;
            background: #F1FAF6;
        }}
        .case-title {{ font-size: 1.05rem; font-weight: 700; color: #243746; }}
        .case-tag {{ font-size: 0.92rem; color: #334155; margin: 0.15rem 0 0.45rem 0; }}
        .case-meta {{ font-size: 0.84rem; color: #64748B; line-height: 1.4; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    meta, roster, wide, curves, volume_meta = load_assets()
    default_pt = float(meta.get("primary_threshold", 0.20))
    if "selected_case" not in st.session_state:
        st.session_state.selected_case = GRADCAM_CASES[0]["case_id"]

    st.title("Mid-fusion Demo")

    pt = default_pt
    with st.sidebar:
        event_opt = st.selectbox("Outcome", ["All", "Events", "Censored"])
        reclass_opt = st.selectbox(
            "Reclass",
            ["All", "upward", "downward", "concordant_low", "concordant_high"],
        )
        only_gradcam = st.checkbox("Case 1-3", value=False)

    view = roster.copy()
    view["live_reclass"] = [
        live_reclass(float(a), float(b), pt)
        for a, b in zip(view["p24_Clinical-only"], view["p24_Mid"])
    ]
    view["has_gradcam"] = view["case_id"].isin(GRADCAM_IDS).astype(int)
    if event_opt == "Events":
        view = view.loc[view["OS_event"] == 1]
    elif event_opt == "Censored":
        view = view.loc[view["OS_event"] == 0]
    if reclass_opt != "All":
        view = view.loc[view["live_reclass"] == reclass_opt]
    if only_gradcam:
        view = view.loc[view["has_gradcam"] == 1]

    table = view[
        [
            "case_id",
            "OS_event",
            "OS_time_days",
            "age",
            "TNM",
            "p24_Clinical-only",
            "p24_Mid",
            "live_reclass",
            "has_gradcam",
        ]
    ].rename(
        columns={
            "case_id": "Case ID",
            "OS_event": "Event",
            "OS_time_days": "Days",
            "age": "Age",
            "p24_Clinical-only": "Clin. prob",
            "p24_Mid": "Mid prob",
            "live_reclass": "Reclass",
            "has_gradcam": "CAM",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True, height=240)

    case_ids = view["case_id"].tolist()
    if not case_ids:
        st.warning("No matching cases.")
        return
    if st.session_state.selected_case not in case_ids:
        st.session_state.selected_case = case_ids[0]

    card_cols = st.columns(3)
    for col, case in zip(card_cols, GRADCAM_CASES):
        row = roster.loc[roster["case_id"].eq(case["case_id"])].iloc[0]
        active = st.session_state.selected_case == case["case_id"]
        with col:
            st.markdown(
                f"""
                <div class="{'case-card active' if active else 'case-card'}">
                  <div class="case-title">{case['title']}</div>
                  <div class="case-tag">{case['tag']}</div>
                  <div class="case-meta">
                    {case['case_id']}<br/>
                    Age {int(row['age'])} · TNM {fmt_tnm(row['TNM'])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open",
                key=f"open_{case['key']}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.selected_case = case["case_id"]
                st.rerun()

    case_id = st.selectbox(
        "Case",
        case_ids,
        index=case_ids.index(st.session_state.selected_case),
    )
    st.session_state.selected_case = case_id

    row = roster.loc[roster["case_id"].eq(case_id)].iloc[0]
    w = wide.loc[wide["case_id"].eq(case_id)].iloc[0]
    p_clin = float(w["p24_Clinical-only"])
    p_mid = float(w["p24_Mid"])
    reclass = live_reclass(p_clin, p_mid, pt)
    gradcam_case = next((c for c in GRADCAM_CASES if c["case_id"] == case_id), None)
    title = f"{gradcam_case['title']} · {gradcam_case['tag']}" if gradcam_case else case_id
    st.subheader(title)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clinical p(24m)", f"{p_clin:.3f}", risk_label(p_clin, pt))
    m2.metric("Mid p(24m)", f"{p_mid:.3f}", risk_label(p_mid, pt))
    m3.metric("Δp", f"{p_mid - p_clin:+.3f}")
    m4.metric("Reclass", reclass)

    left, right = st.columns([1.35, 1.0])
    with left:
        st.plotly_chart(survival_figure(curves, case_id), use_container_width=True)
    with right:
        st.plotly_chart(dumbbell_figure(p_clin, p_mid, pt), use_container_width=True)
        st.markdown(
            f"""
            | | |
            |---|---|
            | Case ID | {case_id} |
            | Age / Sex / TNM | {int(row['age'])} / {display_gender(row.get('gender'))} / {fmt_tnm(row.get('TNM'))} |
            | Follow-up | {int(row['OS_time_days'])} d · {'Event' if int(row['OS_event']) == 1 else 'Censored'} |
            """
        )

    if gradcam_case is None:
        return

    vmeta = volume_meta.loc[volume_meta["case_id"].eq(case_id)].iloc[0]
    st.image(str(panel_path(case_id)), use_container_width=True)

    dose, cams = load_volume(case_id)
    default_z = int(vmeta["mid_cam_peak_z"])
    c1, c2, c3 = st.columns(3)
    with c1:
        z = st.slider("z", 0, int(dose.shape[0] - 1), default_z)
    with c2:
        cam_choice = st.selectbox("CAM", ["Mid", "Early", "Late"], index=0)
    with c3:
        alpha = st.slider("Opacity", 0.0, 0.9, 0.60, 0.05)

    a, b = st.columns(2)
    with a:
        st.image(
            upsample_gray(adaptive01(dose[z])),
            clamp=True,
            use_container_width=True,
            caption=f"Dose · z={z}",
        )
    with b:
        st.image(
            overlay_slice(dose, cams.get(cam_choice, cams["Mid"]), z, alpha=alpha),
            clamp=True,
            use_container_width=True,
            caption=f"{cam_choice} · z={z}",
        )


if __name__ == "__main__":
    main()
