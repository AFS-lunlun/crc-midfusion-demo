#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig7 Mid-fusion 交互演示（浏览型预计算 · 瘦身 _b）。

基于 app_mid_fusion_demo_tte_a.py：全 42 例表格式浏览；剂量/CAM 仅精选 7 例。
用法：
  streamlit run app_mid_fusion_demo_tte_b.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ASSETS = APP_DIR / "demo_assets_b"
VOLUMES = ASSETS / "volumes"

COLORS = {
    "Clinical-only": "#000000",
    "Dose-only": "#999999",
    "Early": "#0072B2",
    "Mid": "#009E73",
    "Late": "#D55E00",
}


@st.cache_data(show_spinner=False)
def load_assets():
    import json

    meta_path = ASSETS / "demo_meta_tte_b.json"
    if not meta_path.is_file():
        meta_path = ASSETS / "demo_meta_tte_a.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    roster = pd.read_csv(ASSETS / "demo_case_roster_tte_a.csv")
    wide = pd.read_csv(ASSETS / "demo_case_probabilities_wide_tte_a.csv")
    labels = pd.read_csv(ASSETS / "demo_reclass_labels_tte_a.csv")
    curves = pd.read_parquet(ASSETS / "demo_survival_curves_tte_a.parquet")
    volume_meta_path = ASSETS / "demo_volume_meta_tte_b.csv"
    if not volume_meta_path.is_file():
        volume_meta_path = ASSETS / "demo_volume_meta_tte_a.csv"
    if volume_meta_path.is_file():
        volume_meta = pd.read_csv(volume_meta_path)
    else:
        volume_meta = pd.read_csv(ASSETS / "demo_showcase_meta_tte_a.csv")
    blurbs = json.loads((ASSETS / "demo_case_blurbs_tte_a.json").read_text(encoding="utf-8"))
    roster["case_id"] = roster["case_id"].astype(str)
    wide["case_id"] = wide["case_id"].astype(str)
    labels["case_id"] = labels["case_id"].astype(str)
    curves["case_id"] = curves["case_id"].astype(str)
    volume_meta["case_id"] = volume_meta["case_id"].astype(str)
    return meta, roster, wide, labels, curves, volume_meta, blurbs


@st.cache_data(show_spinner=False)
def load_volume(case_id: str):
    root = VOLUMES / case_id
    npz_path = root / "dose_cam.npz"
    if npz_path.is_file():
        packed = np.load(npz_path)
        dose = np.asarray(packed["dose"], float)
        cam = np.asarray(packed["cam"], float)
        return dose, cam
    dose = np.load(root / "dose_patch.npy")
    cam = np.load(root / "gradcam_mid.npy")
    return np.asarray(dose, float), np.asarray(cam, float)


def risk_label(p: float, pt: float) -> str:
    return "高危" if p >= pt else "低危"


def overlay_slice(dose: np.ndarray, cam: np.ndarray, z: int, alpha: float = 0.45) -> np.ndarray:
    z = int(np.clip(z, 0, dose.shape[0] - 1))
    d = dose[z]
    c = cam[z]
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8)
    rgb = np.stack([d_norm, d_norm, d_norm], axis=-1)
    heat = np.zeros_like(rgb)
    heat[..., 0] = c_norm
    heat[..., 1] = 0.2 * c_norm
    mask = c_norm[..., None]
    out = (1.0 - alpha * mask) * rgb + (alpha * mask) * heat
    return np.clip(out, 0, 1)


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
        xaxis_title="随访时间（月）",
        yaxis_title="预测生存概率 S(t)",
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
    color = COLORS["Mid"]
    fig.add_trace(
        go.Scatter(
            x=[p_mid],
            y=["Clinical → Mid"],
            mode="markers",
            name="Mid",
            marker=dict(size=12, color=color),
        )
    )
    fig.update_layout(
        height=160,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(range=[-0.02, 1.02], title="24 个月事件概率 p(24m)"),
        yaxis=dict(showticklabels=False),
        template="plotly_white",
        legend=dict(orientation="h", y=1.2),
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Fig7 Mid-fusion Demo",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    meta, roster, wide, labels, curves, volume_meta, blurbs = load_assets()

    st.title("Figure 7 · Mid-fusion 病例级演示（瘦身版）")
    st.caption(meta.get("disclaimer", ""))
    if meta.get("slim_note"):
        st.caption(meta["slim_note"])
    st.markdown(
        f"冻结测试集 **n={meta['n_test']}**（事件 {meta['n_events']}）。"
        f"展示口径与 Figure 4D/6 一致：Mid λ={meta['lam']}, c={meta['scale']}；"
        f"主决策时点 {meta['primary_month']} 月。"
    )

    with st.sidebar:
        st.header("筛选")
        pt = st.slider(
            "决策阈值 pt",
            min_value=0.05,
            max_value=0.50,
            value=float(meta.get("primary_threshold", 0.20)),
            step=0.01,
        )
        event_opt = st.selectbox("结局", ["全部", "仅事件", "仅删失"])
        reclass_opt = st.selectbox(
            "相对 Clinical-only 重分类",
            ["全部", "upward", "downward", "concordant_low", "concordant_high"],
        )
        showcase_only = st.checkbox("仅精选加深例", value=True)
        models_curve = st.multiselect(
            "生存曲线模型",
            ["Clinical-only", "Dose-only", "Early", "Mid", "Late"],
            default=["Clinical-only", "Mid"],
        )

    view = roster.copy()
    # live reclass vs clinical at chosen pt
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

    if event_opt == "仅事件":
        view = view.loc[view["OS_event"] == 1]
    elif event_opt == "仅删失":
        view = view.loc[view["OS_event"] == 0]
    if reclass_opt != "全部":
        view = view.loc[view["live_reclass"] == reclass_opt]
    if showcase_only:
        view = view.loc[view["is_showcase"] == 1]

    st.subheader("测试集病例列表")
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
                "is_showcase": "showcase",
            }
        ),
        use_container_width=True,
        hide_index=True,
        height=280,
    )

    case_ids = view["case_id"].tolist()
    if not case_ids:
        st.warning("当前筛选无病例。")
        return

    default_idx = 0
    preferred = ["035873-1986", "040807-3014", "042165-3014"]
    for pref in preferred:
        if pref in case_ids:
            default_idx = case_ids.index(pref)
            break

    case_id = st.selectbox("选择病例", case_ids, index=default_idx)
    row = view.loc[view["case_id"].eq(case_id)].iloc[0]
    w = wide.loc[wide["case_id"].eq(case_id)].iloc[0]

    st.subheader(f"病例详情 · {case_id}")
    c1, c2, c3, c4 = st.columns(4)
    p_clin = float(w["p24_Clinical-only"])
    p_mid = float(w["p24_Mid"])
    c1.metric("Clinical p(24m)", f"{p_clin:.3f}", risk_label(p_clin, pt))
    c2.metric("Mid p(24m)", f"{p_mid:.3f}", risk_label(p_mid, pt))
    c3.metric("Δp (Mid−Clinical)", f"{p_mid - p_clin:+.3f}")
    c4.metric(
        "随访",
        f"{int(row['OS_time_days'])} 天",
        "事件" if int(row["OS_event"]) == 1 else "删失",
    )

    info_cols = st.columns(3)
    info_cols[0].write(
        {
            "年龄": row.get("age"),
            "性别": row.get("gender"),
            "TNM": row.get("TNM"),
            "KPS": row.get("karnofsky"),
        }
    )
    info_cols[1].write(
        {
            "剂量分组": row.get("dose_group"),
            "总剂量 Gy": row.get("total_dose_gy"),
            "技术": row.get("radiotherapy_technique"),
        }
    )
    info_cols[2].write(
        {
            "同步化疗强度": row.get("concurrent_chemo_intensity"),
            "术前化疗": row.get("preRT_chemo_any"),
            "精选角色": row.get("showcase_roles") or "—",
        }
    )

    left, right = st.columns([1.2, 1.0])
    with left:
        st.markdown("**预测生存曲线**")
        st.plotly_chart(survival_figure(curves, case_id, models_curve or ["Mid"]), use_container_width=True)
    with right:
        st.markdown("**24 个月概率 dumbbell（Clinical → Mid）**")
        st.plotly_chart(dumbbell_figure(p_clin, p_mid, pt), use_container_width=True)
        st.markdown(
            f"当前阈值下分层：**Clinical {risk_label(p_clin, pt)}** → "
            f"**Mid {risk_label(p_mid, pt)}**（{row['live_reclass']}）"
        )

    # Dose + Mid Grad-CAM for all packaged cases
    st.subheader("剂量切片与 Mid Grad-CAM")
    vol_ids = set(volume_meta["case_id"])
    has_vol = case_id in vol_ids and (
        (VOLUMES / case_id / "dose_cam.npz").is_file()
        or (VOLUMES / case_id / "dose_patch.npy").is_file()
    )
    if not has_vol:
        st.info("瘦身版仅打包精选 7 例的剂量/CAM。请勾选「仅精选加深例」，或改选精选病例。")
        return

    meta_row = volume_meta.loc[volume_meta["case_id"].eq(case_id)].iloc[0]
    blurb = blurbs.get(case_id) or str(meta_row.get("blurb") or "")
    if blurb:
        st.write(blurb)
    elif int(meta_row.get("is_showcase", 0) or 0) == 0:
        st.caption("普通测试例：可浏览剂量与 Mid Grad-CAM；精选例另有重分类/机制叙事文案。")

    dose, cam = load_volume(case_id)
    default_z = int(meta_row["mid_cam_peak_z"])
    z = st.slider("轴位层 z", 0, int(dose.shape[0] - 1), default_z)
    alpha = st.slider("CAM 叠加透明度", 0.0, 0.9, 0.45, 0.05)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**剂量灰度 · z={z}**")
        d = dose[z]
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        st.image(d_norm, clamp=True, use_container_width=True)
    with col_b:
        st.markdown(f"**剂量 + Mid Grad-CAM · z={z}**")
        st.image(overlay_slice(dose, cam, z, alpha=alpha), clamp=True, use_container_width=True)
    st.caption(
        f"默认锚定层 mid_cam_peak_z={default_z}；dose_peak_z={meta_row.get('dose_peak_z')}。"
        " Grad-CAM 为预计算精选例（瘦身包），非在线推理。"
    )


if __name__ == "__main__":
    main()
