import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="AgriDrone Digital Twin - Field Intelligence Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Minimal CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #111418; }
::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #4ADE80; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2530 !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: background 0.15s;
    font-size: 0.875rem !important;
    color: #94A3B8 !important;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1A2133 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #475569 !important; font-size: 0.78rem !important; }

/* Page title */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 0.25rem !important;
}
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #CBD5E1 !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: #94A3B8 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #131920 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #2D4A6A !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #4B6478 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    color: #4ADE80 !important;
}

/* Bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #131920 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0D1117 !important;
    border: 1px solid #1E2530 !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 6px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #4B6478 !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1A2B3C !important;
    color: #CBD5E1 !important;
    font-weight: 600 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
    font-size: 0.875rem !important;
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #1E2530 !important;
    border-radius: 8px !important;
    background: #0D1117 !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #94A3B8 !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    border: 1px solid #1E2530 !important;
    overflow: hidden !important;
}

/* Buttons */
[data-testid="stDownloadButton"] button {
    background: #1A2B3C !important;
    color: #CBD5E1 !important;
    border: 1px solid #2D4A6A !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #1E3448 !important;
    border-color: #4ADE80 !important;
    color: #4ADE80 !important;
}

/* Progress */
[data-testid="stProgressBar"] > div > div {
    background: #4ADE80 !important;
    border-radius: 4px !important;
}

/* Divider */
hr { border-color: #1E2530 !important; margin: 1.25rem 0 !important; }

/* Caption */
[data-testid="stCaptionContainer"] { color: #384B60 !important; font-size: 0.76rem !important; }

/* Select */
[data-baseweb="select"] > div { border-color: #1E2D3D !important; background: #131920 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


def _plotly_layout(fig, **extra):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1117",
        font=dict(family="Inter, sans-serif", color="#64748B", size=11),
        title_font=dict(family="Inter, sans-serif", color="#94A3B8", size=13),
        legend=dict(bgcolor="#131920", bordercolor="#1E2530", borderwidth=1, font=dict(color="#64748B")),
        margin=dict(l=8, r=8, t=44, b=8),
        xaxis=dict(gridcolor="#161E2A", zerolinecolor="#161E2A", tickfont=dict(color="#475569")),
        yaxis=dict(gridcolor="#161E2A", zerolinecolor="#161E2A", tickfont=dict(color="#475569")),
    )
    fig.update_layout(**{**base, **extra})
    return fig


def _badge(text, color="#4ADE80"):
    return (f"<span style='background:{color}18;color:{color};border:1px solid {color}35;"
            f"border-radius:5px;padding:2px 9px;font-size:0.76rem;font-weight:600;"
            f"font-family:JetBrains Mono,monospace;letter-spacing:0.02em'>{text}</span>")

CLASS_HEX = {
    "HOTSPOT_HIGH": "#DC1414",
    "HOTSPOT_MODERATE": "#FF7828",
    "COLDSPOT_HIGH": "#2878C8",
    "COLDSPOT_MODERATE": "#50BEE6",
    "NEUTRAL": "#8CAA8C",
}
SEV_HEX = {"HEALTHY": "#4ADE80", "MILD_STRESS": "#38BDF8", "MODERATE_STRESS": "#F59E0B", "SEVERE_STRESS": "#F87171"}
PATHO_HEX = {"Rust_Lesion": "#F87171", "LeafSpot_Brown": "#D97706", "Chlorosis_Yellow": "#FBBF24", "General_Lesion": "#A78BFA"}


@st.cache_data(ttl=300)
def load_json(rel):
    p = BASE / rel
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_csv(rel):
    p = BASE / rel
    if not p.exists():
        return None
    return pd.read_csv(p)


def img(rel):
    p = BASE / rel if rel else None
    return str(p) if p and p.exists() else None


artifact_index = load_json("artifacts/artifact_index.json") or {}
health_report = load_json("metadata/health_report.json")
pathogen_report = load_json("metadata/pathogen_report.json")
weather_report = load_json("metadata/weather_report.json")
epi_report = load_json("epidemiology/epidemiology_report.json")
scenario_report = load_json("metadata/scenario_analysis_report.json")
zone_manifest = load_json("metadata/zone_manifest.json")

with st.sidebar:
    st.markdown("""
    <div style='padding:20px 4px 12px 4px'>
        <div style='display:flex;align-items:center;gap:10px;margin-bottom:6px'>
            <span style='font-size:1.5rem'>🛰️</span>
            <span style='font-weight:700;font-size:0.95rem;color:#E2E8F0;letter-spacing:-0.3px'>AgriDrone Digital Twin</span>
        </div>
        <div style='font-size:0.72rem;color:#334155;line-height:1.5'>Drone orthomosaic → micro-zones → health → pathogens → weather → epidemiology → GenAI</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1E2530;margin:0 0 12px 0'></div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "1️⃣ Field Overview",
            "2️⃣ Weather @ Capture Time",
            "3️⃣ Zone Inspector",
            "4️⃣ Plant Health Analytics",
            "5️⃣ Pathogen Detections",
            "6️⃣ Hotspot / Coldspot Map",
            "7️⃣ FULL SCENARIO ANALYSIS",
        ],
    )
    if artifact_index:
        st.markdown("<div style='height:1px;background:#1E2530;margin:12px 0'></div>", unsafe_allow_html=True)
        st.metric("Pipeline version", artifact_index.get("pipeline_version", "?"))
        gps = artifact_index.get("gps_center") or {}
        st.caption(f"Field center: {gps.get('lat', '?'):.5f}N, {gps.get('lon', '?'):.5f}E" if gps else "")
        st.caption(f"{artifact_index.get('zone_count', 9)} micro-zones | Korba, Chhattisgarh")
        st.success("All modules complete ✔", icon="✅")
        if scenario_report:
            gs = scenario_report.get("genai_status", {})
            st.info(f"Scenario engine: **{gs.get('source')}**\n\nModel: `{gs.get('model')}`", icon="🤖")

_PAGE_TITLES = {
    "1️⃣ Field Overview": "Field Digital Twin",
    "2️⃣ Weather @ Capture Time": "Weather at Capture Time",
    "3️⃣ Zone Inspector": "Micro-Zone Inspector",
    "4️⃣ Plant Health Analytics": "Plant Health Analytics",
    "5️⃣ Pathogen Detections": "Pathogen Detection Results",
    "6️⃣ Hotspot / Coldspot Map": "Spatial Epidemiology",
    "7️⃣ FULL SCENARIO ANALYSIS": "Full Scenario Analysis & Advisory",
}
_PAGE_ICONS = {
    "1️⃣ Field Overview": "🗺️",
    "2️⃣ Weather @ Capture Time": "🌦️",
    "3️⃣ Zone Inspector": "🔍",
    "4️⃣ Plant Health Analytics": "🌿",
    "5️⃣ Pathogen Detections": "🦠",
    "6️⃣ Hotspot / Coldspot Map": "📍",
    "7️⃣ FULL SCENARIO ANALYSIS": "🧠",
}
st.markdown(
    f"<div style='margin-bottom:1.5rem'>"
    f"<div style='font-size:0.75rem;color:#334155;font-weight:500;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px'>{_PAGE_ICONS[page]} {page.split(' ', 1)[0].replace('️⃣','').strip()}</div>"
    f"<h1 style='margin:0;padding:0'>{_PAGE_TITLES[page]}</h1>"
    f"<div style='height:2px;background:#1E2530;margin-top:12px;border-radius:1px'></div>"
    f"</div>",
    unsafe_allow_html=True,
)

if page == "1️⃣ Field Overview":
    c1, c2, c3, c4 = st.columns(4)
    hfs = (health_report or {}).get("field_summary", {})
    pts = (pathogen_report or {}).get("field_summary", {})
    efs = (epi_report or {}).get("field_summary", {})
    c1.metric("Mean field health", f"{hfs.get('mean_field_score', 0):.2f} / 1.00", help="Composite of ExG+VARI+NDVIproxy+MGRVI vegetation indices")
    c2.metric("Pathogen detections", pts.get("total_detections", 0), help="Unsupervised color-anomaly lesions across 9 zones")
    c3.metric("Mean anomaly coverage", f"{pts.get('mean_zone_coverage_pct', 0):.2f}%")
    c4.metric("Max spread risk", f"{efs.get('max_spread_risk', 0):.2f}", help="Adjacency-weighted epidemic spread index")

    ov = artifact_index.get("orthomosaic_files", {})
    tabs = st.tabs(["Raw Orthomosaic", "3x3 Grid Twin", "Health Overlay", "Pathogen Overlay", "Epidemiology Overlay"])
    for t, key, cap in [
        (tabs[0], "raw_no_annotation", "Feature-matched stitched canvas (SCANS mode) - no annotations"),
        (tabs[1], "annotated_grid", "Digital twin divided into 9 GPS-tagged micro-zones"),
        (tabs[2], "health_overlay", "Per-zone vegetation health severity tint"),
        (tabs[3], "pathogen_overlay", "Infection-risk tint by detected pathogen classes"),
        (tabs[4], "epidemiology_overlay", "Getis-Ord Gi* hotspot / coldspot classification + spread risk"),
    ]:
        with t:
            p = img(ov.get(key, ""))
            if p:
                st.image(p, width='stretch')
                st.caption(cap)
            else:
                st.warning(f"Missing: {ov.get(key)}")

    st.divider()
    st.subheader("📦 Output catalog")
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        det_n = len(list((BASE / 'detections').rglob('*.jpg')))
        sp_n = len(list((BASE / 'epidemiology' / 'zone_spread_heatmaps').glob('*.jpg'))) if (BASE / 'epidemiology').exists() else 0
        _catalog_items = [
            ("Stitching stage images", len(artifact_index.get('stitching_stages', []))),
            ("Micro-zone crops", len(list((BASE / 'micro_zones').glob('*.jpg')))),
            ("Health heatmap renders", len(list((BASE / 'health_heatmaps').glob('*.jpg')))),
            ("Pathogen diagnostics", det_n),
            ("Spread-risk renders", sp_n),
        ]
        for _lbl, _val in _catalog_items:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"padding:9px 14px;margin-bottom:5px;background:#0D1117;"
                f"border-radius:7px;border:1px solid #1E2530'>"
                f"<span style='color:#475569;font-size:0.83rem'>{_lbl}</span>"
                f"<span style='color:#CBD5E1;font-weight:600;font-family:JetBrains Mono,monospace;font-size:0.83rem'>{_val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    with cc2:
        mf = artifact_index.get("metadata_files", {})
        st.dataframe(pd.DataFrame({"report": list(mf.keys()), "path": list(mf.values())}), width='stretch', hide_index=True)

elif page == "2️⃣ Weather @ Capture Time":
    if not weather_report:
        st.error("weather_report.json missing"); st.stop()
    ds = weather_report["daily_summary"]["capture_day"]
    prev7 = weather_report["daily_summary"]["previous_7_days"]
    agro = weather_report.get("inferred_agro_conditions", {})

    src = weather_report.get("capture_window", {}).get("date_source", "")
    st.info(f"📍 {weather_report['location']['region_guess']}  |  🗓️ Capture day **{ds.get('date')}**  |  Source: `{src}` · Open-Meteo archive API", icon="📡")

    m = st.columns(6)
    m[0].metric("Temp min", f"{ds.get('temp_min_c')} °C")
    m[1].metric("Temp max", f"{ds.get('temp_max_c')} °C")
    m[2].metric("Humidity mean", f"{ds.get('humidity_mean_pct')} %")
    m[3].metric("Humidity max", f"{ds.get('humidity_max_pct')} %")
    m[4].metric("Rain (day)", f"{ds.get('precipitation_sum_mm')} mm")
    m[5].metric("Wind max", f"{ds.get('wind_max_kmh')} km/h")

    st.subheader("⚠️ Inferred agro-meteorological risk flags")
    f1, f2, f3, f4 = st.columns(4)
    def flag(col, label, val, why):
        _color = "#4ADE80" if val else "#F87171"
        _bg    = "#0D1A12" if val else "#1A0D0D"
        _status = "Active" if val else "Inactive"
        col.markdown(
            f"<div style='background:{_bg};border:1px solid #1E2530;"
            f"border-left:3px solid {_color};"
            f"border-radius:8px;padding:14px 16px;height:100%'>"
            f"<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#334155;margin-bottom:6px'>{label}</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{_color};margin-bottom:8px'>{_status}</div>"
            f"<div style='color:#334155;font-size:0.74rem;line-height:1.5'>{why}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    flag(f1, "High leaf-wetness risk", agro.get("high_leaf_wetness_risk"), "RH ≥85% or 7-day rain ≥25 mm → long nightly infection windows")
    flag(f2, "Favorable rust conditions", agro.get("favorable_rust_conditions"), "Wetness + temp 18–28 °C range")
    flag(f3, "Favorable leaf-spot conditions", agro.get("favorable_leafspot_conditions"), "Wetness + temp 25–35 °C range (Bipolaris/Cercospora optimum)")
    flag(f4, "Favorable chlorosis conditions", agro.get("favorable_chlorosis_conditions"), "7-day rain >40 mm + RH >75% → nutrient leaching/hypoxia yellows")

    st.subheader("Hourly conditions on capture day")
    hourly = load_csv("metadata/weather_hourly.csv")
    if hourly is not None and "time" in hourly.columns:
        hourly["time"] = pd.to_datetime(hourly["time"])
        day = str(ds.get("date"))
        ddf = hourly[hourly["time"].dt.strftime("%Y-%m-%d") == day].copy()
        capture_hour = None
        try:
            from PIL import Image as _I
            from PIL.ExifTags import TAGS as _T
            first_src = next(iter((BASE / "Kaggle image").glob("*.JPG")), None)
            if first_src:
                ex = _I.open(first_src)._getexif() or {}
                dto = { _T.get(k, k): v for k, v in ex.items() }.get("DateTimeOriginal")
                if dto:
                    capture_hour = int(str(dto).split()[1].split(":")[0])
                    st.caption(f"EXIF shutter time: {dto} IST (drone DJI)")
        except Exception:
            pass
        if ddf.empty:
            ddf = hourly.tail(24)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["temp_c"], name="Temperature °C", line=dict(color="#EF553B", width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["humidity_pct"], name="Humidity %", line=dict(color="#636EFA", width=2.5, dash="dot")), secondary_y=True)
        fig.add_trace(go.Bar(x=ddf["time"], y=ddf["precipitation_mm"], name="Rain mm", marker_color="#00BFFF", opacity=0.35), secondary_y=False)
        if "soil_moisture_pct" in ddf.columns and ddf["soil_moisture_pct"].notna().any():
            fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["soil_moisture_pct"], name="Soil moisture %", line=dict(color="#8B5E3C", width=2)), secondary_y=True)
        if capture_hour is not None:
            tcut = ddf["time"].iloc[0].normalize() + pd.Timedelta(hours=capture_hour)
            fig.add_vline(x=tcut.timestamp() * 1000, line=dict(color="#00E5A0", width=2, dash="dash"),
                          annotation_text=f"UAV survey {capture_hour}:00 IST", annotation_position="top")
        _plotly_layout(fig, height=420, legend=dict(orientation="h", y=1.12, bgcolor="rgba(8,13,20,0.6)"))
        fig.update_yaxes(title_text="°C / mm", secondary_y=False, gridcolor="rgba(255,255,255,0.05)")
        fig.update_yaxes(title_text="% RH / soil", secondary_y=True, gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig, width='stretch')

        with st.expander("Previous 7 days aggregation"):
            p1, p2, p3 = st.columns(3)
            p1.metric("Rainfall total", f"{prev7.get('precipitation_total_mm')} mm")
            p2.metric("Avg daily Tmax", f"{prev7.get('avg_temp_max_c')} °C")
            p3.metric("Avg RH mean", f"{prev7.get('avg_humidity_mean_pct')} %")
        with st.expander("Raw hourly table"):
            st.dataframe(ddf, width='stretch')
    else:
        st.warning("weather_hourly.csv unavailable")

elif page == "3️⃣ Zone Inspector":
    zids = [f"Z{i:02d}" for i in range(1, 10)]
    zsel = st.selectbox("Micro-zone", zids)
    zrow = int(zsel[1:]) - 1
    grid_pos = {"row": zrow // 3, "col": zrow % 3}

    hz = next((r for r in (health_report or {}).get("zones", []) if r["zone_id"] == zsel), None)
    pz = next((r for r in (pathogen_report or {}).get("zones", []) if r["zone_id"] == zsel), None)
    ez = next((r for r in (epi_report or {}).get("zones", []) if r["zone_id"] == zsel), None)

    a, b, c = st.columns(3)
    if hz:
        a.metric("Health composite", f"{hz['health']['composite_score']:.3f}", hz["health"]["severity_label"])
        a.progress(min(1.0, hz["health"]["composite_score"]))
    if pz:
        b.metric("Lesion detections", pz["detection_count"], f"sev={pz['pathogen_severity']}")
        b.metric("Anomaly coverage", f"{pz['zone_coverage_pct']:.2f}%")
    if ez:
        c.metric("Spread risk", f"{ez['spread_risk_index']:.3f}")
        _hcls = ez["hotspot_class"]
        _hcolor = CLASS_HEX.get(_hcls, "#888")
        c.markdown(_badge(_hcls, _hcolor), unsafe_allow_html=True)
        c.caption(f"Gi* z = {ez['getis_ord_gistar_z']:+.3f}")

    t = st.tabs(["Zone crop", "Health heatmap", "Detection result", "Spread risk"])
    with t[0]:
        p = img(f"micro_zones/{zsel}.jpg"); p and st.image(p, width='stretch')
    with t[1]:
        p = img(f"health_heatmaps/{zsel}_overlay.jpg"); p and st.image(p, width='stretch')
        st.caption("JET colormap of composite vegetation index over the crop")
    with t[2]:
        p = img((pz or {}).get("files", {}).get("final_annotated", ""))
        if p:
            st.image(p, width='stretch')
        if pz:
            st.write("**Class distribution:**", pz["class_distribution"])
            dets = [d for d in pathogen_report.get("detections", []) if d["zone_id"] == zsel]
            if dets:
                st.dataframe(pd.DataFrame(dets)[["detection_id", "class_label", "confidence", "area_px", "contour_fill_ratio"]], width='stretch', hide_index=True)
    with t[3]:
        p = img(f"epidemiology/zone_spread_heatmaps/{zsel}_spread_risk.jpg"); p and st.image(p, width='stretch')

    if ez and ez.get("top_adjacent_zones"):
        st.subheader("Adjacency (nearest connected zones)")
        st.dataframe(pd.DataFrame(ez["top_adjacent_zones"]), width='stretch', hide_index=True)

    diag_dir = BASE / "detections" / zsel
    if diag_dir.exists():
        with st.expander("🔬 Full 10-stage diagnostic pipeline images"):
            cols = st.columns(2)
            for i, f in enumerate(sorted(diag_dir.glob("*.jpg"))):
                with cols[i % 2]:
                    st.image(str(f), caption=f.name.replace(".jpg", ""), width='stretch')

elif page == "4️⃣ Plant Health Analytics":
    if not health_report:
        st.error("health_report.json missing"); st.stop()
    df = load_csv("metadata/health_report.csv")
    fs = health_report["field_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean score", f"{fs['mean_field_score']:.3f}")
    c2.metric("Min / Max", f"{fs['min_field_score']:.3f} / {fs['max_field_score']:.3f}")
    c3.metric("Mean stress pixels", f"{fs['mean_stress_pct']}%")
    c4.metric("Mean vegetation", f"{fs['mean_vegetation_pct']}%")

    l, r = st.columns([3, 2])
    with l:
        fig = px.bar(df.sort_values("zone_id"), x="zone_id", y="composite_score", color="severity_label",
                     color_discrete_map=SEV_HEX, title="Per-zone composite health score (higher = healthier)")
        for thr, nm in [(0.42, "SEVERE"), (0.60, "MODERATE"), (0.78, "HEALTHY")]:
            fig.add_hline(y=thr, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                          annotation_text=f"{nm} ≥{thr}", annotation_font_color="#A8C5DA")
        _plotly_layout(fig, height=380)
        st.plotly_chart(fig, width='stretch')
    with r:
        dist = fs["severity_distribution"]
        figp = px.pie(names=list(dist.keys()), values=list(dist.values()), title="Severity class distribution",
                      color_discrete_map=SEV_HEX, hole=0.4)
        figp.update_traces(textfont_color="#DCE8F5")
        _plotly_layout(figp, height=380)
        st.plotly_chart(figp, width='stretch')

    st.subheader("Stress vs vegetation pixel share")
    fig2 = go.Figure()
    fig2.add_bar(x=df["zone_id"], y=df["stress_pixels_pct"], name="Stress pixels % (<0.60)", marker_color="#EF553B", marker_line_width=0)
    fig2.add_bar(x=df["zone_id"], y=df["vegetation_pixels_pct"], name="Vegetation pixels % (≥0.50)", marker_color="#00E5A0", marker_line_width=0)
    _plotly_layout(fig2, barmode="group", height=340)
    st.plotly_chart(fig2, width='stretch')

    with st.expander("Zone-level table"):
        st.dataframe(df, width='stretch', hide_index=True)

elif page == "5️⃣ Pathogen Detections":
    if not pathogen_report:
        st.error("pathogen_report.json missing"); st.stop()
    ts = pathogen_report["field_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", ts["total_detections"])
    c2.metric("Mean coverage", f"{ts['mean_zone_coverage_pct']}%")
    c3.metric("Peak zone coverage", f"{ts['max_zone_coverage_pct']}%")
    c4.metric("Zones w/ HIGH sev", ts["severity_counts"].get("HIGH", 0))

    l, r = st.columns(2)
    with l:
        cc = ts["class_counts"]
        fig = px.bar(x=list(cc.keys()), y=list(cc.values()),
                     color=list(cc.keys()), color_discrete_map=PATHO_HEX,
                     title="Detection class distribution", labels={"x": "class", "y": "count"})
        fig.update_traces(marker_line_width=0)
        _plotly_layout(fig, showlegend=False, height=360)
        st.plotly_chart(fig, width='stretch')
    with r:
        sc = ts["severity_counts"]
        order = [s for s in ["HIGH", "MEDIUM", "LOW", "NONE"] if s in sc]
        fig2 = px.pie(values=[sc[s] for s in order], names=order, title="Zone infection severity mix",
                      color_discrete_sequence=["#DC1414", "#FF7828", "#F5D060", "#00E5A0"], hole=0.4)
        fig2.update_traces(textfont_color="#DCE8F5")
        _plotly_layout(fig2, height=360)
        st.plotly_chart(fig2, width='stretch')

    zs = load_csv("metadata/pathogen_zone_summary.csv")
    if zs is not None:
        fig3 = px.bar(zs.sort_values("zone_id"), x="zone_id", y="zone_coverage_pct", color="pathogen_severity",
                      color_discrete_map={"HIGH": "#DC1414", "MEDIUM": "#FF7828", "LOW": "#F5D060", "NONE": "#00E5A0"},
                      title="Anomaly coverage % per zone")
        fig3.update_traces(marker_line_width=0)
        _plotly_layout(fig3, height=340)
        st.plotly_chart(fig3, width='stretch')
        with st.expander("Zone summary table"):
            st.dataframe(zs, width='stretch', hide_index=True)
    with st.expander("Every raw detection (bbox-level CSV)"):
        dd = load_csv("metadata/pathogen_detections.csv")
        dd is not None and st.dataframe(dd, width='stretch', hide_index=True)

elif page == "6️⃣ Hotspot / Coldspot Map":
    if not epi_report:
        st.error("epidemiology_report.json missing"); st.stop()
    efs = epi_report["field_summary"]
    zones_df = pd.DataFrame(epi_report["zones"])
    st.info(efs.get("classification_note", ""), icon="📐")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean severity index", f"{efs['mean_severity_index']:.3f}")
    c2.metric("Std-dev (spatial contrast)", f"{efs['severity_stddev']:.3f}")
    c3.metric("Adjacency edges", efs["network_edges_count"])

    hc = efs["hotspot_counts"]
    chips = st.columns(len(hc))
    for (cls, cnt), col in zip(hc.items(), chips):
        _c = CLASS_HEX.get(cls, '#888')
        col.markdown(
            f"<div style='background:#0D1117;border:1px solid #1E2530;"
            f"border-top:2px solid {_c};"
            f"border-radius:8px;padding:14px 12px;text-align:center'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{_c};font-family:JetBrains Mono,monospace'>{cnt}</div>"
            f"<div style='font-size:0.68rem;font-weight:600;margin-top:4px;color:#475569;text-transform:uppercase;letter-spacing:0.05em'>{cls.replace('_',' ')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    p = img(artifact_index.get("orthomosaic_files", {}).get("epidemiology_overlay", ""))
    p and st.image(p, width='stretch', caption="Full-field epidemiology overlay: red/orange = hotspots, blue/yellow = coldspots")

    l, r = st.columns(2)
    with l:
        zd = zones_df.sort_values("getis_ord_gistar_z", ascending=False)
        fig = px.bar(zd, x="zone_id", y="getis_ord_gistar_z", color="hotspot_class",
                     color_discrete_map=CLASS_HEX, title="Getis-Ord Gi* z-score per zone (high = disease cluster)")
        fig.update_traces(marker_line_width=0)
        _plotly_layout(fig, height=380)
        st.plotly_chart(fig, width='stretch')
    with r:
        fig2 = px.bar(zones_df.sort_values("spread_risk_index"), x="zone_id", y=["spread_risk_index", "neighbor_weighted_severity"],
                      barmode="group", title="Spread risk vs neighbor severity", color_discrete_sequence=["#EF553B", "#00E5A0"])
        fig2.update_traces(marker_line_width=0)
        _plotly_layout(fig2, height=380)
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Zone epidemiology table")
    st.dataframe(zones_df[["zone_id", "severity_index", "getis_ord_gistar_z", "hotspot_class", "spread_risk_index", "neighbor_weighted_severity"]],
                 width='stretch', hide_index=True)

    adj = load_csv("epidemiology/epidemiology_adjacency.csv")
    if adj is not None:
        with st.expander(f"Adjacency graph ({len(adj)} weighted links)"):
            st.dataframe(adj.sort_values("weight", ascending=False), width='stretch', hide_index=True)

else:
    if not scenario_report:
        st.error("scenario_analysis_report.json missing - run finalize_pipeline.py"); st.stop()
    sa = scenario_report["scenario_analysis"]
    gs = scenario_report.get("genai_status", {})

    _src = gs.get("source", "")
    if _src in ("openai", "groq"):
        _provider = "Groq" if _src == "groq" else "OpenAI"
        st.success(f"Generated by {_provider} LLM ({gs.get('model')})", icon="🤖")
    elif _src == "rule_based_expert_system":
        st.warning("Deterministic rule-based expert engine used (no LLM key configured). Set `GROQ_API_KEY` or `OPENAI_API_KEY` in .env to auto-upgrade this report via GenAI on the next pipeline run.", icon="🧮")
    else:
        st.info(f"Analysis source: {_src or 'unknown'}", icon="ℹ️")

    st.markdown(
        f"<div style='background:#0D1117;border:1px solid #1E2530;border-left:3px solid #4ADE80;"
        f"border-radius:8px;padding:18px 20px;margin:16px 0'>"
        f"<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;"
        f"color:#334155;margin-bottom:10px'>📌 Executive Summary</div>"
        f"<div style='color:#94A3B8;font-size:0.9rem;line-height:1.7'>{sa['executive_summary']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.header("🔥 Hotspot Analysis - why these zones became infected clusters")
    ha = sa["hotspots_analysis"]
    st.markdown(ha["why_hotspots_formed"])
    st.subheader("Zone-by-zone evidence")
    for d in ha["hotspot_zones_details"]:
        with st.container(border=True):
            m1, m2 = st.columns([1, 3])
            m1.markdown(
                f"<div style='text-align:center;padding:8px 0'>"
                f"<div style='font-size:1.6rem;font-weight:700;color:#F87171;font-family:JetBrains Mono,monospace'>{d['zone_id']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            m1.markdown(_badge(d['primary_pathogen'], "#F59E0B"), unsafe_allow_html=True)
            m2.markdown(f"**Environmental driver:** {d['environmental_driver']}")
            for ev in d["evidence"]:
                m2.markdown(f"- {ev}")
    st.subheader("Spread mechanisms")
    for s in ha["spread_mechanisms"]:
        st.markdown(f"- 🌬️ {s}")

    st.divider()
    st.header("❄️ Coldspot Analysis - why these zones were NOT affected")
    ca = sa["coldspots_analysis"]
    st.markdown(ca["why_not_affected"])
    for d in ca["coldspot_zones_details"]:
        with st.container(border=True):
            m1, m2 = st.columns([1, 3])
            m1.markdown(
                f"<div style='text-align:center;padding:8px 0'>"
                f"<div style='font-size:1.6rem;font-weight:700;color:#38BDF8;font-family:JetBrains Mono,monospace'>{d['zone_id']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            m2.markdown("**Protective factors:**")
            for pf in d["protective_factors"]:
                m2.markdown(f"- {pf}")
    st.success(f"**Protective lessons:** {ca['protective_lessons']}", icon="🎓")

    st.divider()
    st.header("🌦️ Environmental Inferences (weather-linked)")
    ei = sa["environmental_inferences"]
    _env_cfg = [
        ("temperature_effect",          "🌡️", "Temperature effect",           "#EF553B"),
        ("humidity_leaf_wetness_effect", "💧", "Humidity / leaf-wetness effect","#636EFA"),
        ("rainfall_effect_7d",           "🌧️", "7-day rainfall effect",         "#00BFFF"),
        ("wind_dispersal_risk",          "🌬️", "Wind dispersal risk",           "#8CAA8C"),
        ("soil_moisture_inference",      "🟤", "Soil moisture inference",       "#8B5E3C"),
    ]
    _ei_cols = st.columns(2)
    _ei_items = [(k, ic, lb, cl) for k, ic, lb, cl in _env_cfg if k in ei]
    for _idx, (k, ic, lb, cl) in enumerate(_ei_items):
        with _ei_cols[_idx % 2]:
            st.markdown(
                f"<div style='background:#0D1117;border:1px solid #1E2530;"
                f"border-left:2px solid {cl};border-radius:8px;padding:14px 16px;margin-bottom:10px'>"
                f"<div style='font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#334155;margin-bottom:6px'>{ic} {lb}</div>"
                f"<div style='color:#64748B;font-size:0.85rem;line-height:1.6'>{ei[k]}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.header("📈 Risk Inferences")
    ri = sa["risk_inferences"]
    r1, r2 = st.columns(2)
    r1.error(f"**72-hour outlook**\n\n{ri['short_term_72h_risk']}")
    r2.warning(f"**2-week outlook**\n\n{ri['medium_term_2week_risk']}")
    st.error(f"**Yield risk assessment**\n\n{ri['yield_risk_assessment']}", icon="🌾")
    st.markdown(
        "<span style='font-size:0.78rem;color:#475569;text-transform:uppercase;font-weight:600;letter-spacing:0.06em'>Highest-risk zones</span>  " +
        " → ".join(
            f"<span style='background:#1A0D0D;color:#F87171;border:1px solid #2D1515;"
            f"border-radius:5px;padding:2px 9px;font-family:JetBrains Mono,monospace;font-size:0.83rem;font-weight:600'>{z}</span>"
            for z in ri["highest_risk_zones_ordered"]
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    st.header("✅ Recommended Actions")
    ra = sa["recommended_actions"]
    p1, p2, p3 = st.columns(3)
    _priority_cfg = [
        (p1, "P1", "within 24h",  ra["priority_1_immediate_24h"],    "#F87171"),
        (p2, "P2", "1–3 days",    ra["priority_2_shortterm_1_3d"],   "#F59E0B"),
        (p3, "P3", "1–2 weeks",   ra["priority_3_mediumterm_1_2wk"], "#4ADE80"),
    ]
    for _col, _pnum, _ptime, _actions, _color in _priority_cfg:
        with _col:
            _col.markdown(
                f"<div style='margin-bottom:14px'>"
                f"<span style='background:{_color}18;color:{_color};border:1px solid {_color}35;"
                f"border-radius:5px;padding:2px 8px;font-size:0.7rem;font-weight:700;font-family:JetBrains Mono,monospace'>{_pnum}</span>"
                f"<span style='color:#334155;font-size:0.75rem;margin-left:8px'>{_ptime}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            for _a in _actions:
                _col.markdown(f"<div style='color:#64748B;font-size:0.83rem;padding:4px 0;border-bottom:1px solid #111418'>— {_a}</div>", unsafe_allow_html=True)
    st.subheader("Zone-specific prescriptions")
    rx = pd.DataFrame(ra["zone_specific_prescriptions"])
    st.dataframe(rx, width='stretch', hide_index=True)

    st.divider()
    st.header("🛠️ Mitigation Plan")
    mp = sa["mitigation_plan"]
    t1, t2 = st.tabs(["Cultural + chemical program", "Biological + irrigation/nutrition + surveillance"])
    with t1:
        st.markdown("**Cultural practices**")
        for cp in mp["cultural_practices"]:
            st.markdown(f"- {cp}")
        st.markdown("**Chemical fungicide program**")
        st.info(mp["chemical_fungicide_program"], icon="💊")
    with t2:
        st.markdown("**Biological options**")
        for bo in mp["biological_options"]:
            st.markdown(f"- 🧫 {bo}")
        st.markdown("**Irrigation & nutrition adjustments**")
        for ia in mp["irrigation_nutrition_adjustments"]:
            st.markdown(f"- 💦 {ia}")
        st.markdown("**Follow-up surveillance schedule**")
        st.info(mp["followup_surveillance_schedule"], icon="📅")

    st.divider()
    with st.expander("⚠️ Confidence & limitations notes"):
        st.caption(sa["confidence_notes"])

    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("Download scenario JSON",     json.dumps(scenario_report, indent=2),              "scenario_analysis_report.json")
    dl2.download_button("Download epidemiology JSON", json.dumps(epi_report, indent=2),                   "epidemiology_report.json")
    dl3.download_button("Download weather JSON",      json.dumps(weather_report, indent=2, default=str), "weather_report.json")

st.markdown("<div style='height:1px;background:#1E2530;margin:2rem 0 1rem 0'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#1E2D3D;font-size:0.73rem;text-align:center;padding-bottom:1rem'>"
    "Smart India Hackathon &nbsp;·&nbsp; UAV crop-health digital twin &nbsp;·&nbsp; "
    "stitching → ExG/VARI/NDVIproxy/MGRVI → HSV/LAB pathogen detection → Getis-Ord Gi* epidemiology → GenAI advisory"
    "</div>",
    unsafe_allow_html=True,
)
