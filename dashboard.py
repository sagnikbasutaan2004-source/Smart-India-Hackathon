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

CLASS_HEX = {
    "HOTSPOT_HIGH": "#DC1414",
    "HOTSPOT_MODERATE": "#FF7828",
    "COLDSPOT_HIGH": "#2878C8",
    "COLDSPOT_MODERATE": "#50BEE6",
    "NEUTRAL": "#8CAA8C",
}
SEV_HEX = {"HEALTHY": "#2EBD32", "MILD_STRESS": "#2BDBFF", "MODERATE_STRESS": "#2080FF", "SEVERE_STRESS": "#2020DC"}
PATHO_HEX = {"Rust_Lesion": "#DC283C", "LeafSpot_Brown": "#965A28", "Chlorosis_Yellow": "#FFDC1E", "General_Lesion": "#B450C8"}


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
    st.title("🛰️ AgriDrone Digital Twin")
    st.caption("Drone orthomosaic → micro-zones → health → pathogens → weather → spatial epidemiology → GenAI advisory")
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
        st.divider()
        st.metric("Pipeline version", artifact_index.get("pipeline_version", "?"))
        gps = artifact_index.get("gps_center") or {}
        st.caption(f"Field center: {gps.get('lat', '?'):.5f}N, {gps.get('lon', '?'):.5f}E" if gps else "")
        st.caption(f"{artifact_index.get('zone_count', 9)} micro-zones | Korba, Chhattisgarh")
        st.success("All modules complete ✔", icon="✅")
        if scenario_report:
            gs = scenario_report.get("genai_status", {})
            st.info(f"Scenario engine: **{gs.get('source')}**\n\nModel: `{gs.get('model')}`", icon="🤖")

st.title({"1️⃣ Field Overview": "🗺️ Field Digital Twin - Overview",
          "2️⃣ Weather @ Capture Time": "🌦️ Local Weather at Capture Time",
          "3️⃣ Zone Inspector": "🔍 Micro-Zone Inspector",
          "4️⃣ Plant Health Analytics": "🌿 Plant Health Analytics",
          "5️⃣ Pathogen Detections": "🦠 Pathogen Detection Results",
          "6️⃣ Hotspot / Coldspot Map": "📍 Spatial Epidemiology - Hotspots & Coldspots",
          "7️⃣ FULL SCENARIO ANALYSIS": "🧠 Full Scenario Analysis & Advisory"}[page])

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

    st.subheader("📦 Output catalog")
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.write(f"- Stitching stage images: **{len(artifact_index.get('stitching_stages', []))}**")
        st.write(f"- Micro-zone crops: **{len(list((BASE / 'micro_zones').glob('*.jpg')))}**")
        st.write(f"- Health heatmap renders: **{len(list((BASE / 'health_heatmaps').glob('*.jpg')))}**")
        det_n = len(list((BASE / 'detections').rglob('*.jpg')))
        st.write(f"- Pathogen diagnostics: **{det_n}**")
        sp_n = len(list((BASE / 'epidemiology' / 'zone_spread_heatmaps').glob('*.jpg'))) if (BASE / 'epidemiology').exists() else 0
        st.write(f"- Spread-risk renders: **{sp_n}**")
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
        col.markdown(f"**{label}**")
        (col.success if val else col.error)("ACTIVE" if val else "not active")
        col.caption(why)
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
        fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["temp_c"], name="Temperature °C", line=dict(color="#EF553B", width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["humidity_pct"], name="Humidity %", line=dict(color="#636EFA", width=3, dash="dot")), secondary_y=True)
        fig.add_trace(go.Bar(x=ddf["time"], y=ddf["precipitation_mm"], name="Rain mm", marker_color="#33A1FF", opacity=0.4), secondary_y=False)
        if "soil_moisture_pct" in ddf.columns and ddf["soil_moisture_pct"].notna().any():
            fig.add_trace(go.Scatter(x=ddf["time"], y=ddf["soil_moisture_pct"], name="Soil moisture %", line=dict(color="#8B5E3C", width=2)), secondary_y=True)
        if capture_hour is not None:
            tcut = ddf["time"].iloc[0].normalize() + pd.Timedelta(hours=capture_hour)
            fig.add_vline(x=tcut.timestamp() * 1000, line=dict(color="#22C55E", width=2, dash="dash"),
                          annotation_text=f"UAV survey {capture_hour}:00 IST", annotation_position="top")
        fig.update_layout(height=420, legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=30, b=10))
        fig.update_yaxes(title_text="°C / mm", secondary_y=False)
        fig.update_yaxes(title_text="% RH / soil", secondary_y=True)
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
        c.metric("Hotspot class", ez["hotspot_class"])
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
            fig.add_hline(y=thr, line_dash="dot", annotation_text=f"{nm} ≥{thr}")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width='stretch')
    with r:
        dist = fs["severity_distribution"]
        figp = px.pie(names=list(dist.keys()), values=list(dist.values()), title="Severity class distribution",
                      color_discrete_map=SEV_HEX)
        figp.update_layout(height=380)
        st.plotly_chart(figp, width='stretch')

    st.subheader("Stress vs vegetation pixel share")
    fig2 = go.Figure()
    fig2.add_bar(x=df["zone_id"], y=df["stress_pixels_pct"], name="Stress pixels % (<0.60)", marker_color="#EF553B")
    fig2.add_bar(x=df["zone_id"], y=df["vegetation_pixels_pct"], name="Vegetation pixels % (≥0.50)", marker_color="#2EBD32")
    fig2.update_layout(barmode="group", height=340)
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
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, width='stretch')
    with r:
        sc = ts["severity_counts"]
        order = [s for s in ["HIGH", "MEDIUM", "LOW", "NONE"] if s in sc]
        fig2 = px.pie(values=[sc[s] for s in order], names=order, title="Zone infection severity mix",
                      color_discrete_sequence=["#DC1414", "#FF7828", "#F5D060", "#77C97F"])
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, width='stretch')

    zs = load_csv("metadata/pathogen_zone_summary.csv")
    if zs is not None:
        fig3 = px.bar(zs.sort_values("zone_id"), x="zone_id", y="zone_coverage_pct", color="pathogen_severity",
                      color_discrete_map={"HIGH": "#DC1414", "MEDIUM": "#FF7828", "LOW": "#F5D060", "NONE": "#77C97F"},
                      title="Anomaly coverage % per zone")
        fig3.update_layout(height=340)
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
        col.markdown(f"<div style='background:{CLASS_HEX.get(cls,'#888')};color:#fff;border-radius:8px;padding:10px;text-align:center'><b>{cnt}</b><br>{cls}</div>", unsafe_allow_html=True)

    p = img(artifact_index.get("orthomosaic_files", {}).get("epidemiology_overlay", ""))
    p and st.image(p, width='stretch', caption="Full-field epidemiology overlay: red/orange = hotspots, blue/yellow = coldspots")

    l, r = st.columns(2)
    with l:
        zd = zones_df.sort_values("getis_ord_gistar_z", ascending=False)
        fig = px.bar(zd, x="zone_id", y="getis_ord_gistar_z", color="hotspot_class",
                     color_discrete_map=CLASS_HEX, title="Getis-Ord Gi* z-score per zone (high = disease cluster)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, width='stretch')
    with r:
        fig2 = px.bar(zones_df.sort_values("spread_risk_index"), x="zone_id", y=["spread_risk_index", "neighbor_weighted_severity"],
                      barmode="group", title="Spread risk vs neighbor severity", color_discrete_sequence=["#EF553B", "#8CAA8C"])
        fig2.update_layout(height=380)
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

    if gs.get("source") == "openai":
        st.success(f"Generated by LLM ({gs.get('model')})", icon="🤖")
    else:
        st.warning("Deterministic rule-based expert engine used (no LLM key configured). Set `OPENAI_API_KEY` env/.env to auto-upgrade this report via GenAI on the next pipeline run.", icon="🧮")

    st.header("📌 Executive Summary")
    st.info(sa["executive_summary"], icon="📋")

    st.divider()
    st.header("🔥 Hotspot Analysis - why these zones became infected clusters")
    ha = sa["hotspots_analysis"]
    st.markdown(ha["why_hotspots_formed"])
    st.subheader("Zone-by-zone evidence")
    for d in ha["hotspot_zones_details"]:
        with st.container(border=True):
            m1, m2 = st.columns([1, 3])
            m1.markdown(f"### {d['zone_id']}")
            m1.markdown(f"`primary pathogen:` **{d['primary_pathogen']}**")
            m2.markdown(f"**Environmental driver:** {d['environmental_driver']}")
            for ev in d["evidence"]:
                m2.markdown(f"- 🔸 {ev}")
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
            m1.markdown(f"### {d['zone_id']}")
            m2.markdown("**Protective factors:**")
            for pf in d["protective_factors"]:
                m2.markdown(f"- 🛡️ {pf}")
    st.success(f"**Protective lessons:** {ca['protective_lessons']}", icon="🎓")

    st.divider()
    st.header("🌦️ Environmental Inferences (weather-linked)")
    ei = sa["environmental_inferences"]
    icons = {"temperature_effect": "🌡️ Temperature effect", "humidity_leaf_wetness_effect": "💧 Humidity / leaf-wetness effect",
             "rainfall_effect_7d": "🌧️ 7-day rainfall effect", "wind_dispersal_risk": "🌬️ Wind dispersal risk",
             "soil_moisture_inference": "🟤 Soil moisture inference"}
    for k, label in icons.items():
        if k in ei:
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.markdown(ei[k])

    st.divider()
    st.header("📈 Risk Inferences")
    ri = sa["risk_inferences"]
    r1, r2 = st.columns(2)
    r1.error(f"**72-hour outlook**\n\n{ri['short_term_72h_risk']}")
    r2.warning(f"**2-week outlook**\n\n{ri['medium_term_2week_risk']}")
    st.error(f"**Yield risk assessment**\n\n{ri['yield_risk_assessment']}", icon="🌾")
    st.markdown("**Highest-risk zones (ordered):** " + " → ".join(f"`{z}`" for z in ri["highest_risk_zones_ordered"]))

    st.divider()
    st.header("✅ Recommended Actions")
    ra = sa["recommended_actions"]
    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("P1 · within 24h")
        for a in ra["priority_1_immediate_24h"]:
            st.markdown(f"- 🔴 {a}")
    with p2:
        st.subheader("P2 · 1–3 days")
        for a in ra["priority_2_shortterm_1_3d"]:
            st.markdown(f"- 🟠 {a}")
    with p3:
        st.subheader("P3 · 1–2 weeks")
        for a in ra["priority_3_mediumterm_1_2wk"]:
            st.markdown(f"- 🟡 {a}")
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
    dl1.download_button("Download scenario JSON", json.dumps(scenario_report, indent=2), "scenario_analysis_report.json")
    dl2.download_button("Download epidemiology JSON", json.dumps(epi_report, indent=2), "epidemiology_report.json")
    dl3.download_button("Download weather JSON", json.dumps(weather_report, indent=2, default=str), "weather_report.json")

st.divider()
st.caption("Smart India Hackathon · UAV crop-health digital twin · stitching → ExG/VARI/NDVIproxy/MGRVI health → HSV/LAB anomaly pathogen detection → Getis-Ord Gi* spatial epidemiology → weather-correlated GenAI advisory")
