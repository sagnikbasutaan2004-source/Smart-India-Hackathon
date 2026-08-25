# 🛰️ AgriDrone Digital Twin - Field Intelligence & Spatial Epidemiology

An end-to-end UAV crop-health digital twin system built for precision agriculture. The platform processes raw drone aerial orthomosaics into 9 GPS-tagged micro-zones, extracts vegetation health metrics (ExG, VARI, NDVI-proxy, MGRVI), detects pathogen anomalies (HSV/LAB color segmentation), performs Getis-Ord $G_i^*$ spatial epidemiology clustering, and generates GenAI agro-meteorological advisories linked to local weather data.

---

## 🌟 Key Features

1. **🗺️ Field Digital Twin Overview**: Interactive full-field orthomosaic with 3x3 grid partitioning, health overlays, and pathogen anomaly distributions.
2. **🌦️ Capture-Time Weather Correlation**: Integrates historical hourly temperature, humidity, rainfall, and leaf-wetness risk from Open-Meteo archive APIs.
3. **🔍 Micro-Zone Inspector**: Deep-dive analysis for each zone ($Z01$–$Z09$), displaying multi-band heatmaps, contour-level diagnostic bboxes, and spread-risk indices.
4. **🌿 Plant Health Analytics**: Multi-index composite vegetation scoring with severity classification (`HEALTHY`, `MILD_STRESS`, `MODERATE_STRESS`, `SEVERE_STRESS`).
5. **🦠 Pathogen Detection Engine**: Unsupervised HSV/LAB color-space lesion detection for Rust, Leaf Spot, Chlorosis, and General Lesions.
6. **📍 Spatial Epidemiology Map**: Spatial autocorrelation using Getis-Ord $G_i^*$ statistics to classify disease hotspots/coldspots and calculate adjacency risk vectors.
7. **🧠 Scenario Analysis & Advisory**: Automated agronomic risk forecasting, priority intervention schedules (24h / 1–3d / 1–2wk), chemical/biological recommendations, and GenAI LLM synthesis.

---

## 🏗️ Architecture & Pipeline

```
Raw Drone Imagery (.JPG)
        │
        ▼
[Reconstruction Engine] ──► 3x3 Micro-Zone Grid Crops
        │
        ├──► [Vegetation Index Engine] (ExG / VARI / NDVI-proxy / MGRVI)
        │
        ├──► [Pathogen Diagnostic Pipeline] (HSV / LAB Segmentation)
        │
        └──► [Weather API Integration] (Open-Meteo Capture Window)
        │
        ▼
[Spatial Epidemiology Engine] ──► Getis-Ord Gi* Hotspots & Adjacency Risk
        │
        ▼
[GenAI Scenario Engine] ──► Prescriptive Agronomic Advisory
        │
        ▼
[Streamlit Interactive Dashboard] (dashboard.py)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9+
- `pip` package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Smart-India-Hackathon.git
cd Smart-India-Hackathon

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Cloud Deployment Options

### Option A: Streamlit Community Cloud (Recommended - Free 1-Click)
1. Fork / push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app** -> Select your repository `Smart-India-Hackathon` -> Main file: `dashboard.py`.
4. Click **Deploy!**

### Option B: Vercel / Render / Railway
- Deploy as a Web Service on Render or Railway using `streamlit run dashboard.py --server.port $PORT`.

---

## 📁 Repository Structure

```
.
├── dashboard.py                  # Main Streamlit Dashboard UI
├── finalize_pipeline.py          # End-to-end Pipeline Finalizer
├── reconstruction.py             # Drone Image Processing & Feature Matching Engine
├── verify_full_pipeline.py       # Full Pipeline Verification Script
├── verify_outputs.py             # Artifact Verification Utility
├── verify_v3_pathogens.py        # Pathogen Detection Verification
├── requirements.txt              # Dependencies
├── .gitignore                    # Git Exclusion Rules
├── .streamlit/
│   └── config.toml               # Dashboard Theme Settings
├── artifacts/                    # Pipeline Artifact Index & State
├── detections/                   # Zone-level Pathogen Diagnostics & Bounding Boxes
├── epidemiology/                 # Spatial Gi* Hotspot Maps & Adjacency Matrices
├── health_heatmaps/              # JET Colormap Vegetation Heatmaps
├── metadata/                     # JSON/CSV Reports (Health, Weather, Scenario, Zones)
└── micro_zones/                  # Cropped 3x3 Micro-Zone Drone Imagery
```

---

## 📄 License
Developed for Smart India Hackathon.
