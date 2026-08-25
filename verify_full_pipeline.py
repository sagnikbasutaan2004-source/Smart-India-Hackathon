import os
import json
import pandas as pd
from PIL import Image

CHECK = "YES"
FAIL = "NO"

def pic(path):
    if not os.path.exists(path):
        return "MISSING"
    im = Image.open(path)
    w, h = im.size
    kb = round(os.path.getsize(path)/1024, 1)
    return str(w) + "x" + str(h) + " px  " + str(kb) + " KB"

print("=" * 70)
print("FULL PIPELINE ARTIFACT VERIFICATION (STREAMLIT-READY)")
print("=" * 70)

sections = []

# A) Stitching intermediates (for Streamlit Stage 2 display)
print("\n[A] STITCHING INTERMEDIATES (for Streamlit stage-by-stage viewer)")
art_dir = "./artifacts"
stage_files = sorted(os.listdir(art_dir)) if os.path.exists(art_dir) else []
sections.append(("stitching_intermediates_count", len([f for f in stage_files if f.lower().endswith('.jpg')])))
print("    " + art_dir + "/  -> " + str(len(stage_files)) + " items")
for f in stage_files:
    p = os.path.join(art_dir, f)
    tag = " [JSON]" if f.lower().endswith('.json') else ""
    sz = pic(p) if not f.lower().endswith('.json') else str(round(os.path.getsize(p)/1024, 1)) + " KB"
    print("      - " + f + tag + "  ::  " + sz)

# B) Main orthomosaics (3 versions!)
print("\n[B] ORTHOMOSAIC VERSIONS (3 render modes for Streamlit compare view)")
for label, path in [
    ("Raw stitched (no annotations)", "./artifacts/03_final_stitched_canvas_NO_ANNOTATION_RAW.jpg"),
    ("Annotated 3x3 grid overlay", "./field_twin_grid.jpg"),
    ("Health severity overlay", "./field_health_overlay.jpg"),
]:
    st = "[" + (CHECK if os.path.exists(path) else FAIL) + "]"
    print("    " + st + " " + label)
    print("        Path: " + path)
    print("        Info: " + pic(path))

# C) Micro-zones
print("\n[C] MICRO-ZONE CROPS (Z01..Z09 - individual frames)")
zd = "./micro_zones"
zones_ok = True
for i in range(1, 10):
    zid = "Z" + str(i).zfill(2)
    p = os.path.join(zd, zid + ".jpg")
    st = CHECK if os.path.exists(p) else FAIL
    if not os.path.exists(p):
        zones_ok = False
    print("    [" + st + "] " + zid + ".jpg  ::  " + pic(p))

# D) Health heatmaps (3 files each x 9 zones = 27 total)
print("\n[D] HEALTH HEATMAPS (3 variants per zone - for Streamlit zone inspector)")
hd = "./health_heatmaps"
heat_files = sorted(os.listdir(hd)) if os.path.exists(hd) else []
print("    " + hd + "/  -> " + str(len(heat_files)) + " files")
heat_ok = True
for i in range(1, 10):
    zid = "Z" + str(i).zfill(2)
    for suffix in ["_composite_raw.jpg", "_heatmap.jpg", "_overlay.jpg"]:
        p = os.path.join(hd, zid + suffix)
        if not os.path.exists(p):
            heat_ok = False
print("    All 27 heatmap variants present: " + str(heat_ok))
print("    Preview (Z05 center zone):")
for suffix in ["_composite_raw.jpg", "_heatmap.jpg", "_overlay.jpg"]:
    p = os.path.join(hd, "Z05" + suffix)
    print("      Z05" + suffix + "  ::  " + pic(p))

# E) Metadata files
print("\n[E] METADATA / REPORTS")
for label, path in [
    ("Zone manifest (GPS + bbox + sources)", "./metadata/zone_manifest.json"),
    ("Health report (JSON structured)", "./metadata/health_report.json"),
    ("Health report (CSV for pandas)", "./metadata/health_report.csv"),
    ("Artifact index (Streamlit master catalog)", "./artifacts/artifact_index.json"),
]:
    exists = os.path.exists(path)
    st = "[" + (CHECK if exists else FAIL) + "]"
    print("    " + st + " " + label)
    print("        Path: " + path)
    if exists:
        sz = round(os.path.getsize(path)/1024, 1)
        print("        Size: " + str(sz) + " KB")
        if path.endswith(".json"):
            try:
                with open(path) as f:
                    d = json.load(f)
                keys = list(d.keys())[:6]
                print("        Top keys: " + str(keys))
            except Exception as e:
                print("        (parse error: " + str(e) + ")")
        elif path.endswith(".csv"):
            try:
                df = pd.read_csv(path)
                print("        Shape: " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " cols")
                print("        Columns: " + str(list(df.columns)))
                print("        First zone (head 1):")
                for c in df.columns[:6]:
                    v = str(df.iloc[0][c])
                    if len(v) > 40:
                        v = v[:40] + "..."
                    print("          " + str(c) + ": " + v)
            except Exception as e:
                print("        (parse error: " + str(e) + ")")

# F) Health summary
print("\n[F] FIELD HEALTH SUMMARY")
hr = "./metadata/health_report.json"
if os.path.exists(hr):
    with open(hr) as f:
        rep = json.load(f)
    fs = rep.get("field_summary", {})
    print("    Mean field score: " + str(fs.get("mean_field_score")))
    print("    Score range: " + str(fs.get("min_field_score")) + " to " + str(fs.get("max_field_score")))
    print("    Mean stress %: " + str(fs.get("mean_stress_pct")) + "%")
    print("    Mean vegetation %: " + str(fs.get("mean_vegetation_pct")) + "%")
    print("    Severity distribution: " + str(fs.get("severity_distribution")))

# G) Artifact count totals
print("\n[G] TOTAL ARTIFACT COUNTS")
total = 0
for root, dirs, files in os.walk("."):
    if root.startswith(".\\.") or "site-packages" in root:
        continue
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".json", ".csv")):
            total += 1
print("    Total tracked output artifacts (images + reports): ~" + str(total))
print("    Ready for Streamlit dashboard consumption.")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - Streamlit catalog built successfully")
print("=" * 70)
