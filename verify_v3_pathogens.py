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
    kb = round(os.path.getsize(path) / 1024, 1)
    return str(w) + "x" + str(h) + " px  " + str(kb) + " KB"

print("=" * 70)
print("VERIFY PIPELINE v3.0 (PATHOGEN STAGE + ALL PREVIOUS)")
print("=" * 70)

print("\n[1] ORTHOMOSAIC VARIANTS (4 total - Streamlit compare widget)")
variants = [
    ("A: Raw stitched (NO annotation)", "./artifacts/03_final_stitched_canvas_NO_ANNOTATION_RAW.jpg"),
    ("B: 3x3 grid boxes", "./field_twin_grid.jpg"),
    ("C: Health severity overlay", "./field_health_overlay.jpg"),
    ("D: Pathogen severity overlay", "./field_pathogen_overlay.jpg"),
]
for name, path in variants:
    ok = os.path.exists(path)
    mark = "[" + (CHECK if ok else FAIL) + "]"
    print("  " + mark + " " + name)
    if ok:
        print("      " + pic(path))

print("\n[2] STITCHING INTERMEDIATES (for Streamlit slider)")
art = sorted(os.listdir("./artifacts"))
im_files = [f for f in art if f.lower().endswith((".jpg", ".jpeg", ".png"))]
json_files = [f for f in art if f.endswith(".json")]
print("  Total image stages: " + str(len(im_files)) + " images, JSON files: " + str(len(json_files)))
for f in sorted(im_files):
    print("    - " + f)

print("\n[3] MICRO ZONES")
zd = "./micro_zones"
z_ok = all(os.path.exists(os.path.join(zd, "Z" + str(i).zfill(2) + ".jpg")) for i in range(1, 10))
print("  All 9 zone crops present: " + str(z_ok))

print("\n[4] HEALTH HEATMAPS (3 x 9 = 27)")
hd = "./health_heatmaps"
count = 0
for z in range(1, 10):
    zid = "Z" + str(z).zfill(2)
    for suf in ["_composite_raw.jpg", "_heatmap.jpg", "_overlay.jpg"]:
        if os.path.exists(os.path.join(hd, zid + suf)):
            count += 1
print("  Expected 27, found: " + str(count))

print("\n[5] PATHOGEN DETECTION DIAGNOSTICS (10 images x 9 zones = 90 expected)")
dd = "./detections"
zids = ["Z" + str(i).zfill(2) for i in range(1, 10)]
total_det_imgs = 0
expected_names = None
for zid in zids:
    zp = os.path.join(dd, zid)
    if not os.path.isdir(zp):
        continue
    files_here = sorted([f for f in os.listdir(zp) if f.lower().endswith(".jpg")])
    total_det_imgs += len(files_here)
    if expected_names is None:
        expected_names = files_here
print("  Total detection diagnostic images found: " + str(total_det_imgs))
print("  Per-zone image names (Z01 sample):")
if expected_names:
    for n in expected_names:
        sz = pic(os.path.join(dd, "Z01", n))
        print("    " + n + " -> " + sz)

print("\n[6] PATHOGEN STAGES SAMPLE - Z05 CENTER ZONE:")
z5_samples = [
    ("LAB a* red pigment",        "./detections/Z05/01_LAB_a_channel_RedPigment.jpg"),
    ("Green veg mask",            "./detections/Z05/02_HSV_GreenVegetationMask.jpg"),
    ("Morph cleaned mask",        "./detections/Z05/07_AnomalyMask_afterMorphCleanup.jpg"),
    ("Final detections annotated", "./detections/Z05/09_FinalDetections_Annotated.jpg"),
    ("Class heatmap overlay",     "./detections/Z05/10_DetectionClass_Heatmap.jpg"),
]
for n, p in z5_samples:
    print("  [" + n + "]: " + pic(p))

print("\n[7] METADATA REPORTS (expected 6 files)")
expected_paths = [
    "./metadata/zone_manifest.json",
    "./metadata/health_report.json",
    "./metadata/health_report.csv",
    "./metadata/pathogen_report.json",
    "./metadata/pathogen_detections.csv",
    "./metadata/pathogen_zone_summary.csv",
]
for p in expected_paths:
    ok = os.path.exists(p)
    mark = "[" + (CHECK if ok else FAIL) + "]"
    print("  " + mark + " " + os.path.basename(p))
    if ok and p.endswith(".csv"):
        df = pd.read_csv(p)
        print("      shape: " + str(df.shape[0]) + " rows x " + str(df.shape[1]) + " cols")
    if ok and p.endswith("pathogen_report.json"):
        with open(p) as fh:
            d = json.load(fh)
        tot = d.get("field_summary", {})
        print("      total_detections: " + str(tot.get("total_detections")))
        print("      class_counts: " + str(tot.get("class_counts")))
        print("      severity_counts: " + str(tot.get("severity_counts")))

print("\n[8] ARTIFACT INDEX JSON CONTENT")
ai = "./artifacts/artifact_index.json"
if os.path.exists(ai):
    with open(ai) as fh:
        idx = json.load(fh)
    print("  pipeline_version: " + str(idx.get("pipeline_version", "?")))
    print("  stitching_stages count: " + str(len(idx.get("stitching_stages", []))))
    print("  orthomosaic_files keys: " + str(list(idx.get("orthomosaic_files", {}).keys())))
    print("  metadata_files keys: " + str(list(idx.get("metadata_files", {}).keys())))
    print("  micro_zones_dir: " + str(idx.get("micro_zones_dir")))
    print("  health_heatmaps_dir: " + str(idx.get("health_heatmaps_dir")))
    print("  detections_dir: " + str(idx.get("detections_dir")))
    print("  gps_center: " + str(idx.get("gps_center")))

print("\n[9] GRAND TOTALS")
total_imgs = 0
for root, _, files in os.walk("."):
    skip = False
    for bad in ["Kaggle", "site-packages", ".git"]:
        if bad in root:
            skip = True
            break
    if skip:
        continue
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            total_imgs += 1
print("  Total pipeline-generated images (excluding source dataset): ~" + str(total_imgs))

print("\n" + "=" * 70)
print("VERIFICATION DONE")
print("=" * 70)
