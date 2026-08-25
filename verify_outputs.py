import os
import json
from PIL import Image

CHECK = "[OK]"
FAIL = "[MISSING]"

print("=" * 65)
print("ARTIFACT VERIFICATION REPORT - Micro Zones + Metadata")
print("=" * 65)

zone_dir = "./micro_zones"
expected = [f"Z{i:02d}.jpg" for i in range(1, 10)]
found = sorted(os.listdir(zone_dir)) if os.path.exists(zone_dir) else []
print("\n[1] MICRO-ZONE CROPS (" + zone_dir + "/)")
print("    Expected: " + str(len(expected)) + " files  |  Found: " + str(len(found)) + " files")
all_zones_ok = all(e in found for e in expected)
status = CHECK if all_zones_ok else FAIL
print("    All present: " + status)
for z in expected:
    p = os.path.join(zone_dir, z)
    if os.path.exists(p):
        im = Image.open(p)
        sz = os.path.getsize(p)
        line = "      " + z + "  " + str(im.size[0]).rjust(5) + "x" + str(im.size[1]).ljust(5) + " px   "
        line += str(round(sz/1024, 1)).rjust(6) + " KB"
        print(line)

ortho = "./field_twin_grid.jpg"
if os.path.exists(ortho):
    im = Image.open(ortho)
    print("\n[2] ANNOTATED ORTHOMOSAIC: " + CHECK)
    print("      Path: " + ortho)
    line = "      Size: " + str(im.size[0]) + "x" + str(im.size[1])
    line += " px  |  " + str(round(os.path.getsize(ortho)/1024, 1)) + " KB"
    print(line)
else:
    print("\n[2] ANNOTATED ORTHOMOSAIC: " + FAIL)

print("\n[3] ZONE MANIFEST JSON  (./metadata/zone_manifest.json)")
mp = "./metadata/zone_manifest.json"
if os.path.exists(mp):
    with open(mp, "r") as f:
        manifest = json.load(f)
    print("      Valid JSON: " + CHECK)
    print("      Manifest Version: " + str(manifest.get("manifest_version", "N/A")))
    ortho_info = manifest.get("orthomosaic", {})
    dims = ortho_info.get("dimensions_px", {})
    print("      Orthomosaic dims: " + str(dims))
    gb = ortho_info.get("gps_bounds", {})
    if gb:
        line = "      GPS center: (" + str(gb.get("lat_center", "?"))
        line += ", " + str(gb.get("lon_center", "?")) + ")"
        print(line)
        line = "      Lat range: " + str(round(gb["lat_min"], 6))
        line += " -> " + str(round(gb["lat_max"], 6))
        print(line)
        line = "      Lon range: " + str(round(gb["lon_min"], 6))
        line += " -> " + str(round(gb["lon_max"], 6))
        print(line)
        line = "      Avg flight height: " + str(gb["avg_flight_height_m"]) + " m"
        print(line)
    gl = ortho_info.get("grid_layout", {})
    line = "      Grid layout: " + str(gl.get("rows")) + " rows x " + str(gl.get("cols")) + " cols"
    print(line)
    zones = manifest.get("micro_zones", [])
    sources = manifest.get("source_images", [])
    line = "      Micro-zones: " + str(len(zones)) + "  |  Source images: " + str(len(sources))
    print(line)

    if zones:
        z = zones[0]
        print("\n      [Sample Zone: " + z["zone_id"] + "]")
        line = "        Grid (r,c): (" + str(z["grid_position"]["row"])
        line += ", " + str(z["grid_position"]["col"]) + ")"
        print(line)
        pb = z["pixel_bbox"]
        line = "        Pixel bbox: x1=" + str(pb["x1"]) + ", y1=" + str(pb["y1"])
        line += ", x2=" + str(pb["x2"]) + ", y2=" + str(pb["y2"])
        print(line)
        line = "        Dimensions: " + str(pb["width_px"]) + "w x " + str(pb["height_px"]) + "h px"
        print(line)
        gp = z["gps_bounds"]
        ctr = gp["center"]
        line = "        GPS center: (" + str(ctr["lat"]) + ", " + str(ctr["lon"]) + ")"
        print(line)
        print("        Crop file:  " + z["crop_image"])

    if sources:
        s = sources[0]
        print("\n      [Sample Source: " + s["filename"] + "]")
        print("        Flight height: " + str(s["flight_height_m"]) + " m")
        line = "        GPS lat: " + str(s["gps"]["lat"])
        line += "  |  lon: " + str(s["gps"]["lon"])
        print(line)
        orig = s["original_size"]
        line = "        Original res: " + str(orig["width_px"]) + "x" + str(orig["height_px"]) + " px"
        print(line)
else:
    print("      " + FAIL)

print("\n" + "=" * 65)
all_ok = all_zones_ok and os.path.exists(mp) and os.path.exists(ortho)
if all_ok:
    print("VERIFICATION PASSED")
else:
    print("VERIFICATION FAILED - check missing files above")
print("=" * 65)
