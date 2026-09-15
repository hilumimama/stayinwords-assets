"""
Merge the multi-segment Apple Health hiking GPX exports into one continuous
route for Tataka trailhead -> Paiyun Lodge -> Yushan Main Peak (round trip),
filling recording gaps by snapping to a reference trail GPX and interpolating
along it.

Output: output/yushan_merged.json — a list of points, each tagged
recorded=True/False, in chronological/trip order, ready for rendering.
"""
import json
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import gpxpy

RAW = Path(__file__).resolve().parent.parent / "raw-tracks"
OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)

REAL_SEGMENT_FILES = [
    "route_2026-09-14_10.48am.gpx",
    "route_2026-09-14_3.46pm.gpx",
    "route_2026-09-15_4.42am.gpx",
    "route_2026-09-15_7.44am.gpx",
    "route_2026-09-15_1.45pm.gpx",
]
REFERENCE_FILE = "guide_offline_map.gpx"

GAP_FILL_DISTANCE_THRESHOLD_M = 30  # below this, just connect directly


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def load_points(path):
    with open(path, encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    pts = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                pts.append({
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "ele": p.elevation,
                    "time": p.time,
                })
    return pts


def nearest_index(pt, ref_pts, start_idx):
    best_i, best_d = start_idx, float("inf")
    for i in range(start_idx, len(ref_pts)):
        d = haversine(pt["lat"], pt["lon"], ref_pts[i]["lat"], ref_pts[i]["lon"])
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def cumulative_distances(pts):
    dists = [0.0]
    for a, b in zip(pts, pts[1:]):
        dists.append(dists[-1] + haversine(a["lat"], a["lon"], b["lat"], b["lon"]))
    return dists


def build_interpolated_slice(ref_slice, t_start, t_end):
    """Assign synthetic timestamps proportional to cumulative distance."""
    if len(ref_slice) < 2:
        return []
    cum = cumulative_distances(ref_slice)
    total = cum[-1] or 1.0
    span = (t_end - t_start).total_seconds()
    out = []
    for p, c in zip(ref_slice, cum):
        frac = c / total
        t = t_start + timedelta(seconds=span * frac)
        out.append({"lat": p["lat"], "lon": p["lon"], "ele": p["ele"], "time": t, "recorded": False})
    return out


def main():
    segments = []
    for fname in REAL_SEGMENT_FILES:
        pts = load_points(RAW / fname)
        for p in pts:
            p["recorded"] = True
        segments.append(pts)
    # sort chronologically by first point time
    segments.sort(key=lambda s: s[0]["time"])

    ref_pts = load_points(RAW / REFERENCE_FILE)

    merged = []
    ref_cursor = 0
    gaps_report = []

    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = merged[-1]
            gap_start_pt = prev_end
            gap_end_pt = seg[0]
            dist = haversine(gap_start_pt["lat"], gap_start_pt["lon"], gap_end_pt["lat"], gap_end_pt["lon"])
            if dist > GAP_FILL_DISTANCE_THRESHOLD_M:
                idx_a, d_a = nearest_index(gap_start_pt, ref_pts, ref_cursor)
                idx_b, d_b = nearest_index(gap_end_pt, ref_pts, idx_a)
                ref_slice = ref_pts[idx_a:idx_b + 1]
                filled = build_interpolated_slice(ref_slice, gap_start_pt["time"], gap_end_pt["time"])
                merged.extend(filled)
                ref_cursor = idx_b
                gaps_report.append({
                    "after_segment": i - 1,
                    "gap_distance_m": round(dist, 1),
                    "time_start": gap_start_pt["time"].isoformat(),
                    "time_end": gap_end_pt["time"].isoformat(),
                    "filled_points": len(filled),
                    "snap_error_start_m": round(d_a, 1),
                    "snap_error_end_m": round(d_b, 1),
                })
            else:
                gaps_report.append({
                    "after_segment": i - 1,
                    "gap_distance_m": round(dist, 1),
                    "time_start": gap_start_pt["time"].isoformat(),
                    "time_end": gap_end_pt["time"].isoformat(),
                    "filled_points": 0,
                    "note": "direct connect, below threshold",
                })
        merged.extend(seg)

    # serialize
    def ser(p):
        return {"lat": p["lat"], "lon": p["lon"], "ele": p["ele"],
                "time": p["time"].isoformat() if p["time"] else None,
                "recorded": p["recorded"]}

    with open(OUT / "yushan_merged.json", "w", encoding="utf-8") as fh:
        json.dump([ser(p) for p in merged], fh, ensure_ascii=False)

    total_dist = cumulative_distances(merged)[-1]
    recorded_dist = cumulative_distances([p for p in merged if p["recorded"]])[-1]
    ele_gain = sum(max(0, b["ele"] - a["ele"]) for a, b in zip(merged, merged[1:]) if a["ele"] is not None and b["ele"] is not None)
    summary = {
        "total_points": len(merged),
        "total_distance_km": round(total_dist / 1000, 2),
        "interpolated_points": sum(1 for p in merged if not p["recorded"]),
        "elevation_gain_m": round(ele_gain, 1),
        "start_time": merged[0]["time"].isoformat(),
        "end_time": merged[-1]["time"].isoformat(),
        "max_elevation_m": round(max(p["ele"] for p in merged if p["ele"] is not None), 1),
        "gaps": gaps_report,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
