import gpxpy
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "raw-tracks"

def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp/2)**2 + cos(p1)*cos(p2)*sin(dl/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

for f in sorted(RAW.glob("*.gpx")):
    with open(f, encoding="utf-8") as fh:
        gpx = gpxpy.parse(fh)
    pts = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            pts.extend(seg.points)
    if not pts:
        print(f.name, "NO POINTS")
        continue
    times = [p.time for p in pts if p.time]
    eles = [p.elevation for p in pts if p.elevation is not None]
    dist = 0.0
    for a, b in zip(pts, pts[1:]):
        dist += haversine(a.latitude, a.longitude, b.latitude, b.longitude)
    lats = [p.latitude for p in pts]
    lons = [p.longitude for p in pts]
    name = gpx.tracks[0].name if gpx.tracks else None
    print(f"== {f.name} ==")
    print(f"  track name : {name}")
    print(f"  points     : {len(pts)}")
    if times:
        print(f"  time range : {min(times)}  ->  {max(times)}  (dur {max(times)-min(times)})")
    if eles:
        print(f"  elevation  : {min(eles):.0f} m -> {max(eles):.0f} m")
    print(f"  distance   : {dist/1000:.2f} km")
    print(f"  bbox       : lat[{min(lats):.5f},{max(lats):.5f}] lon[{min(lons):.5f},{max(lons):.5f}]")
    print(f"  start pt   : {pts[0].latitude:.5f},{pts[0].longitude:.5f}")
    print(f"  end pt     : {pts[-1].latitude:.5f},{pts[-1].longitude:.5f}")
    print()
