"""
Render a Relive-style animated route video from the merged Yushan track.
No live satellite tiles are available in this sandbox (egress is allow-listed
and blocks tile servers), so the background is a hillshaded terrain relief
built from the hike's own recorded/interpolated elevation samples.
"""
import json
from datetime import timedelta, datetime, timezone
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Noto Sans CJK TC"
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "output"
TZ = timezone(timedelta(hours=8))

W, H, DPI = 1080, 1920, 120
FIGSIZE = (W / DPI, H / DPI)

RESAMPLE_STEP_M = 8
FPS = 25
INTRO_SEC, MAIN_SEC, OUTRO_SEC = 2.5, 40, 3.5
ASPECT = W / H  # width/height of the frame
FOLLOW_HALF_HEIGHT_M = 1300  # camera zoom window while following the hiker
M_PER_DEG_LAT = 110540


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def load():
    with open(OUT / "yushan_merged.json", encoding="utf-8") as fh:
        pts = json.load(fh)
    for p in pts:
        p["time"] = datetime.fromisoformat(p["time"])
    with open(OUT / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    return pts, summary


def smooth(values, window=15):
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def resample(pts, step_m=RESAMPLE_STEP_M):
    lat = np.array([p["lat"] for p in pts])
    lon = np.array([p["lon"] for p in pts])
    ele = smooth([p["ele"] for p in pts])
    rec = np.array([p["recorded"] for p in pts])
    t = [p["time"] for p in pts]

    cum = np.zeros(len(pts))
    for i in range(1, len(pts)):
        cum[i] = cum[i - 1] + haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])

    keep = [0]
    last = 0.0
    for i in range(1, len(pts)):
        if cum[i] - last >= step_m or rec[i] != rec[i - 1]:
            keep.append(i)
            last = cum[i]
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    keep = np.array(keep)

    idx_summit = int(np.argmax(ele))
    if idx_summit not in keep:
        keep = np.sort(np.append(keep, idx_summit))

    return {
        "lat": lat[keep], "lon": lon[keep], "ele": ele[keep],
        "rec": rec[keep], "time": [t[i] for i in keep], "cum": cum[keep],
        "idx_summit": int(np.where(keep == idx_summit)[0][0]),
    }


def camera_windows(rs, mean_lat_rad):
    """Half-height/half-width (in degrees) for the follow camera and the
    overview camera, plus the overall bbox the terrain grid must cover."""
    m_per_deg_lon = 111320 * cos(mean_lat_rad)
    follow_half_h_deg = FOLLOW_HALF_HEIGHT_M / M_PER_DEG_LAT
    follow_half_w_deg = (FOLLOW_HALF_HEIGHT_M * ASPECT) / m_per_deg_lon

    route_w_m = (rs["lon"].max() - rs["lon"].min()) * m_per_deg_lon
    route_h_m = (rs["lat"].max() - rs["lat"].min()) * M_PER_DEG_LAT
    overview_half_h_m = max(route_h_m, route_w_m * ASPECT) / 2 * 1.25
    overview_half_h_deg = overview_half_h_m / M_PER_DEG_LAT
    overview_half_w_deg = (overview_half_h_m * ASPECT) / m_per_deg_lon
    overview_center = (rs["lon"].mean(), rs["lat"].mean())

    lon_min = min(rs["lon"].min() - follow_half_w_deg, overview_center[0] - overview_half_w_deg)
    lon_max = max(rs["lon"].max() + follow_half_w_deg, overview_center[0] + overview_half_w_deg)
    lat_min = min(rs["lat"].min() - follow_half_h_deg, overview_center[1] - overview_half_h_deg)
    lat_max = max(rs["lat"].max() + follow_half_h_deg, overview_center[1] + overview_half_h_deg)

    return {
        "follow_half_w": follow_half_w_deg, "follow_half_h": follow_half_h_deg,
        "overview_half_w": overview_half_w_deg, "overview_half_h": overview_half_h_deg,
        "overview_center": overview_center,
        "grid_bbox": (lon_min, lon_max, lat_min, lat_max),
    }


def build_terrain(pts, grid_bbox, grid_n=600, blur_sigma=14):
    lat = np.array([p["lat"] for p in pts][::3])
    lon = np.array([p["lon"] for p in pts][::3])
    ele = np.array([p["ele"] for p in pts][::3])

    lon_min, lon_max, lat_min, lat_max = grid_bbox
    mean_lat_rad = radians((lat_min + lat_max) / 2)
    aspect_ll = (lat_max - lat_min) / (lon_max - lon_min)
    nx = grid_n
    ny = max(2, int(round(grid_n * aspect_ll)))

    gx = np.linspace(lon_min, lon_max, nx)
    gy = np.linspace(lat_min, lat_max, ny)
    GX, GY = np.meshgrid(gx, gy)

    grid = griddata((lon, lat), ele, (GX, GY), method="linear")
    grid_nn = griddata((lon, lat), ele, (GX, GY), method="nearest")
    grid = np.where(np.isnan(grid), grid_nn, grid)
    grid = np.nan_to_num(grid, nan=float(np.nanmin(ele)))
    grid = gaussian_filter(grid, sigma=blur_sigma)

    terrain_cmap = LinearSegmentedColormap.from_list(
        "yushan_terrain",
        ["#1f3d2b", "#3c5a3a", "#6b7a4f", "#9a8f6b", "#c9c0a8", "#eeeee6"],
    )
    dx = (lon_max - lon_min) * 111320 * cos(mean_lat_rad) / nx
    dy = (lat_max - lat_min) * 110540 / ny
    ls = LightSource(azdeg=315, altdeg=55)
    rgb = ls.shade(grid, cmap=terrain_cmap, blend_mode="soft",
                    vert_exag=2.5, dx=dx, dy=dy)

    extent = (lon_min, lon_max, lat_min, lat_max)
    return rgb, extent, mean_lat_rad


def lonlat_to_xy(lon, lat):
    return lon, lat


def make_segments(lon, lat, rec):
    pts_xy = np.column_stack([lon, lat])
    segs = np.stack([pts_xy[:-1], pts_xy[1:]], axis=1)
    seg_rec = rec[:-1] & rec[1:]
    return segs, seg_rec


def fmt_hm(td):
    total_min = int(td.total_seconds() // 60)
    h, m = divmod(total_min, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


def main():
    pts, summary = load()
    rs = resample(pts)
    n = len(rs["lat"])
    print(f"resampled to {n} points")

    mean_lat_rad = radians(float(np.mean(rs["lat"])))
    cam = camera_windows(rs, mean_lat_rad)
    rgb, extent, _ = build_terrain(pts, cam["grid_bbox"])

    segs, seg_rec = make_segments(rs["lon"], rs["lat"], rs["rec"])

    start_time = rs["time"][0]
    total_dist_km = rs["cum"][-1] / 1000
    max_ele = rs["ele"].max()
    idx_summit = rs["idx_summit"]
    final_gains = np.diff(rs["ele"])
    final_gain = final_gains[final_gains > 0].sum()

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor("#0b0f0c")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0b0f0c")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / cos(mean_lat_rad))
    ax.axis("off")
    ax.imshow(rgb, extent=extent, origin="lower", interpolation="bilinear", zorder=0)

    lc_rec = LineCollection([], colors="#ffd23f", linewidths=3.2, zorder=3, capstyle="round")
    lc_int = LineCollection([], colors="#ffd23f", linewidths=2.4, linestyles=(0, (3, 3)), alpha=0.85, zorder=3)
    ax.add_collection(lc_rec)
    ax.add_collection(lc_int)
    marker, = ax.plot([], [], "o", color="white", markersize=9,
                       markeredgecolor="#ffd23f", markeredgewidth=2.5, zorder=4)

    # elevation profile strip (top)
    prof_ax = fig.add_axes([0.06, 0.855, 0.88, 0.09])
    prof_ax.set_facecolor("none")
    for s in prof_ax.spines.values():
        s.set_visible(False)
    prof_ax.set_xticks([]); prof_ax.set_yticks([])
    prof_ax.set_xlim(0, total_dist_km)
    prof_ax.set_ylim(rs["ele"].min() - 50, max_ele + 80)
    prof_fill = prof_ax.fill_between([0], [0], [0], color="white", alpha=0.85)
    prof_line, = prof_ax.plot([], [], color="white", linewidth=1.6)

    stats_text = fig.text(0.06, 0.815, "", color="white", fontsize=15,
                           fontweight="bold", va="top")
    bottom_title = fig.text(0.06, 0.045, "玉山主峰 兩天一夜", color="white",
                             fontsize=22, fontweight="bold", va="bottom")
    bottom_sub = fig.text(0.06, 0.02, "塔塔加登山口 → 排雲山莊 → 玉山主峰",
                           color="#d8d8d8", fontsize=12, va="bottom")

    intro_group = [stats_text, bottom_title, bottom_sub, prof_line, prof_fill]

    big_title = fig.text(0.5, 0.5, "", color="white", fontsize=30, ha="center",
                          va="center", fontweight="bold", zorder=10)
    big_sub = fig.text(0.5, 0.44, "", color="#ffd23f", fontsize=15, ha="center",
                        va="center", fontweight="bold", zorder=10)
    dim_overlay = fig.add_axes([0, 0, 1, 1], zorder=9)
    dim_overlay.axis("off")
    dim_overlay.set_facecolor("black")
    dim_overlay.patch.set_alpha(0.0)

    n_intro = int(INTRO_SEC * FPS)
    n_main = int(MAIN_SEC * FPS)
    n_outro = int(OUTRO_SEC * FPS)
    total_frames = n_intro + n_main + n_outro
    main_indices = np.linspace(0, n - 1, n_main).astype(int)

    prof_fill_holder = {"artist": None}

    def set_camera(cx, cy, half_w, half_h):
        ax.set_xlim(cx - half_w, cx + half_w)
        ax.set_ylim(cy - half_h, cy + half_h)

    def lerp(a, b, t):
        return a + (b - a) * t

    def set_progress(k):
        rk = seg_rec[:k]
        lc_rec.set_segments(segs[:k][rk]) if k > 0 else lc_rec.set_segments([])
        lc_int.set_segments(segs[:k][~rk]) if k > 0 else lc_int.set_segments([])
        if k > 0:
            marker.set_data([rs["lon"][k]], [rs["lat"][k]])
        d = rs["cum"][:k + 1] / 1000
        e = rs["ele"][:k + 1]
        prof_line.set_data(d, e)
        if prof_fill_holder["artist"] is not None:
            prof_fill_holder["artist"].remove()
        prof_fill_holder["artist"] = prof_ax.fill_between(
            d, e, rs["ele"].min() - 50, color="white", alpha=0.85)

        dist_km = rs["cum"][k] / 1000
        gains = np.diff(rs["ele"][:k + 1])
        gain = gains[gains > 0].sum() if k > 0 else 0
        elapsed = rs["time"][k] - start_time
        stats_text.set_text(f"距離 {dist_km:.1f} km    時間 {fmt_hm(elapsed)}    爬升 {gain:.0f} m")

    ocx, ocy = cam["overview_center"]
    ohw, ohh = cam["overview_half_w"], cam["overview_half_h"]
    fhw, fhh = cam["follow_half_w"], cam["follow_half_h"]
    CAM_BLEND_SEC = 1.0

    def draw_frame(i):
        if i < n_intro:
            dim_overlay.patch.set_alpha(0.55)
            for a in intro_group:
                a.set_alpha(0)
            set_progress(0)
            set_camera(ocx, ocy, ohw, ohh)
            big_title.set_text("玉山主峰\n兩天一夜")
            big_sub.set_text("塔塔加登山口 → 排雲山莊 → 玉山主峰\n2026.09.14 – 09.15")
            big_title.set_alpha(min(1, i / (n_intro * 0.4)))
            big_sub.set_alpha(min(1, i / (n_intro * 0.4)))
        elif i < n_intro + n_main:
            fade = min(1, (i - n_intro) / (FPS * 0.6))
            big_title.set_alpha(max(0, 1 - fade * 2))
            big_sub.set_alpha(max(0, 1 - fade * 2))
            dim_overlay.patch.set_alpha(max(0, 0.55 * (1 - fade * 2)))
            for a in intro_group:
                a.set_alpha(min(1, fade))
            k = main_indices[i - n_intro]
            set_progress(k)
            cx, cy = rs["lon"][k], rs["lat"][k]
            blend_frames = CAM_BLEND_SEC * FPS
            t_in = min(1, (i - n_intro) / blend_frames)
            t_out = min(1, (n_intro + n_main - i) / blend_frames)
            t = min(t_in, t_out)
            cam_cx = lerp(ocx, cx, t)
            cam_cy = lerp(ocy, cy, t)
            cam_hw = lerp(ohw, fhw, t)
            cam_hh = lerp(ohh, fhh, t)
            set_camera(cam_cx, cam_cy, cam_hw, cam_hh)
        else:
            set_progress(n - 1)
            set_camera(ocx, ocy, ohw, ohh)
            reveal = min(1, (i - n_intro - n_main) / (n_outro * 0.5))
            for a in intro_group:
                a.set_alpha(max(0, 1 - reveal * 1.5))
            dim_overlay.patch.set_alpha(min(0.6, reveal))
            big_title.set_text(f"{summary['total_distance_km']} km")
            big_sub.set_text(f"爬升 {final_gain:.0f} m   ·   最高 {max_ele:.0f} m")
            big_title.set_alpha(reveal)
            big_sub.set_alpha(reveal)
        return [lc_rec, lc_int, marker, prof_line, stats_text, big_title, big_sub]

    anim = FuncAnimation(fig, draw_frame, frames=total_frames, blit=False)
    out_path = OUT / "yushan_hike.mp4"
    anim.save(out_path, fps=FPS, dpi=DPI,
               savefig_kwargs={"facecolor": fig.get_facecolor()},
               extra_args=["-pix_fmt", "yuv420p"])
    print("saved", out_path)


if __name__ == "__main__":
    main()
