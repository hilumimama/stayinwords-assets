# 玉山主峰兩天一夜 GPS 路線影片

塔塔加登山口 → 排雲山莊 → 玉山主峰 兩天一夜來回（2026.09.14–09.15）的
Relive 風格動畫路線影片。與 stayinwords 每日一字專案無關，獨立資料夾。

## 資料來源

- `raw-tracks/route_*.gpx` — Apple 健康 App 匯出的實際分段紀錄（5 段）
- `raw-tracks/guide_offline_map.gpx` — 嚮導提供的官方步道參考路線（健行筆記匯出），
  用來在紀錄有斷點的地方內插銜接

## 處理流程

1. `scripts/inspect_tracks.py` — 檢視各段 GPX 的時間範圍/座標/爬升
2. `scripts/merge_tracks.py` — 依時間排序拼接 5 段紀錄，偵測斷點（距離 > 30m），
   在參考步道上找最近點內插補上缺口，輸出 `output/yushan_merged.json` +
   `output/summary.json`（缺口清單見 summary 的 `gaps`）
3. `scripts/render_video.py` — 讀取合併後的路線，渲染動畫影片到
   `output/yushan_hike.mp4`

背景地形是用紀錄本身的海拔採樣做 hillshade 暈渲圖（這個環境連不到衛星圖磚伺服器，
無法疊加真實衛星空拍），內插的路段在影片上以虛線呈現，跟實際 GPS 紀錄做區別。

## 重新產生影片

```
python3 scripts/merge_tracks.py
python3 scripts/render_video.py
```
