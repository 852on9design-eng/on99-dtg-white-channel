# TIFF 匯出規格（PrintEXP Spot）

## 舊檔為何失敗

Hoson **PrintExp** Spot 模式要嘅係 Photoshop **Spot Color Channel**。

舊版白墨 App 用 `tifffile` 寫：

- `photometric=rgb` + `ExtraSamples=UNSPECIFIED`（第 4 sample）
- ImageResources 只有 `ALPHA_NAMES`（Pascal / Unicode）

呢啲係 **Alpha / ExtraSamples 命名**，唔係 Spot Color。PrintExp Import 會報：

`Invalid image format`

廠方流程：Photoshop → **New Spot Channel** 名叫 `white` → Save As TIFF 勾 **Spot Colors** → PrintExp Import（white Color → Data Source Type = **Spot**）。

## 新檔結構差異

| 項目 | Legacy ExtraSamples（舊） | PrintExp Spot（新預設） |
|------|---------------------------|-------------------------|
| 像素 | RGB + 1 extra plane | 相同（仍係 extra sample 存像素） |
| 通道名資源 | 只有 ALPHA_NAMES | ALPHA_NAMES + Spot 標記 |
| DisplayInfo 1007/1077 | 無 | **kind = 2（Spot）** |
| Alternate Spot Colors 1067 | 無 | 有 |
| Spot Halftone 1043 | 無 | 有 |
| Alpha Identifiers | 無 / 0 | 非 0（唔當 transparency） |
| Software | tifffile.py | Adobe Photoshop … |
| 預設通道名 | white / W1（只係字串） | **`white`**（Spot） |

重點：像素一樣用 ExtraSamples 存放；**差在 ImageResources 令 Photoshop / PrintExp 認成 Spot Color**。

## 匯出模式（UI）

1. **PrintExp Spot（預設）** — `tiff_export.write_tiff_with_spot(..., mode="printexp_spot")`
2. **Legacy ExtraSamples** — 舊輸出，方便對照（PrintExp Spot 仍會 Invalid）

可選第二 Spot：`varnish`（或 Maintop 風格 `W2`）。

## PrintExp 設定對應

1. Import 新 TIFF  
2. Spot channel / Spot Color Setting  
3. **white Color → Data Source Type = Spot**  
4. Channel 1 = 白墨；Channel 2 無光油可設 None  

## 限制

- 無法喺呢個雲端環境實機跑 PrintExp；以 ImageResources 結構 + 單元測試驗證 Spot 標記。
- 最終驗收仍要以 Photoshop Channels 見到 Spot `white`，以及 PrintExp Import 成功為準。
- 壓縮建議：**無壓縮**（最穩）；亦可 ZIP / LZW。
- DPI：保留來源或 300。

## 本機測試

```bash
python test_tiff_export.py
```
