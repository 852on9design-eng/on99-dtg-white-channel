# TIFF 匯出規格（PrintEXP）

## 點解仲會 Invalid image format

Hoson PrintExp 對 TIFF 好嚴。常見失敗原因：

1. **只有 ExtraSamples + ALPHA_NAMES**（當 Alpha，唔係 Spot）
2. **RGB + 4 samples** 被部分 PrintExp TIFF loader 拒收
3. **檔名有空格 / 括號**（例如 `vintage_print (12)_white.tif`）
4. **奇怪 ICC / 多餘 tag**

## 而家預設（PrintExp CMYK+Spot）

跟 Photoshop 印前專色 TIFF 接近：

- Photometric = **SEPARATED（CMYK）**
- Extra plane = Spot **`white`**
- ImageResources：DisplayInfo **kind=2**、Alternate Spot Colors、Spot Halftone
- **無 ICC**（減少 loader 拒收）
- 下載檔名自動清走空格/括號

## 其他模式

| 模式 | 用途 |
|------|------|
| PrintExp CMYK+Spot | 預設，最接近廠方可 Import TIFF |
| PrintExp RGB+Spot | 跟影片 RGB 文件模式 |
| Legacy ExtraSamples | 舊輸出對照（預期 PrintExp 會 Invalid） |

## PrintExp 設定

1. Import `.tif`
2. white Color → Data Source Type = **Spot**
3. Channel 1 = 白墨；無光油則 Channel 2 = None

## 本機測試

```bash
python3 test_tiff_export.py
```
