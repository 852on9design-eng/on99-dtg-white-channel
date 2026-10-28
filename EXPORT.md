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

### 外框全白／印出來相反

App 預設會 **反相寫入 Spot**（介面預覽仍係白=噴白）。  
若 PrintEXP 外四方一圈白、或印出嚟同預覽剛相反，保持呢個選項開啟再重新匯出。

### 左右鏡像

DTG 直噴若衫左右相反：打開「水平鏡像」再下載。  
DTF 轉印視乎 PrintEXP 自己有冇鏡像，唔好雙重鏡像。

## 本機測試

```bash
python3 test_tiff_export.py
```
