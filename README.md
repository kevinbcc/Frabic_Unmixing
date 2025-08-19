# 紗線高光譜辨識系統 API 文件

## 系統概述

本系統是一套針對紗線樣本進行高光譜分析的圖形化介面，支援 HDR+RAW 資料上傳、ROI 框選、SNV/PCA 等前處理、CNN 訓練、紗線成分比例預測與 RMSE 誤差可視化。

## 系統需求

- Python >= 3.10
- GPU: 例如 NVIDIA RTX 5070Ti (16GB VRAM)
- 記憶體需求: 至少 16 GB
- 建議解析度: 1920x1080 以上

---

## 環境安裝（使用 YML）

請使用 [Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) 建立獨立環境，確保所有套件與依賴一致，Anaconda 安裝```Anaconda3-2023.03-0-Windows-x86_64.exe```此版本。

### 安裝步驟：

1. 開啟Anaconda Powershell Prompt
2. 執行以下指令：

```bash
# 建立 conda 環境
conda env create -f environment.yml
```

```bash
# 啟用環境
conda activate FRABIC_Multi-Layer_Perceptron2 
```
---

## 執行程式

```bash
python main.py
```

---

## 專案結構

```
FABRIC_UNMIXING/
├── environment.yml
├── README.md
├── data/
├── data_preprocessing/
├── uploaded/
├── result/
│   └── sourcewise_rmse.csv
├── src/
│   ├── UI/
│   │  │── data_processing.py
│   │  ├── gui.py
│   │  ├── model.training.py
│   │  └── plotting.py
│   ├── config_loader.py
│   ├── generate_env_file.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
└── main.py  # 主程式(分析比例和RMSE誤差值)
```

---
## 模型訓練與分析端點

### 1. `upload_and_process_files()`

**功能**: 上傳 HDR+RAW 檔案並進行 ROI 框選及 SAM Map 分析，輸出為 `.npy` 檔案。

**輸入參數**: 使用者介面選擇檔案。  
**輸出資料夾**: `./data/*.npy`

---

### 2. `run_model_training(mode: str, band_num: int)`

**功能**: 對上傳的 `.npy` 檔案進行前處理、訓練 CNN 模型並輸出預測結果與 RMSE。

**輸入參數**:
- `mode`: 前處理模式，支援：
  - `SNV`
  - `PCA_BANDSELECT`
  - `SNV_PCABANDSELECT`
- `band_num`: 波段選擇數量，例如 10、20、50、100、224

**輸出**:
- `result/output.json`: 各紗線樣本預測比例與 RMSE
- `result/sourcewise_rmse.csv`: 每一樣本來源的 RMSE 統計

**JSON 輸出格式範例**:

```
{
  "COMPACT100C_RT_roi_25x500": {
    "avg_predicted": {
      "cotton": 99.93,
      "poly": 0.07
    },
    "rmse": {
      "cotton": 0.13,
      "poly": 0.13
    }
  }
}
```

---

### 3. `plot_spectra_in_gui(mode, band_num)`

**功能**: 顯示 `data` 中原始與處理後的平均光譜圖。

**輸入參數**:  
與 `run_model_training()` 相同的 `mode` 與 `band_num`。

**輸出**: 以 GUI 顯示光譜折線圖。

---

### 4. `display_ratio_analysis(mode_selector, band_num_selector)`

**功能**: 執行模型訓練並以圓餅圖與柱狀圖顯示比例與 RMSE 分析。

**輸入**: 由 GUI 選取 mode 與 band_num。

**輸出**:
- 各紗線類別的圓餅圖：顯示棉與聚酯比例。
- 各來源的 RMSE 柱狀圖。

## 資料夾說明

- `uploaded/`: 上傳 HDR/RAW 檔案暫存。
- `data/`: ROI 範圍轉換的原始光譜 `.npy`
- `preprocessing_data/`: 前處理後 `.npy` 檔案
- `result/`:
  - `output.json`: 模型預測的平均比例與 RMSE
  - `sourcewise_rmse.csv`: 每筆資料的 RMSE 統計

## 模型架構

使用 `SimpleCNN_MLP` 模型：
- CNN: 提取光譜局部特徵
- MLP: 預測 (cotton_ratio, poly_ratio)
- Loss: MSELoss
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

## 訓練樣本對應標籤

以下為 `.npy` 檔案與對應的棉/聚酯比例：

| 檔名                            | 棉比例 | 聚酯纖維比例 |
|--------------------------------|--------|----------|
| COMPACT100C_RT_roi_25x500.npy | 1.0    | 0.0      |
| COMPACT100P_RT_roi_25x500.npy | 0.0    | 1.0      |
| COMPACT5050_RT_roi_25x500.npy | 0.5    | 0.5      |
| MVS100C_RT_roi_25x500.npy     | 1.0    | 0.0      |
| MVS100P_RT_roi_25x500.npy     | 0.0    | 1.0      |
| MVS5050_RT_roi_25x500.npy     | 0.5    | 0.5      |
| OE100C_RT_roi_25x500.npy      | 1.0    | 0.0      |
| OE100P_RT_roi_25x500.npy      | 0.0    | 1.0      |
| OE5050_RT_roi_25x500.npy      | 0.5    | 0.5      |

---

