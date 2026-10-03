# 人臉表情分析器 (Face Expression Analyzer)

基於 Python + OpenCV + MediaPipe + yt-dlp 的人臉表情分析工具，可分析影片中人臉的嘴唇開合程度（MAR），並判斷三種狀態：FOCUSED / SPEAKING / DROWSY。

## 專案簡介

本專案透過 MediaPipe FaceMesh 偵測人臉網格，提取嘴唇 4 點（13、14、78、308）計算 MAR（Mouth Aspect Ratio），並根據閾值進行三態判斷。支援本機影片與 YouTube URL 作為輸入，輸出標註後的影片與統計報告。

## 環境安裝

使用 `uv` 安裝依賴（建議 Python 3.10+）：

```bash
uv pip install mediapipe==0.10.9 opencv-python numpy yt-dlp
```

> 注意：`mediapipe` 版本限制為 `0.10.9`，請勿安裝其他版本以避免相容性問題。

## 使用方式

### 1. 啟動 FastAPI Web 視覺化儀表板 (推薦，Port 3000)

直接執行 `main.py`，預設會在 `http://localhost:3000` 啟動 Web 服務並自動開啟瀏覽器：

```bash
uv run python main.py
```

或指定連接埠：

```bash
uv run python main.py --port 3000
```

Web 介面特色：
- **即時 MJPEG 影像串流**：以視覺化 HUD 面板顯示嘴唇 4 點與即時 MAR。
- **多元來源切換**：支援筆電/本機鏡頭 (0)、YouTube 網址或本機影片檔案。
- **即時儀表板**：顯示即時 MAR 量尺、狀態燈號、累計處理幀數、發言比例與疲倦次數。
- **報表與影片下載**：一鍵檢視 `report.txt` 並直接下載成果影片 `annotated.mp4`。

---

### 2. 命令列 (CLI) 批次模式

若需在無介面環境或以腳本自動化執行，可加上 `--cli`：

#### 本機影片
```bash
uv run python main.py --cli --source "video.mp4"
```

#### YouTube 串流
```bash
uv run python main.py --cli --source "https://www.youtube.com/watch?v=aVwzLrSAp78"
```

執行後按 `q` 可提前結束分析並產出報表。

## 輸出檔案說明

- `outputs/annotated.mp4`：標註後的影片（解析度 800x600），包含嘴唇 4 點標記、MAR 數值與狀態文字。
- `outputs/report.txt`：統計報告，包含時間戳、總幀數、SPEAKING 幀數與比例、DROWSY 觸發次數。

## 常見問題

### 無法讀取串流

若出現 `無法讀取影片串流`，請確認：
- YouTube URL 是否有效
- 網路連線正常
- `yt-dlp` 已正確安裝

可嘗試改用本機影片路徑測試：

```bash
python main.py --source "test.mp4"
```

### Python 版本不符

`mediapipe==0.10.9` 需要 Python 3.9 ~ 3.11。若版本不符，請切換 Python 版本或使用 `uv` 建立對應環境：

```bash
uv python install 3.11
uv venv --python 3.11
```
