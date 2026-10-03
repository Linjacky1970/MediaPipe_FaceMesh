"""MediaPipe Face Mesh 專注度與嘴型動態分析系統 - FastAPI Web 服務 (main.py).

功能包含：
1. FastAPI Web 應用程式 (預設啟動於 Port 3000)。
2. Web 即時 MJPEG 影像串流輸出 (/video_feed)。
3. 即時遙測數據 API (/api/status, /api/start, /api/stop, /api/report)。
4. 現代化科技儀表板 Web 介面 (Glassmorphism HUD 風格)。
5. 支援 WebCam、YouTube 串流、本機影片檔案分析。
6. 支援檔案上傳分析與成果影片/報表直接下載。
7. 自動於瀏覽器開啟 http://localhost:3000。
8. 保留 --cli 命令列批次執行模式。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import yt_dlp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

# 輸出路徑配置
OUTPUTS_DIR = Path("outputs").resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_VIDEO_OUTPUT = OUTPUTS_DIR / "annotated.mp4"
DEFAULT_REPORT_OUTPUT = OUTPUTS_DIR / "report.txt"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def resolve_video_source(source: str) -> Tuple[str, bool]:
    """判斷並解析影片來源，若為 YouTube 則透過 yt-dlp 取得串流網址."""
    is_youtube = any(source.lower().startswith(prefix) for prefix in ("http://", "https://", "www.youtube.", "youtu.be"))
    if not is_youtube:
        if source.isdigit():
            return source, False
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"找不到本機影片檔案：{source_path.resolve()}")
        return str(source_path.resolve()), False

    ydl_opts = {
        "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)
        stream_url = info.get("url")
        if not stream_url:
            raise ValueError("yt-dlp 未能成功提取串流 URL")
        return stream_url, True


class VideoAnalyzer:
    """即時 Face Mesh 與 MAR 分析管理核心."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None

        # 遙測數據
        self.source_name = "待命中"
        self.current_mar = 0.0
        self.current_state = "IDLE"
        self.status_text = "STATUS: IDLE"
        self.status_color = (128, 128, 128)
        self.total_frames = 0
        self.speaking_frames = 0
        self.focused_frames = 0
        self.drowsy_frames = 0
        self.no_face_frames = 0
        self.drowsy_event_count = 0
        self.fps = 25.0
        self.elapsed_time = 0.0
        self.start_timestamp: Optional[float] = None
        self.error_message: Optional[str] = None

        # 串流影像暫存 (JPEG bytes)
        self.latest_jpeg: Optional[bytes] = None

    def get_status(self) -> Dict[str, Any]:
        """取得當前執行狀態與遙測指標."""
        with self.lock:
            total = max(1, self.total_frames) if self.is_running or self.total_frames > 0 else 1
            speaking_pct = round(self.speaking_frames / total * 100, 1) if self.total_frames > 0 else 0.0
            focused_pct = round(self.focused_frames / total * 100, 1) if self.total_frames > 0 else 0.0
            drowsy_pct = round(self.drowsy_frames / total * 100, 1) if self.total_frames > 0 else 0.0
            elapsed = time.time() - self.start_timestamp if self.is_running and self.start_timestamp else self.elapsed_time

            return {
                "is_running": self.is_running,
                "source": self.source_name,
                "mar": round(self.current_mar, 3),
                "state": self.current_state,
                "status_text": self.status_text,
                "status_color_bgr": self.status_color,
                "total_frames": self.total_frames,
                "speaking_frames": self.speaking_frames,
                "speaking_pct": speaking_pct,
                "focused_frames": self.focused_frames,
                "focused_pct": focused_pct,
                "drowsy_frames": self.drowsy_frames,
                "drowsy_pct": drowsy_pct,
                "drowsy_event_count": self.drowsy_event_count,
                "fps": round(self.fps, 1),
                "elapsed_time": round(elapsed, 1),
                "error": self.error_message,
                "has_report": DEFAULT_REPORT_OUTPUT.exists(),
                "has_video": DEFAULT_VIDEO_OUTPUT.exists(),
            }

    def start(
        self,
        source: str,
        threshold_open: float = 0.15,
        threshold_yawn: float = 0.30,
        yawn_duration: float = 2.0,
        record_video: bool = True,
    ) -> None:
        """啟動影像分析執行緒."""
        if self.is_running:
            self.stop()

        self.is_running = True
        self.error_message = None
        self.source_name = source
        self.worker_thread = threading.Thread(
            target=self._run_analysis_loop,
            args=(source, threshold_open, threshold_yawn, yawn_duration, record_video),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        """停止影像分析."""
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        self.worker_thread = None

    def _run_analysis_loop(
        self,
        source: str,
        threshold_open: float,
        threshold_yawn: float,
        yawn_duration: float,
        record_video: bool,
    ) -> None:
        """背景影像讀取與 MediaPipe 運算主迴圈."""
        cap = None
        out = None
        face_mesh = None

        try:
            stream_url, is_stream = resolve_video_source(source)
            source_arg = int(stream_url) if stream_url.isdigit() else stream_url

            if isinstance(source_arg, str) and source_arg.startswith("http"):
                cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(source_arg)
            else:
                cap = cv2.VideoCapture(source_arg)

            if not cap.isOpened():
                raise RuntimeError(f"無法開啟影像來源：{source}")

            frame_w = 854
            frame_h = 480
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = int(raw_fps) if raw_fps > 0 else 25
            self.fps = float(fps)

            # 初始化輸出影片
            if record_video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(DEFAULT_VIDEO_OUTPUT), fourcc, fps, (frame_w, frame_h))

            # 初始化 MediaPipe Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            yawn_limit = max(1, int(fps * yawn_duration))
            yawn_counter = 0

            with self.lock:
                self.total_frames = 0
                self.speaking_frames = 0
                self.focused_frames = 0
                self.drowsy_frames = 0
                self.no_face_frames = 0
                self.drowsy_event_count = 0
                self.start_timestamp = time.time()
                self.current_state = "RUNNING"

            was_drowsy = False

            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.total_frames += 1
                frame = cv2.resize(frame, (frame_w, frame_h))
                h, w, _ = frame.shape

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                status_text = "STATUS: NO FACE DETECTED"
                status_color = (128, 128, 128)
                current_state = "NO_FACE"
                mar = 0.0

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        pts = face_landmarks.landmark
                        p13 = np.array([pts[13].x * w, pts[13].y * h])
                        p14 = np.array([pts[14].x * w, pts[14].y * h])
                        p78 = np.array([pts[78].x * w, pts[78].y * h])
                        p308 = np.array([pts[308].x * w, pts[308].y * h])

                        dist_v = float(np.linalg.norm(p13 - p14))
                        dist_h = float(np.linalg.norm(p78 - p308))
                        mar = dist_v / dist_h if dist_h > 1e-6 else 0.0

                        # 繪製嘴唇追蹤點
                        for pt in (p13, p14, p78, p308):
                            cv2.circle(frame, tuple(pt.astype(int)), 3, (255, 255, 0), -1)

                        # 三態判定
                        if mar > threshold_yawn:
                            yawn_counter += 1
                            if yawn_counter >= yawn_limit:
                                current_state = "DROWSY"
                                status_text = "WARNING: DROWSY / YAWNING"
                                status_color = (0, 0, 255)
                            else:
                                current_state = "SPEAKING"
                                status_text = "STATUS: SPEAKING (LARGE)"
                                status_color = (255, 0, 0)
                        elif mar > threshold_open:
                            yawn_counter = 0
                            current_state = "SPEAKING"
                            status_text = "STATUS: SPEAKING"
                            status_color = (255, 0, 0)
                        else:
                            yawn_counter = max(0, yawn_counter - 1)
                            current_state = "FOCUSED"
                            status_text = "STATUS: FOCUSED / SILENT"
                            status_color = (0, 255, 0)

                        cv2.putText(
                            frame,
                            f"MAR: {mar:.2f}",
                            (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    yawn_counter = max(0, yawn_counter - 1)

                # 更新內部計數
                with self.lock:
                    self.current_mar = mar
                    self.current_state = current_state
                    self.status_text = status_text
                    self.status_color = status_color

                    if current_state == "SPEAKING":
                        self.speaking_frames += 1
                    elif current_state == "DROWSY":
                        self.drowsy_frames += 1
                    elif current_state == "FOCUSED":
                        self.focused_frames += 1
                    else:
                        self.no_face_frames += 1

                    is_drowsy_now = (current_state == "DROWSY")
                    if is_drowsy_now and not was_drowsy:
                        self.drowsy_event_count += 1
                    was_drowsy = is_drowsy_now

                # 繪製半透明左上角 HUD 控制面板
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (410, 100), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.rectangle(frame, (10, 10), (410, 100), status_color, 2)
                cv2.putText(
                    frame,
                    "AI CLASSROOM ANALYZER v1.0",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    status_text,
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                if out and out.isOpened():
                    out.write(frame)

                # 壓縮 JPEG 供 Web 即時串流
                ret_enc, enc_jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret_enc:
                    with self.lock:
                        self.latest_jpeg = enc_jpeg.tobytes()

                # 控制處理速度與讓出 CPU
                time.sleep(max(0.001, 1.0 / (fps * 1.5)))

        except Exception as exc:
            with self.lock:
                self.error_message = str(exc)
                self.status_text = f"ERROR: {exc}"
                self.current_state = "ERROR"
            print(f"[VideoAnalyzer Error] {exc}", file=sys.stderr)
        finally:
            if cap:
                cap.release()
            if out and out.isOpened():
                out.release()
            if face_mesh:
                face_mesh.close()

            with self.lock:
                self.is_running = False
                if self.start_timestamp:
                    self.elapsed_time = time.time() - self.start_timestamp
                self._save_report()

    def _save_report(self) -> None:
        """儲存統計報表至 outputs/report.txt."""
        total = self.total_frames
        speaking_pct = (self.speaking_frames / total * 100) if total > 0 else 0.0
        focused_pct = (self.focused_frames / total * 100) if total > 0 else 0.0
        drowsy_pct = (self.drowsy_frames / total * 100) if total > 0 else 0.0
        video_sec = (total / self.fps) if self.fps > 0 else 0.0

        report_lines = [
            "==================================================",
            "          MediaPipe Face Mesh 分析統計報表         ",
            "==================================================",
            f"產生時間      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"輸入來源      : {self.source_name}",
            f"輸出影片路徑  : {DEFAULT_VIDEO_OUTPUT}",
            f"分析總時長    : {video_sec:.2f} 秒 (實際運算耗時: {self.elapsed_time:.2f} 秒)",
            f"處理總幀數    : {total} 幀 (@ {self.fps:.1f} FPS)",
            "--------------------------------------------------",
            "【狀態統計】",
            f"  - 專注/安靜 (FOCUSED)  : {self.focused_frames} 幀 ({focused_pct:.1f}%)",
            f"  - 發言狀態  (SPEAKING) : {self.speaking_frames} 幀 ({speaking_pct:.1f}%)",
            f"  - 疲倦警示  (DROWSY)   : {self.drowsy_frames} 幀 ({drowsy_pct:.1f}%)",
            f"  - 無人臉偵測 (NO FACE) : {self.no_face_frames} 幀",
            "--------------------------------------------------",
            "【事件計數】",
            f"  - 觸發打哈欠警示 (DROWSY) 次數 : {self.drowsy_event_count} 次",
            "==================================================",
        ]
        DEFAULT_REPORT_OUTPUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


# 實體化分析器
analyzer = VideoAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式啟動與關閉生命週期管理."""
    print("FastAPI MediaPipe Face Mesh 服務啟動！運行於 Port 3000。")
    # 自動於 1 秒後開啟瀏覽器
    def open_browser():
        time.sleep(1.0)
        webbrowser.open("http://localhost:3000")

    threading.Thread(target=open_browser, daemon=True).start()
    yield
    analyzer.stop()


app = FastAPI(
    title="MediaPipe Face Mesh 專注度與發言分析系統",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    source: str = "0"
    threshold_open: float = 0.15
    threshold_yawn: float = 0.30
    yawn_duration: float = 2.0
    record_video: bool = True


@app.get("/", response_class=HTMLResponse)
def index_page() -> str:
    """現代化科技儀表板 Web 介面."""
    return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MediaPipe Face Mesh 專注度與發言動態分析</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang TC", "Microsoft JhengHei", sans-serif; }
    body { background-color: var(--bg); color: var(--text-main); min-height: 100vh; display: flex; flex-direction: column; }
    header { background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px); border-bottom: 1px solid var(--card-border); padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 50; }
    .header-title { display: flex; align-items: center; gap: 12px; font-size: 1.15rem; font-weight: 700; letter-spacing: 0.5px; }
    .header-title span { color: var(--primary); }
    .badge { padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; background: #374151; color: #d1d5db; transition: all 0.3s; }
    .badge.active { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge.danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }

    main { max-width: 1440px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 1.8fr 1fr; gap: 24px; flex: 1; width: 100%; }
    @media (max-width: 1024px) { main { grid-template-columns: 1fr; } }

    .card { background: var(--card-bg); backdrop-filter: blur(16px); border: 1px solid var(--card-border); border-radius: 16px; padding: 20px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4); display: flex; flex-direction: column; gap: 16px; }
    .card-title { font-size: 1rem; font-weight: 600; color: #e5e7eb; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid var(--card-border); padding-bottom: 10px; }

    .video-viewport { position: relative; width: 100%; aspect-ratio: 16/9; background: #000; border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.05); }
    .video-viewport img { width: 100%; height: 100%; object-fit: contain; }
    .video-placeholder { position: absolute; text-align: center; color: var(--text-muted); font-size: 0.95rem; }

    .control-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-group.full { grid-column: span 2; }
    label { font-size: 0.82rem; color: var(--text-muted); font-weight: 500; }
    input, select { background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); border-radius: 8px; padding: 9px 12px; color: #fff; font-size: 0.88rem; outline: none; transition: border-color 0.2s; }
    input:focus, select:focus { border-color: var(--primary); }

    .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .preset-btn { background: rgba(255, 255, 255, 0.05); border: 1px solid var(--card-border); border-radius: 6px; padding: 4px 9px; font-size: 0.75rem; color: #d1d5db; cursor: pointer; transition: all 0.2s; }
    .preset-btn:hover { background: rgba(59, 130, 246, 0.2); border-color: var(--primary); color: #fff; }

    .btn-row { display: flex; gap: 10px; margin-top: 8px; }
    .btn { flex: 1; padding: 11px 18px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; cursor: pointer; border: none; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn-primary { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4); }
    .btn-primary:hover { background: linear-gradient(135deg, #3b82f6, #2563eb); }
    .btn-danger { background: linear-gradient(135deg, #dc2626, #b91c1c); color: white; box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4); }
    .btn-danger:hover { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    /* 遙測指標卡片 */
    .metric-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .metric-card { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 10px; padding: 14px; display: flex; flex-direction: column; gap: 4px; }
    .metric-card.highlight { border-color: var(--primary); }
    .metric-label { font-size: 0.78rem; color: var(--text-muted); }
    .metric-value { font-size: 1.45rem; font-weight: 700; color: #fff; }

    /* 動態狀態燈號卡 */
    .status-banner { border-radius: 10px; padding: 14px; text-align: center; font-weight: 700; font-size: 1.05rem; letter-spacing: 0.5px; border: 2px solid #4b5563; background: rgba(31, 41, 55, 0.5); transition: all 0.3s; }
    .status-banner.focused { border-color: var(--success); color: #34d399; background: rgba(16, 185, 129, 0.15); box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
    .status-banner.speaking { border-color: var(--primary); color: #60a5fa; background: rgba(59, 130, 246, 0.15); box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
    .status-banner.drowsy { border-color: var(--danger); color: #f87171; background: rgba(239, 68, 68, 0.2); box-shadow: 0 0 25px rgba(239, 68, 68, 0.4); animation: pulse 1s infinite alternate; }
    @keyframes pulse { from { opacity: 0.8; } to { opacity: 1; } }

    /* MAR 量尺動態儀 */
    .gauge-container { display: flex; flex-direction: column; gap: 6px; }
    .gauge-bar-bg { width: 100%; height: 16px; background: #1e293b; border-radius: 999px; overflow: hidden; position: relative; }
    .gauge-bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #10b981, #3b82f6, #ef4444); transition: width 0.15s ease-out; }
    .gauge-marks { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text-muted); }

    /* 報表查看視窗 */
    .report-box { background: rgba(0, 0, 0, 0.5); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.82rem; color: #a7f3d0; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
    .download-links { display: flex; gap: 10px; }
    .link-btn { text-decoration: none; padding: 7px 12px; background: rgba(255,255,255,0.06); border: 1px solid var(--card-border); border-radius: 6px; font-size: 0.8rem; color: #cbd5e1; transition: all 0.2s; }
    .link-btn:hover { background: rgba(59, 130, 246, 0.2); color: #fff; border-color: var(--primary); }
  </style>
</head>
<body>
  <header>
    <div class="header-title">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/><circle cx="9" cy="9" r="1.5"/><circle cx="15" cy="9" r="1.5"/><path d="M8 15s1.5 2 4 2 4-2 4-2"/></svg>
      MediaPipe Face Mesh <span>表情與專注度動態分析</span>
    </div>
    <div id="service-badge" class="badge">待命狀態</div>
  </header>

  <main>
    <!-- 左欄：影像串流與控制 -->
    <div class="card">
      <div class="card-title">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="7" x2="7" y2="7"/><line x1="2" y1="17" x2="7" y2="17"/><line x1="17" y1="17" x2="22" y2="17"/><line x1="17" y1="7" x2="22" y2="7"/></svg>
        即時影像辨識 HUD 串流
      </div>

      <div class="video-viewport">
        <div id="video-placeholder" class="video-placeholder">
          <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 6px;">尚未啟動分析串流</p>
          <p>請於下方選擇來源後點擊「開始分析」</p>
        </div>
        <img id="stream-img" src="" alt="Video Feed" style="display: none;">
      </div>

      <!-- 控制面板 -->
      <div class="control-grid">
        <div class="form-group full">
          <label>輸入來源 (YouTube 網址 / 本機攝影機 0 / 影片路徑)</label>
          <input type="text" id="source-input" value="https://www.youtube.com/watch?v=aVwzLrSAp78" placeholder="輸入 URL 或 0">
          <div class="presets">
            <span style="font-size: 0.72rem; color: var(--text-muted); align-self: center;">快速選單：</span>
            <button class="preset-btn" onclick="setSource('0')">📷 筆電/本機鏡頭 (0)</button>
            <button class="preset-btn" onclick="setSource('https://www.youtube.com/watch?v=aVwzLrSAp78')">🎥 教學影片 (aVwzLrSAp78)</button>
            <button class="preset-btn" onclick="setSource('https://www.youtube.com/watch?v=jKrt3aaQLJk')">🎥 主播影片 (jKrt3aaQLJk)</button>
          </div>
        </div>

        <div class="form-group">
          <label>發言閾值 (MAR Open)</label>
          <input type="number" id="thresh-open" value="0.15" step="0.01" min="0.05" max="0.5">
        </div>
        <div class="form-group">
          <label>打哈欠閾值 (MAR Yawn)</label>
          <input type="number" id="thresh-yawn" value="0.30" step="0.01" min="0.1" max="0.8">
        </div>

        <div class="form-group">
          <label>打哈欠持續秒數門檻</label>
          <input type="number" id="yawn-duration" value="2.0" step="0.5" min="0.5" max="5.0">
        </div>
        <div class="form-group">
          <label>匯出成果影片</label>
          <select id="record-opt">
            <option value="true">啟用 (outputs/annotated.mp4)</option>
            <option value="false">關閉 (提升速度)</option>
          </select>
        </div>

        <div class="form-group full">
          <div class="btn-row">
            <button id="start-btn" class="btn btn-primary" onclick="startAnalysis()">
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              開始分析
            </button>
            <button id="stop-btn" class="btn btn-danger" onclick="stopAnalysis()" disabled>
              <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12"/></svg>
              停止分析並產出報表
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 右欄：數據遙測與診斷報表 -->
    <div style="display: flex; flex-direction: column; gap: 24px;">
      <!-- 即時狀態卡 -->
      <div class="card">
        <div class="card-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
          即時時序狀態判定
        </div>

        <div id="status-banner" class="status-banner">
          STATUS: IDLE
        </div>

        <div class="gauge-container">
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
            <span>即時嘴型高寬比 (MAR)</span>
            <strong id="mar-val" style="color: #60a5fa;">0.00</strong>
          </div>
          <div class="gauge-bar-bg">
            <div id="gauge-fill" class="gauge-bar-fill"></div>
          </div>
          <div class="gauge-marks">
            <span>0.0 (閉合)</span>
            <span>0.15 (發言門檻)</span>
            <span>0.30 (哈欠門檻)</span>
            <span>0.50+</span>
          </div>
        </div>
      </div>

      <!-- 統計數據指標 -->
      <div class="card">
        <div class="card-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
          累計分析統計
        </div>

        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">處理總幀數 (FPS)</div>
            <div class="metric-value" id="val-frames">0</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">已運行秒數</div>
            <div class="metric-value" id="val-time">0.0s</div>
          </div>
          <div class="metric-card highlight">
            <div class="metric-label">發言時長比例 (Speaking)</div>
            <div class="metric-value" id="val-speaking-pct" style="color: #60a5fa;">0.0%</div>
          </div>
          <div class="metric-card highlight">
            <div class="metric-label">打哈欠/疲倦次數 (Drowsy)</div>
            <div class="metric-value" id="val-drowsy-cnt" style="color: #f87171;">0</div>
          </div>
        </div>
      </div>

      <!-- 成果報表查看 -->
      <div class="card">
        <div class="card-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          分析成果報表 (report.txt)
        </div>
        <div id="report-content" class="report-box">尚無報表資料。停止分析後將自動於此呈現。</div>
        <div class="download-links">
          <a href="/outputs/report.txt" target="_blank" class="link-btn">下載 report.txt</a>
          <a href="/outputs/annotated.mp4" target="_blank" class="link-btn">下載 annotated.mp4</a>
        </div>
      </div>
    </div>
  </main>

  <script>
    const streamImg = document.getElementById('stream-img');
    const videoPlaceholder = document.getElementById('video-placeholder');
    const serviceBadge = document.getElementById('service-badge');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusBanner = document.getElementById('status-banner');
    const marVal = document.getElementById('mar-val');
    const gaugeFill = document.getElementById('gauge-fill');
    const reportBox = document.getElementById('report-content');

    function setSource(val) {
      document.getElementById('source-input').value = val;
    }

    async function startAnalysis() {
      const source = document.getElementById('source-input').value.trim();
      const openThresh = parseFloat(document.getElementById('thresh-open').value) || 0.15;
      const yawnThresh = parseFloat(document.getElementById('thresh-yawn').value) || 0.30;
      const yawnDur = parseFloat(document.getElementById('yawn-duration').value) || 2.0;
      const record = document.getElementById('record-opt').value === 'true';

      startBtn.disabled = true;
      startBtn.innerText = '連線啟動中...';

      try {
        const res = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source: source,
            threshold_open: openThresh,
            threshold_yawn: yawnThresh,
            yawn_duration: yawnDur,
            record_video: record
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || '啟動失敗');

        // 開啟影像串流
        streamImg.src = '/video_feed?t=' + Date.now();
        streamImg.style.display = 'block';
        videoPlaceholder.style.display = 'none';

        stopBtn.disabled = false;
        startBtn.innerText = '分析進行中';
      } catch (err) {
        alert('啟動錯誤：' + err.message);
        startBtn.disabled = false;
        startBtn.innerText = '開始分析';
      }
    }

    async function stopAnalysis() {
      stopBtn.disabled = true;
      stopBtn.innerText = '正在結算報表...';

      try {
        await fetch('/api/stop', { method: 'POST' });
        streamImg.style.display = 'none';
        videoPlaceholder.style.display = 'block';
        videoPlaceholder.innerHTML = '<p style="color:#34d399; font-weight:600;">分析已完成！</p><p>請查看右側報表或下載成果影片</p>';
        startBtn.disabled = false;
        startBtn.innerText = '開始分析';
        stopBtn.innerText = '停止分析並產出報表';
        fetchReport();
      } catch (err) {
        console.error(err);
      }
    }

    async function fetchReport() {
      try {
        const res = await fetch('/api/report');
        if (res.ok) {
          const txt = await res.text();
          reportBox.innerText = txt;
        }
      } catch (_) {}
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const s = await res.json();

        // 徽章與按鈕狀態同步
        if (s.is_running) {
          serviceBadge.className = 'badge active';
          serviceBadge.innerText = '即時分析中 (' + s.fps + ' FPS)';
          startBtn.disabled = true;
          stopBtn.disabled = false;
        } else {
          serviceBadge.className = 'badge';
          serviceBadge.innerText = '待命狀態';
        }

        // MAR 與量尺
        marVal.innerText = s.mar.toFixed(2);
        const pct = Math.min(100, Math.max(0, (s.mar / 0.50) * 100));
        gaugeFill.style.width = pct + '%';

        // 狀態外觀
        statusBanner.innerText = s.status_text;
        statusBanner.className = 'status-banner';
        if (s.state === 'FOCUSED') statusBanner.classList.add('focused');
        else if (s.state === 'SPEAKING') statusBanner.classList.add('speaking');
        else if (s.state === 'DROWSY') statusBanner.classList.add('drowsy');

        // 統計數字
        document.getElementById('val-frames').innerText = s.total_frames + ' (' + s.fps + ' FPS)';
        document.getElementById('val-time').innerText = s.elapsed_time + 's';
        document.getElementById('val-speaking-pct').innerText = s.speaking_pct + '% (' + s.speaking_frames + ' 幀)';
        document.getElementById('val-drowsy-cnt').innerText = s.drowsy_event_count + ' 次';

      } catch (e) {
        console.error(e);
      }
    }

    setInterval(pollStatus, 300);
    fetchReport();
  </script>
</body>
</html>
"""


def generate_mjpeg_stream():
    """MJPEG 影像串流生成器."""
    blank_frame = np.zeros((480, 854, 3), dtype=np.uint8)
    cv2.putText(
        blank_frame,
        "Waiting for video stream...",
        (220, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (150, 150, 150),
        2,
    )
    ret, blank_jpeg = cv2.imencode(".jpg", blank_frame)
    blank_bytes = blank_jpeg.tobytes() if ret else b""

    while True:
        with analyzer.lock:
            frame_bytes = analyzer.latest_jpeg if analyzer.is_running and analyzer.latest_jpeg else blank_bytes

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.04)


@app.get("/video_feed")
def video_feed():
    """提供即時影像 MJPEG 串流端點."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/start")
def api_start(req: StartRequest):
    """啟動分析 API."""
    try:
        analyzer.start(
            source=req.source,
            threshold_open=req.threshold_open,
            threshold_yawn=req.threshold_yawn,
            yawn_duration=req.yawn_duration,
            record_video=req.record_video,
        )
        return {"status": "ok", "message": f"已啟動來源 {req.source} 的分析"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/stop")
def api_stop():
    """停止分析 API."""
    analyzer.stop()
    return {"status": "ok", "message": "分析已停止並結算"}


@app.get("/api/status")
def api_status():
    """取得分析狀態與指標 API."""
    return analyzer.get_status()


@app.get("/api/report")
def api_report():
    """讀取產出之 report.txt."""
    if not DEFAULT_REPORT_OUTPUT.exists():
        raise HTTPException(status_code=404, detail="尚未產生報表")
    return HTMLResponse(
        content=DEFAULT_REPORT_OUTPUT.read_text(encoding="utf-8"),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/outputs/{filename}")
def get_output_file(filename: str):
    """下載成果影片或報表檔案."""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="檔案不存在")
    return FileResponse(file_path)


def run_cli_mode(args: argparse.Namespace) -> None:
    """純命令列批次執行模式 (保留給自動化或無 Web 介面需求)."""
    source_url, is_stream = resolve_video_source(args.source)
    source_arg = int(source_url) if source_url.isdigit() else source_url

    if isinstance(source_arg, str) and source_arg.startswith("http"):
        cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_arg)
    else:
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影片串流或檔案：{args.source}", file=sys.stderr)
        sys.exit(1)

    frame_width = 854
    frame_height = 480
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(raw_fps) if raw_fps > 0 else 25

    output_video_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    yawn_limit = max(1, int(fps * args.yawn_duration))
    yawn_counter = 0

    total_frames = 0
    speaking_frames = 0
    focused_frames = 0
    drowsy_frames = 0
    no_face_frames = 0
    drowsy_event_count = 0
    was_drowsy = False

    start_time = time.time()
    print("=" * 60)
    print("AI 臉部表情分析器 (CLI 模式啟動)")
    print(f"輸入來源    : {args.source}")
    print(f"輸出影片    : {output_video_path}")
    print(f"輸出報表    : {report_path}")
    print("=" * 60)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            if args.max_frames and total_frames > args.max_frames:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status_text = "STATUS: NO FACE DETECTED"
            status_color = (128, 128, 128)
            current_state = "NO_FACE"
            mar = 0.0

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = face_landmarks.landmark
                    p13 = np.array([pts[13].x * w, pts[13].y * h])
                    p14 = np.array([pts[14].x * w, pts[14].y * h])
                    p78 = np.array([pts[78].x * w, pts[78].y * h])
                    p308 = np.array([pts[308].x * w, pts[308].y * h])

                    dist_v = float(np.linalg.norm(p13 - p14))
                    dist_h = float(np.linalg.norm(p78 - p308))
                    mar = dist_v / dist_h if dist_h > 1e-6 else 0.0

                    for pt in (p13, p14, p78, p308):
                        cv2.circle(frame, tuple(pt.astype(int)), 3, (255, 255, 0), -1)

                    if mar > args.threshold_yawn:
                        yawn_counter += 1
                        if yawn_counter >= yawn_limit:
                            current_state = "DROWSY"
                            status_text = "WARNING: DROWSY / YAWNING"
                            status_color = (0, 0, 255)
                        else:
                            current_state = "SPEAKING"
                            status_text = "STATUS: SPEAKING (LARGE)"
                            status_color = (255, 0, 0)
                    elif mar > args.threshold_open:
                        yawn_counter = 0
                        current_state = "SPEAKING"
                        status_text = "STATUS: SPEAKING"
                        status_color = (255, 0, 0)
                    else:
                        yawn_counter = max(0, yawn_counter - 1)
                        current_state = "FOCUSED"
                        status_text = "STATUS: FOCUSED / SILENT"
                        status_color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.2f}",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
            else:
                yawn_counter = max(0, yawn_counter - 1)

            if current_state == "SPEAKING":
                speaking_frames += 1
            elif current_state == "DROWSY":
                drowsy_frames += 1
            elif current_state == "FOCUSED":
                focused_frames += 1
            else:
                no_face_frames += 1

            is_drowsy_now = (current_state == "DROWSY")
            if is_drowsy_now and not was_drowsy:
                drowsy_event_count += 1
            was_drowsy = is_drowsy_now

            # 繪製半透明控制面板
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (410, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.rectangle(frame, (10, 10), (410, 100), status_color, 2)
            cv2.putText(frame, "AI CLASSROOM ANALYZER v1.0", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(frame, status_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            if out.isOpened():
                out.write(frame)

            if not args.no_display:
                cv2.imshow("MediaPipe Face Mesh Analyzer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if out.isOpened():
            out.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - start_time)
    speaking_pct = (speaking_frames / total_frames * 100) if total_frames > 0 else 0.0
    focused_pct = (focused_frames / total_frames * 100) if total_frames > 0 else 0.0
    drowsy_pct = (drowsy_frames / total_frames * 100) if total_frames > 0 else 0.0
    video_sec = (total_frames / fps) if fps > 0 else 0.0

    report_lines = [
        "==================================================",
        "          MediaPipe Face Mesh 分析統計報表         ",
        "==================================================",
        f"產生時間      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"輸入來源      : {args.source}",
        f"輸出影片路徑  : {output_video_path}",
        f"分析總時長    : {video_sec:.2f} 秒 (實際運算耗時: {elapsed:.2f} 秒)",
        f"處理總幀數    : {total_frames} 幀 (@ {fps} FPS)",
        "--------------------------------------------------",
        "【狀態統計】",
        f"  - 專注/安靜 (FOCUSED)  : {focused_frames} 幀 ({focused_pct:.1f}%)",
        f"  - 發言狀態  (SPEAKING) : {speaking_frames} 幀 ({speaking_pct:.1f}%)",
        f"  - 疲倦警示  (DROWSY)   : {drowsy_frames} 幀 ({drowsy_pct:.1f}%)",
        f"  - 無人臉偵測 (NO FACE) : {no_face_frames} 幀",
        "--------------------------------------------------",
        "【事件計數】",
        f"  - 觸發打哈欠警示 (DROWSY) 次數 : {drowsy_event_count} 次",
        "==================================================",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Mesh 分析系統 (FastAPI Web 服務 / CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cli", action="store_true", help="以純命令列批次模式執行，不啟動 Web 服務")
    parser.add_argument("--port", type=int, default=3000, help="Web 服務監聽連接埠")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web 服務監聽主機")
    parser.add_argument("--source", type=str, default="https://www.youtube.com/watch?v=aVwzLrSAp78", help="CLI 輸入來源")
    parser.add_argument("--output", type=str, default="outputs/annotated.mp4", help="CLI 輸出影片")
    parser.add_argument("--report", type=str, default="outputs/report.txt", help="CLI 輸出報表")
    parser.add_argument("--threshold-open", type=float, default=0.15, help="CLI 張嘴閾值")
    parser.add_argument("--threshold-yawn", type=float, default=0.30, help="CLI 哈欠閾值")
    parser.add_argument("--yawn-duration", type=float, default=2.0, help="CLI 哈欠持續秒數")
    parser.add_argument("--no-display", action="store_true", help="CLI 模式關閉預覽視窗")
    parser.add_argument("--max-frames", type=int, default=None, help="CLI 模式最大處理幀數")

    cli_args = parser.parse_args()

    if cli_args.cli:
        run_cli_mode(cli_args)
    else:
        print(f"正在啟動 FastAPI Web 伺服器：http://localhost:{cli_args.port}")
        uvicorn.run("main:app", host=cli_args.host, port=cli_args.port, reload=False)
