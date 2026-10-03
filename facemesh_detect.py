"""課堂專注度與發言狀態分析器 (facemesh_detect.py).

依據 4MediaPipe_FaceMesh.pdf 第4節規格實作：
1. 透過 yt_dlp 串流或本機影片讀取影像。
2. MediaPipe FaceMesh 人臉網格偵測。
3. 提取嘴唇 4 點 (13, 14, 78, 308) 計算 MAR。
4. 三態邏輯判斷：
   - mar > 0.30 且連續 2 秒 (fps * 2): WARNING: DROWSY / YAWNING (紅色)
   - mar > 0.30 未滿 2 秒: STATUS: SPEAKING (LARGE) (藍色)
   - mar > 0.15: STATUS: SPEAKING (藍色)
   - 其餘: STATUS: FOCUSED / SILENT (綠色)
5. 左上角繪製半透明控制面板 HUD 與動態狀態邊框。
6. 匯出成果影片 ai_classroom_analysis.mp4。
"""
# 2222

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yt_dlp


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="課堂專注度與發言狀態分析器 (facemesh_detect.py)")
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=aVwzLrSAp78&t=9s",
        help="影片來源 URL 或本機檔案路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai_classroom_analysis.mp4",
        help="輸出影片名稱",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="關閉預覽視窗",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多處理幀數 (測試用，未指定則處理至影片結束)",
    )
    return parser.parse_args()


def resolve_stream_url(source: str) -> str:
    if any(source.lower().startswith(prefix) for prefix in ("http://", "https://", "www.youtube.", "youtu.be")):
        print("正在連線抓取影片串流...")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source, download=False)
            return info["url"]
    return source


def main() -> None:
    args = parse_arguments()
    try:
        stream_url = resolve_stream_url(args.source)
    except Exception as err:
        print(f"無法解析影片來源：{err}", file=sys.stderr)
        return

    source_arg = int(stream_url) if stream_url.isdigit() else stream_url
    if isinstance(source_arg, str) and source_arg.startswith("http"):
        cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_arg)
    else:
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print("無法讀取影片串流！", file=sys.stderr)
        return

    frame_width = 854
    frame_height = 480
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(raw_fps) if raw_fps > 0 else 25

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    MAR_THRESHOLD_OPEN = 0.15
    MAR_THRESHOLD_YAWN = 0.30
    yawn_counter = 0
    YAWN_FRAMES_LIMIT = fps * 2

    print(f"AI 分析啟動！成果將自動儲存至 {output_path}。按下 'q' 可提早結束。")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            status_text = "STATUS: SILENT"
            status_color = (0, 255, 0)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = face_landmarks.landmark

                    p13 = np.array([pts[13].x * w, pts[13].y * h])
                    p14 = np.array([pts[14].x * w, pts[14].y * h])
                    p78 = np.array([pts[78].x * w, pts[78].y * h])
                    p308 = np.array([pts[308].x * w, pts[308].y * h])

                    lip_dist_v = float(np.linalg.norm(p13 - p14))
                    lip_dist_h = float(np.linalg.norm(p78 - p308))
                    mar = lip_dist_v / lip_dist_h if lip_dist_h > 1e-6 else 0.0

                    for pt in (p13, p14, p78, p308):
                        cv2.circle(frame, tuple(pt.astype(int)), 3, (255, 255, 0), -1)

                    if mar > MAR_THRESHOLD_YAWN:
                        yawn_counter += 1
                        if yawn_counter >= YAWN_FRAMES_LIMIT:
                            status_text = "WARNING: DROWSY / YAWNING"
                            status_color = (0, 0, 255)
                        else:
                            status_text = "STATUS: SPEAKING (LARGE)"
                            status_color = (255, 0, 0)
                    elif mar > MAR_THRESHOLD_OPEN:
                        yawn_counter = 0
                        status_text = "STATUS: SPEAKING"
                        status_color = (255, 0, 0)
                    else:
                        yawn_counter = max(0, yawn_counter - 1)
                        status_text = "STATUS: FOCUSED / SILENT"
                        status_color = (0, 255, 0)

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.2f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )

            # 繪製半透明黑色左上角面板
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.rectangle(frame, (10, 10), (380, 100), status_color, 2)
            cv2.putText(
                frame,
                "AI CLASSROOM ANALYZER v1.0",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            cv2.putText(
                frame,
                status_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            out.write(frame)
            if not args.no_display:
                cv2.imshow("AI Project - Classroom Analyzer", frame)
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        out.release()
        face_mesh.close()
        cv2.destroyAllWindows()

    print(f"分析完畢！成果已儲存至 {output_path}。")


if __name__ == "__main__":
    main()
