"""MediaPipe Face Mesh 動態關鍵點與 MAR 測試程式 (FaceMesh_Point.py).

依據 4MediaPipe_FaceMesh.pdf 第3節規格實作：
1. 支援 YouTube 串流或本機影片/攝影機輸入。
2. 提取嘴唇 4 個核心幾何關鍵點：
   - 13 : 上唇內側中心
   - 14 : 下唇內側中心
   - 78 : 左嘴角
   - 308: 右嘴角
3. 計算 MAR (Mouth Aspect Ratio) = 垂直距離 / 水平距離。
4. 在畫面上繪製追蹤綠點與即時 MAR 數值及開閉狀態。
5. 按 ESC 鍵離開。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yt_dlp


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe Face Mesh 嘴部 4 點動態追蹤與 MAR 計算")
    parser.add_argument(
        "--source",
        type=str,
        default="https://www.youtube.com/watch?v=jKrt3aaQLJk",
        help="輸入來源：YouTube 網址、本機影片路徑或攝影機編號 (0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="張嘴 (Speaking) 的 MAR 閾值",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="關閉預覽視窗 (無 GUI 環境時使用)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多處理幀數 (測試用，未指定則處理至影片結束)",
    )
    return parser.parse_args()


def resolve_stream_url(source: str) -> str:
    """解析來源網址，若為 YouTube 則透過 yt-dlp 取得串流網址."""
    if any(source.lower().startswith(prefix) for prefix in ("http://", "https://", "www.youtube.", "youtu.be")):
        print(f"正在解析 YouTube 影片串流：{source}")
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
        resolved_source = resolve_stream_url(args.source)
    except Exception as err:
        print(f"解析影片來源失敗：{err}", file=sys.stderr)
        return

    source_arg = int(resolved_source) if resolved_source.isdigit() else resolved_source
    if isinstance(source_arg, str) and source_arg.startswith("http"):
        cap = cv2.VideoCapture(source_arg, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_arg)
    else:
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print(f"無法讀取影片串流：{args.source}", file=sys.stderr)
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

    print("動態關鍵點追蹤啟動！按 ESC 鍵離開。")

    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                break

            # 統一縮放為 854x480 以提高運算效能
            frame = cv2.resize(frame, (854, 480))
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = face_landmarks.landmark

                    # 提取 4 點座標還原為像素
                    p13 = np.array([pts[13].x * w, pts[13].y * h])
                    p14 = np.array([pts[14].x * w, pts[14].y * h])
                    p78 = np.array([pts[78].x * w, pts[78].y * h])
                    p308 = np.array([pts[308].x * w, pts[308].y * h])

                    # 計算垂直與水平歐式距離
                    lip_dist_v = float(np.linalg.norm(p13 - p14))
                    lip_dist_h = float(np.linalg.norm(p78 - p308))
                    mar = lip_dist_v / lip_dist_h if lip_dist_h > 1e-6 else 0.0

                    # 繪製追蹤點
                    for pt in (p13, p14, p78, p308):
                        cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 0), -1)

                    # 狀態判斷與視覺化文字
                    if mar > args.threshold:
                        status = "Mouth: Open / Speaking"
                        color = (0, 0, 255)  # 紅色 (BGR)
                    else:
                        status = "Mouth: Closed"
                        color = (0, 255, 0)  # 綠色 (BGR)

                    cv2.putText(
                        frame,
                        f"MAR: {mar:.2f} - {status}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            if not args.no_display:
                cv2.imshow("AI Face Mesh points", frame)
                if cv2.waitKey(20) & 0xFF == 27:  # ESC 鍵
                    break
    finally:
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
