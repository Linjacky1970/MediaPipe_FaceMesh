"""MediaPipe Face Mesh 靜態圖片分析範例 (FaceMesh_readImage.py).

功能：
1. 讀取靜態圖片 (預設 people01.jpg)。
2. 使用 MediaPipe FaceMesh 進行 468/478 點特徵點偵測。
3. 繪製三角網格 (Tesselation)、臉部輪廓 (Contours) 與虹膜 (Irises)。
4. 儲存標註成果圖片至指定路徑 (預設 annotated_image0.png)。
5. 支援 GUI 即時預覽與 --no-display 無介面模式。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp


def parse_arguments() -> argparse.Namespace:
    """解析命令列參數."""
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Mesh 靜態圖片分析",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default="people01.jpg",
        help="輸入圖片路徑",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_image0.png",
        help="標註圖片輸出路徑",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="關閉視窗預覽 (無 GUI 環境或自動化測試時使用)",
    )
    return parser.parse_args()


def resolve_file_path(path_str: str) -> Path:
    """解析路徑，若相對路徑在目前目錄找不到，則嘗試自程式所在目錄尋找."""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.resolve()
    script_relative = Path(__file__).resolve().parent / path_str
    if script_relative.exists():
        return script_relative
    return candidate.resolve()


def main() -> None:
    args = parse_arguments()

    image_path = resolve_file_path(args.image)
    output_path = Path(args.output).resolve()

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"無法讀取圖片：{image_path}")

    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"圖片中未偵測到人臉，略過：{image_path}")
        return

    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
        # 1. 繪製三角網格 (FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                mp_drawing_styles.get_default_face_mesh_tesselation_style()
            ),
        )
        # 2. 繪製五官與臉部輪廓線 (FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                mp_drawing_styles.get_default_face_mesh_contours_style()
            ),
        )
        # 3. 繪製眼球虹膜 (FACEMESH_IRISES)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            ),
        )

    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError(f"無法儲存標註圖片：{output_path}")

    print(f"標註結果已儲存至：{output_path}")

    if not args.no_display:
        try:
            cv2.imshow("MediaPipe Face Mesh", annotated_image)
            print("請於視窗中按任意鍵關閉...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as err:
            print(f"視窗預覽無法開啟 ({err})，已略過視窗顯示。", file=sys.stderr)


if __name__ == "__main__":
    main()
