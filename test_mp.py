import mediapipe as mp


def main() -> None:
    print(f"MediaPipe 版本：{mp.__version__}")

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
    ):
        print("Face Mesh 初始化成功")


if __name__ == "__main__":
    main()
