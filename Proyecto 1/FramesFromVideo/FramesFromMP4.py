from pathlib import Path
import cv2


def extraer_un_frame_por_segundo(video_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"No existe el archivo de video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("No se pudo detectar el FPS del video.")

    duracion_segundos = int(total_frames / fps)

    print(f"Video: {video_path.name}")
    print(f"FPS detectados: {fps:.3f}")
    print(f"Frames totales: {total_frames}")
    print(f"Duración aproximada: {duracion_segundos} s")

    for segundo in range(duracion_segundos + 1):
        frame_index = int(segundo * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = cap.read()
        if not ok:
            print(f"No se pudo leer el frame del segundo {segundo}")
            continue

        output_file = output_dir / f"frame_{segundo:04d}.jpg"
        cv2.imwrite(str(output_file), frame)
        print(f"Guardado: {output_file}")

    cap.release()
    print("Proceso terminado.")


if __name__ == "__main__":
    carpeta_script = Path(__file__).resolve().parent
    video_entrada = carpeta_script / "JorgeGay2.mp4"
    carpeta_salida = carpeta_script / "frames_jpg2"

    extraer_un_frame_por_segundo(video_entrada, carpeta_salida)