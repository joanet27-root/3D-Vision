import cv2
import os


def extraer_frames(video_path, output_folder="frames_extraidos", salto=50):
    """
    Extrae 1 frame cada 'salto' frames de un vídeo
    y los guarda en una carpeta.

    Parámetros:
    ----------
    video_path : str
        Ruta del vídeo (.mov o .mp4)

    output_folder : str
        Carpeta donde se guardarán las imágenes

    salto : int
        Número de frames entre cada extracción
    """

    # Crear carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el vídeo: {video_path}")

    frame_id = 0
    saved_id = 0

    print(f"Procesando vídeo: {video_path}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Guardar 1 cada 'salto'
        if frame_id % salto == 0:
            nombre = os.path.join(output_folder, f"frame_{saved_id:04d}.jpg")
            cv2.imwrite(nombre, frame)

            print(f"Guardado: {nombre}")

            saved_id += 1

        frame_id += 1

    cap.release()

    print("\nProceso terminado")
    print(f"Frames totales leídos: {frame_id}")
    print(f"Frames guardados: {saved_id}")


if __name__ == "__main__":
    video_path = "video_path.MOV"   # Cambia aquí la ruta
    extraer_frames(video_path, output_folder="frames_extraidos", salto=70)