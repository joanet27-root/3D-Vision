import cv2
import numpy as np

INPUT_IMAGE = "img2.jpeg"
OUTPUT_WARP = "homografia_rectificada.jpg"
OUTPUT_DEBUG = "deteccion_arucos.jpg"

# IDs de las 4 esquinas del tablero
ARUCO_IDS = {
    "tl": 21,   # top-left
    "tr": 23,   # top-right
    "br": 22,   # bottom-right
    "bl": 20    # bottom-left
}

# Medidas del tablero rectificado
WIDTH = 800
HEIGHT = 1800

ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50


def detect_arucos(image, aruco_dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return [], None

    return corners, ids.flatten()


def marker_center(marker_corners):
    return np.mean(marker_corners, axis=0)


def draw_debug(image, ordered_points):
    img = image.copy()
    labels = ["TL", "TR", "BR", "BL"]

    for p, label in zip(ordered_points, labels):
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            img, label, (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

    poly = ordered_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=3)

    return img


def compute_homography_manual(src_pts, dst_pts):
    """
    Calcula la homografía manualmente usando DLT y SVD.
    src_pts: array de forma (4,2) con puntos origen
    dst_pts: array de forma (4,2) con puntos destino
    devuelve H de tamaño (3,3)
    """
    A = []

    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        xp, yp = dst_pts[i][0], dst_pts[i][1]

        # Primera ecuación
        A.append([
            -x, -y, -1,
             0,  0,  0,
             x * xp, y * xp, xp
        ])

        # Segunda ecuación
        A.append([
             0,  0,  0,
            -x, -y, -1,
             x * yp, y * yp, yp
        ])

    A = np.array(A, dtype=np.float64)

    # Resolver Ah = 0 usando SVD
    U, S, Vt = np.linalg.svd(A)

    # El último vector de Vt corresponde al menor valor singular
    h = Vt[-1, :]

    # Reorganizar como matriz 3x3
    H = h.reshape(3, 3)

    # Normalizar para que H[2,2] = 1 si es posible
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


def main():
    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {INPUT_IMAGE}")

    corners_list, ids = detect_arucos(image, ARUCO_DICT_NAME)

    if ids is None or len(ids) < 4:
        raise RuntimeError(
            f"Se detectaron {0 if ids is None else len(ids)} ArUco(s). "
            "Se necesitan al menos 4."
        )

    detected_centers = {}
    for marker_corners, marker_id in zip(corners_list, ids):
        marker_corners = marker_corners.reshape(4, 2)
        center = marker_center(marker_corners)
        detected_centers[int(marker_id)] = center

    required_ids = [
        ARUCO_IDS["tl"],
        ARUCO_IDS["tr"],
        ARUCO_IDS["br"],
        ARUCO_IDS["bl"]
    ]

    missing = [mid for mid in required_ids if mid not in detected_centers]
    if missing:
        raise RuntimeError(
            f"Faltan ArUco requeridos: {missing}. "
            f"Detectados: {sorted(detected_centers.keys())}"
        )

    # Puntos origen: centros de los ArUco
    src_pts = np.array([
        detected_centers[ARUCO_IDS["tl"]],
        detected_centers[ARUCO_IDS["tr"]],
        detected_centers[ARUCO_IDS["br"]],
        detected_centers[ARUCO_IDS["bl"]],
    ], dtype=np.float64)

    # Puntos destino: rectángulo visto desde arriba
    dst_pts = np.array([
        [0, 0],
        [WIDTH - 1, 0],
        [WIDTH - 1, HEIGHT - 1],
        [0, HEIGHT - 1]
    ], dtype=np.float64)

    # Homografía calculada manualmente
    H = compute_homography_manual(src_pts, dst_pts)

    # Aplicar homografía
    warped = cv2.warpPerspective(image, H, (WIDTH, HEIGHT))

    debug_img = draw_debug(image, src_pts)

    cv2.imwrite(OUTPUT_DEBUG, debug_img)
    cv2.imwrite(OUTPUT_WARP, warped)

    print("=== RESULTADOS ===")
    print("Centros usados como esquinas del tablero:")
    print(f"TL (id {ARUCO_IDS['tl']}): {src_pts[0]}")
    print(f"TR (id {ARUCO_IDS['tr']}): {src_pts[1]}")
    print(f"BR (id {ARUCO_IDS['br']}): {src_pts[2]}")
    print(f"BL (id {ARUCO_IDS['bl']}): {src_pts[3]}")
    print()
    print("Puntos destino:")
    print(dst_pts)
    print()
    print("Matriz A del sistema Ah=0:")
    print("(no se imprime completa para no recargar la salida)")
    print()
    print("Matriz de homografía H calculada manualmente:")
    print(H)
    print()
    print(f"Imagen debug guardada en: {OUTPUT_DEBUG}")
    print(f"Imagen rectificada guardada en: {OUTPUT_WARP}")


if __name__ == "__main__":
    main()