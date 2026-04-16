import json
import numpy as np


# ==========================================
# CONFIGURACIÓN
# ==========================================

INPUT_JSON = "arucos_detectados.json"
OUTPUT_JSON = "homografias.json"

# Tamaño real del ArUco en mm
ARUCO_SIZE = 100.0

# Centros de los ArUcos en el tablero (mm)
# Sistema fijado por la imagen métrica:
# 23 -> superior izquierda
# 22 -> superior derecha
# 21 -> inferior izquierda
# 20 -> inferior derecha
MARKER_CENTERS = {
    23: (2400.0, 1400.0),
    22: (600.0, 1400.0),
    21: (2400.0, 600.0),
    20: (600.0, 600.0)
}


# ==========================================
# GEOMETRÍA MUNDO
# ==========================================

def get_world_corners(marker_id):
    """
    Devuelve las 4 esquinas mundo del marcador en el mismo orden
    que las detectadas por OpenCV en el JSON: 0,1,2,3.
    """
    cx, cy = MARKER_CENTERS[marker_id]
    s = ARUCO_SIZE / 2.0

    return np.array([
        [cx - s, cy - s],   # corner 0
        [cx + s, cy - s],   # corner 1
        [cx + s, cy + s],   # corner 2
        [cx - s, cy + s]    # corner 3
    ], dtype=np.float64)


# ==========================================
# NORMALIZACIÓN DE HARTLEY
# ==========================================

def normalize_points_2d(points):
    points = np.asarray(points, dtype=np.float64)

    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

    scale = 1.0 if mean_dist < 1e-12 else np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_norm_h = (T @ points_h.T).T

    return points_norm_h[:, :2], T


# ==========================================
# DLT MANUAL
# ==========================================

def compute_homography_manual(world_pts, image_pts):
    """
    Calcula H tal que x ~ H X usando DLT normalizado.
    """
    if len(world_pts) < 4 or len(image_pts) < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias.")

    world_norm, T_world = normalize_points_2d(world_pts)
    image_norm, T_img = normalize_points_2d(image_pts)

    A = []

    for (X, Y), (x, y) in zip(world_norm, image_norm):
        A.append([-X, -Y, -1, 0, 0, 0, x * X, x * Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y * X, y * Y, y])

    A = np.array(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Desnormalizar
    H = np.linalg.inv(T_img) @ H_norm @ T_world

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H, A


# ==========================================
# EXTRAER PUNTOS DESDE JSON
# ==========================================

def extract_points_from_image(image_record):
    world_points = []
    image_points = []

    for aruco in image_record["arucos"]:
        marker_id = int(aruco["id"])

        if marker_id not in MARKER_CENTERS:
            continue

        world_corners = get_world_corners(marker_id)

        corners_sorted = sorted(aruco["corners"], key=lambda c: c["corner_index"])
        img_corners = np.array(
            [[corner["x"], corner["y"]] for corner in corners_sorted],
            dtype=np.float64
        )

        world_points.extend(world_corners)
        image_points.extend(img_corners)

    return np.array(world_points, dtype=np.float64), np.array(image_points, dtype=np.float64)


# ==========================================
# MAIN
# ==========================================

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "input_json": INPUT_JSON,
        "aruco_size_mm": ARUCO_SIZE,
        "marker_centers_mm": {str(k): list(v) for k, v in MARKER_CENTERS.items()},
        "images": []
    }

    valid_count = 0

    for image_record in data["images"]:
        image_name = image_record["image_name"]

        # Compatible con el JSON nuevo
        if not image_record.get("has_all_four_board_arucos", False):
            print(f"[SKIP] {image_name}: no están los 4 ArUcos del tablero")
            continue

        world_pts, image_pts = extract_points_from_image(image_record)

        if len(world_pts) != 16 or len(image_pts) != 16:
            print(f"[SKIP] {image_name}: no tiene 16 puntos válidos")
            continue

        try:
            H, A = compute_homography_manual(world_pts, image_pts)
        except Exception as e:
            print(f"[WARN] {image_name}: error al calcular H -> {e}")
            continue

        results["images"].append({
            "image_name": image_name,
            "status": "ok",
            "num_points": int(len(world_pts)),
            "world_points": world_pts.tolist(),
            "image_points": image_pts.tolist(),
            "homography": H.tolist(),
            "A_shape": list(A.shape)
        })

        valid_count += 1
        print(f"[OK] {image_name}: H calculada con 16 puntos")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\nProceso terminado.")
    print(f"Homografías válidas: {valid_count}")
    print(f"JSON guardado en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()