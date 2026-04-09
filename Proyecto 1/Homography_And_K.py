import os
import glob
import cv2
import numpy as np

# =========================================================
# CONFIGURACIÓN
# =========================================================

IMAGE_FOLDER = "frames"
IMAGE_EXT = "*.jpg"

ARUCO_DICT_NAME = cv2.aruco.DICT_4X4_50

# IDs fijados siempre a la misma referencia lógica
ARUCO_IDS = {
    "tl": 23,
    "tr": 22,
    "br": 20,
    "bl": 21
}

# Tamaño real del lado del ArUco (mm)
MARKER_SIZE = 100.0

# Sistema mundo FIJO del tablero
# Usamos exactamente las coordenadas reales que indicas.
# IMPORTANTE:
# Estas coordenadas deben corresponder al punto TL del marcador
# en el MISMO sistema de referencia para todos.

MARKER_INFO = {
    21: {"origin": (2450,1450), "rotation": 90},
    23: {"origin": (2450,650),  "rotation": 90},
    22: {"origin": (650,650),   "rotation": 90},
    20: {"origin": (650,1450),  "rotation": 90},
}

REQUIRE_ALL_FOUR = True
SAVE_DEBUG_CORNERS = True

# =========================================================
# DETECCIÓN ARUCO
# =========================================================

def detect_arucos(image, aruco_dict_name):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return [], None

    return corners, ids.flatten()


def draw_aruco_corner_order(image, corners_list, ids):
    img = image.copy()

    if ids is None:
        return img

    for marker_corners, marker_id in zip(corners_list, ids):
        corners = marker_corners.reshape(4, 2)

        for i, (x, y) in enumerate(corners):
            x, y = int(round(x)), int(round(y))
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                img, f"{i}", (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(
            img, f"ID {int(marker_id)}", tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    return img


# =========================================================
# GEOMETRÍA DEL PLANO
# =========================================================

def rotate_corners(corners, k):
    """
    Rota la lista de esquinas en pasos de 90 grados.
    k = 0,1,2,3  ->  0,90,180,270 grados
    """
    corners = np.asarray(corners, dtype=np.float64)
    return np.roll(corners, -k, axis=0)


def get_marker_world_corners(marker_id):
    """
    Devuelve las 4 esquinas mundo del ArUco en orden compatible
    con OpenCV: TL, TR, BR, BL en la imagen detectada.
    """
    info = MARKER_INFO[marker_id]
    x0, y0 = info["origin"]
    rot_deg = info["rotation"]
    s = MARKER_SIZE

    # marcador sin rotación en el sistema del tablero
    base_corners = np.array([
        [x0,     y0    ],   # TL
        [x0 + s, y0    ],   # TR
        [x0 + s, y0 + s],   # BR
        [x0,     y0 + s]    # BL
    ], dtype=np.float64)

    k = (rot_deg // 90) % 4
    return rotate_corners(base_corners, k)


def get_correspondences_from_image(image, image_name="debug"):
    """
    Devuelve:
    - world_pts: puntos del plano (Nx2)
    - image_pts: puntos de imagen (Nx2)
    """
    corners_list, ids = detect_arucos(image, ARUCO_DICT_NAME)

    if SAVE_DEBUG_CORNERS:
        debug_img = draw_aruco_corner_order(image, corners_list, ids)
        cv2.imwrite(f"debug_corners_{image_name}.jpg", debug_img)

    if ids is None:
        return None, None

    ids = [int(i) for i in ids]
    required_ids = list(ARUCO_IDS.values())

    if REQUIRE_ALL_FOUR:
        missing = [mid for mid in required_ids if mid not in ids]
        if missing:
            return None, None

    world_pts = []
    image_pts = []

    for marker_corners, marker_id in zip(corners_list, ids):
        if marker_id not in required_ids:
            continue

        img_corners = marker_corners.reshape(4, 2).astype(np.float64)
        obj_corners = get_marker_world_corners(marker_id)

        for obj_p, img_p in zip(obj_corners, img_corners):
            world_pts.append(obj_p)
            image_pts.append(img_p)

    if len(world_pts) < 4:
        return None, None

    return np.array(world_pts, dtype=np.float64), np.array(image_pts, dtype=np.float64)


# =========================================================
# DLT PARA HOMOGRAFÍA
# =========================================================

def normalize_points_2d(pts):
    pts = np.asarray(pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)

    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))

    scale = 1.0 if mean_dist < 1e-12 else np.sqrt(2) / mean_dist

    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_h = (T @ pts_h.T).T

    return pts_norm_h[:, :2], T


def dlt_homography(world_pts, image_pts):
    """
    Calcula H tal que:
        x ~ H X
    donde X son puntos del plano mundo y x puntos en imagen.
    """
    world_pts = np.asarray(world_pts, dtype=np.float64)
    image_pts = np.asarray(image_pts, dtype=np.float64)

    if world_pts.shape[0] != image_pts.shape[0] or world_pts.shape[0] < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias válidas.")

    world_norm, T_world = normalize_points_2d(world_pts)
    image_norm, T_img = normalize_points_2d(image_pts)

    A = []
    for (X, Y), (x, y) in zip(world_norm, image_norm):
        A.append([-X, -Y, -1,  0,  0,  0, x * X, x * Y, x])
        A.append([ 0,  0,  0, -X, -Y, -1, y * X, y * Y, y])

    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    H = np.linalg.inv(T_img) @ H_norm @ T_world

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


# =========================================================
# MÉTODO DE ZHANG PARA K
# =========================================================

def vij(H, i, j):
    h = H.T
    hi = h[i - 1]
    hj = h[j - 1]

    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2]
    ], dtype=np.float64)


def estimate_K_from_homographies(H_list):
    V = []

    for H in H_list:
        V.append(vij(H, 1, 2))
        V.append(vij(H, 1, 1) - vij(H, 2, 2))

    V = np.asarray(V, dtype=np.float64)

    print("Número de H:", len(H_list))
    print("Rango de V:", np.linalg.matrix_rank(V))
    print("Valores singulares de V:", np.linalg.svd(V, compute_uv=False))

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    print("b =", b)

    if b[0] < 0:
        b = -b

    B11, B12, B22, B13, B23, B33 = b

    print("B11 =", B11)
    print("B12 =", B12)
    print("B22 =", B22)
    print("B13 =", B13)
    print("B23 =", B23)
    print("B33 =", B33)
    print("B11*B22 - B12^2 =", B11 * B22 - B12**2)

    denom = B11 * B22 - B12**2
    if abs(denom) < 1e-12:
        raise RuntimeError(
            "Sistema degenerado al estimar K: B11*B22 - B12^2 es casi 0."
        )

    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    if lam <= 0:
        b = -b
        B11, B12, B22, B13, B23, B33 = b
        denom = B11 * B22 - B12**2

        if abs(denom) < 1e-12:
            raise RuntimeError(
                "Sistema degenerado al estimar K tras invertir el signo de b."
            )

        v0 = (B12 * B13 - B11 * B23) / denom
        lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    if B11 <= 0 or lam <= 0:
        raise RuntimeError(
            "No se pudo obtener una K física. "
            "Revisa el orden de esquinas, MARKER_POSITIONS y las homografías."
        )

    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - (B13 * alpha**2) / lam

    K = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

    return K


# =========================================================
# EXTRÍNSECOS A PARTIR DE K Y H
# =========================================================

def estimate_extrinsics_from_H(K, H):
    K_inv = np.linalg.inv(K)

    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    lam = 1.0 / np.linalg.norm(K_inv @ h1)

    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (K_inv @ h3)

    R_approx = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        t *= -1

    return R, t


# =========================================================
# ERROR DE REPROYECCIÓN
# =========================================================

def project_points_planar(K, R, t, world_pts):
    projected = []

    for X, Y in world_pts:
        Pw = np.array([X, Y, 0.0], dtype=np.float64)
        Pc = R @ Pw + t

        x = K @ Pc
        x = x / x[2]

        projected.append(x[:2])

    return np.array(projected, dtype=np.float64)


def reprojection_error(K, H_list, correspondences):
    all_errors = []
    per_image = []

    for H, (world_pts, image_pts, image_name) in zip(H_list, correspondences):
        R, t = estimate_extrinsics_from_H(K, H)
        proj = project_points_planar(K, R, t, world_pts)

        err = np.linalg.norm(proj - image_pts, axis=1)
        mean_err = np.mean(err)

        all_errors.extend(err.tolist())
        per_image.append((image_name, mean_err))

    global_mean = np.mean(all_errors) if all_errors else None
    return global_mean, per_image


# =========================================================
# MAIN
# =========================================================

def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_FOLDER, IMAGE_EXT)))

    if not image_paths:
        raise RuntimeError("No se encontraron imágenes.")

    H_list = []
    correspondences = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] No se pudo leer: {path}")
            continue

        base_name = os.path.splitext(os.path.basename(path))[0]

        world_pts, image_pts = get_correspondences_from_image(image, base_name)

        if world_pts is None or image_pts is None:
            print(f"[WARN] No se obtuvieron correspondencias válidas en: {path}")
            continue

        try:
            H = dlt_homography(world_pts, image_pts)
        except Exception as e:
            print(f"[WARN] Error calculando H en {path}: {e}")
            continue

        H_list.append(H)
        correspondences.append((world_pts, image_pts, os.path.basename(path)))

        print(f"\nImagen: {path}")
        print(f"Número de puntos usados: {len(world_pts)}")
        print("H =")
        print(H)

    if len(H_list) < 3:
        raise RuntimeError("Se necesitan al menos 3 homografías válidas para estimar K.")

    K = estimate_K_from_homographies(H_list)

    print("\n=========================")
    print("Número de homografías usadas:", len(H_list))
    print("Matriz K estimada:")
    print(K)
    print("=========================")

    global_err, per_image = reprojection_error(K, H_list, correspondences)

    print("\nError medio global de reproyección (px):", global_err)
    print("\nError por imagen:")
    for name, err in per_image:
        print(f"{name}: {err:.4f} px")


if __name__ == "__main__":
    main()