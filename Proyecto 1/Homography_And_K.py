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

# IDs de los 4 ArUco usados
ARUCO_IDS = {
    "tl": 21,
    "tr": 23,
    "br": 22,
    "bl": 20
}

# Tamaño real del marcador en las unidades que quieras
# (mm, cm, etc.). Sé consistente.
# Tamaño real del marcador en mm
MARKER_SIZE = 100.0

# Coordenadas aproximadas basadas en las dimensiones del campo (3000x2000 mm)
# Nota: La normativa no define IDs fijos para las esquinas, 
# los IDs 36 y 47 son para las piezas móviles (cajas).
MARKER_POSITIONS = {
    21: (0.0, 0.0),
    23: (300.0, 0.0),
    22: (300.0, 500.0),
    20: (0.0, 500.0)
}

# Si quieres exigir que estén los 4 ArUco siempre:
REQUIRE_ALL_FOUR = True


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


# =========================================================
# GEOMETRÍA DEL PLANO
# =========================================================

def get_marker_world_corners(marker_id):
    """
    Devuelve las 4 esquinas del marcador en el plano del tablero.
    Orden asumido: top-left, top-right, bottom-right, bottom-left
    """
    x0, y0 = MARKER_POSITIONS[marker_id]
    s = MARKER_SIZE

    return np.array([
        [x0,     y0    ],   # TL
        [x0 + s, y0    ],   # TR
        [x0 + s, y0 + s],   # BR
        [x0,     y0 + s]    # BL
    ], dtype=np.float64)


def get_correspondences_from_image(image):
    """
    Devuelve:
    - world_pts: puntos del plano (Nx2)
    - image_pts: puntos de imagen (Nx2)

    Usa las 4 esquinas de cada ArUco detectado.
    """
    corners_list, ids = detect_arucos(image, ARUCO_DICT_NAME)

    if ids is None:
        return None, None

    ids = [int(i) for i in ids]

    required_ids = list(ARUCO_IDS.values())
    detected_required = [i for i in ids if i in required_ids]

    if REQUIRE_ALL_FOUR:
        missing = [mid for mid in required_ids if mid not in ids]
        if missing:
            return None, None

    world_pts = []
    image_pts = []

    for marker_corners, marker_id in zip(corners_list, ids):
        if marker_id not in required_ids:
            continue

        # OpenCV devuelve shape (1,4,2), lo pasamos a (4,2)
        img_corners = marker_corners.reshape(4, 2).astype(np.float64)

        # Esquinas reales del marcador en el plano
        obj_corners = get_marker_world_corners(marker_id)

        # Añadimos las 4 correspondencias
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
    """
    Normalización de Hartley para mejorar estabilidad numérica.
    """
    pts = np.asarray(pts, dtype=np.float64)
    centroid = np.mean(pts, axis=0)

    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))

    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2) / mean_dist

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
    donde X son puntos del plano y x puntos de imagen.
    """
    world_pts = np.asarray(world_pts, dtype=np.float64)
    image_pts = np.asarray(image_pts, dtype=np.float64)

    if world_pts.shape[0] != image_pts.shape[0] or world_pts.shape[0] < 4:
        raise ValueError("Se necesitan al menos 4 correspondencias válidas.")

    world_norm, T_world = normalize_points_2d(world_pts)
    image_norm, T_img = normalize_points_2d(image_pts)

    A = []

    for (X, Y), (x, y) in zip(world_norm, image_norm):
        A.append([-X, -Y, -1,  0,  0,  0, x*X, x*Y, x])
        A.append([ 0,  0,  0, -X, -Y, -1, y*X, y*Y, y])

    A = np.asarray(A, dtype=np.float64)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)

    # Desnormalización
    H = np.linalg.inv(T_img) @ H_norm @ T_world

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    return H


# =========================================================
# MÉTODO DE ZHANG PARA K
# =========================================================

def vij(H, i, j):
    """
    i, j en {1,2,3}
    """
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    cols = [h1, h2, h3]
    hi = cols[i - 1]
    hj = cols[j - 1]

    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2]
    ], dtype=np.float64)


def estimate_K_from_homographies(H_list):
    """
    Estima la matriz intrínseca K a partir de una lista de homografías
    usando el método de Zhang.

    Cada H debe mapear:
        plano_del_mundo -> imagen

    Devuelve:
        K: matriz intrínseca 3x3
    """
    V = []

    for H in H_list:
        V.append(vij(H, 1, 2))
        V.append(vij(H, 1, 1) - vij(H, 2, 2))

    V = np.asarray(V, dtype=np.float64)

    # Resolver Vb = 0 con SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    # La solución está definida salvo escala: b y -b son equivalentes
    # Forzamos un signo consistente
    if b[0] < 0:
        b = -b

    B11, B12, B22, B13, B23, B33 = b

    denom = B11 * B22 - B12**2
    if abs(denom) < 1e-12:
        raise RuntimeError(
            "Sistema degenerado al estimar K: B11*B22 - B12^2 es casi 0."
        )

    v0 = (B12 * B13 - B11 * B23) / denom

    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    # Si lambda sale negativa, probamos el signo opuesto
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

    # Ortonormalización por SVD
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
    """
    world_pts: Nx2, asumimos Z=0
    """
    projected = []

    for X, Y in world_pts:
        Pw = np.array([X, Y, 0.0], dtype=np.float64)
        Pc = R @ Pw + t

        x = K @ Pc
        x = x / x[2]

        projected.append(x[:2])

    return np.array(projected, dtype=np.float64)


def reprojection_error(K, H_list, correspondences):
    """
    correspondences: lista de tuplas (world_pts, image_pts, image_name)
    """
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

        world_pts, image_pts = get_correspondences_from_image(image)

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