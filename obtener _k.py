import os
import json
import numpy as np


# =========================================================
# CONFIGURACIÓN
# =========================================================

INPUT_JSON = "homografias.json"
OUTPUT_JSON = "K_resultado.json"


# =========================================================
# UTILIDADES ZHANG
# =========================================================

def vij(H, i, j):
    """
    Devuelve el vector v_ij de Zhang a partir de la homografía H.
    i, j en {1, 2, 3}
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


def build_V_matrix(H_list):
    """
    Construye la matriz V del método de Zhang.
    Cada homografía aporta dos ecuaciones:
      v12
      v11 - v22
    """
    V = []

    for H in H_list:
        V.append(vij(H, 1, 2))
        V.append(vij(H, 1, 1) - vij(H, 2, 2))

    return np.asarray(V, dtype=np.float64)


def estimate_K_from_homographies(H_list):
    """
    Estima K a partir de varias homografías usando Zhang.
    Devuelve:
      K, diagnostic
    """
    V = build_V_matrix(H_list)

    rank_V = int(np.linalg.matrix_rank(V))
    singular_values = np.linalg.svd(V, compute_uv=False)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1].copy()

    # Fijar signo consistente
    if b[0] < 0:
        b = -b

    B11, B12, B22, B13, B23, B33 = b

    denom = B11 * B22 - B12**2
    if abs(denom) < 1e-18:
        raise RuntimeError("Sistema degenerado: B11*B22 - B12^2 es casi cero.")

    v0 = (B12 * B13 - B11 * B23) / denom

    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    # Probar el signo opuesto si lambda sale mal
    if lam <= 0:
        b = -b
        B11, B12, B22, B13, B23, B33 = b

        denom = B11 * B22 - B12**2
        if abs(denom) < 1e-18:
            raise RuntimeError("Sistema degenerado tras invertir el signo de b.")

        v0 = (B12 * B13 - B11 * B23) / denom
        lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11

    if B11 <= 0:
        raise RuntimeError("B11 no es positiva; no se puede obtener una K física.")

    if lam <= 0:
        raise RuntimeError("Lambda no positiva; no se puede obtener una K física.")

    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - (B13 * alpha**2) / lam

    K = np.array([
        [alpha, gamma, u0],
        [0.0,   beta,  v0],
        [0.0,   0.0,   1.0]
    ], dtype=np.float64)

    B_matrix = np.array([
        [B11, B12, B13],
        [B12, B22, B23],
        [B13, B23, B33]
    ], dtype=np.float64)

    diagnostic = {
        "num_homographies_used": len(H_list),
        "V_shape": list(V.shape),
        "rank_V": rank_V,
        "V_singular_values": singular_values.tolist(),
        "b_vector": b.tolist(),
        "B_matrix": B_matrix.tolist(),
        "denom_B11B22_minus_B12sq": float(denom),
        "lambda": float(lam),
        "fx_alpha": float(alpha),
        "fy_beta": float(beta),
        "skew_gamma": float(gamma),
        "cx_u0": float(u0),
        "cy_v0": float(v0)
    }

    return K, diagnostic


# =========================================================
# LECTURA DEL JSON
# =========================================================

def load_valid_homographies(input_json):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"No existe el archivo: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    H_list = []
    used_images = []

    for image_entry in data.get("images", []):
        if image_entry.get("status") != "ok":
            continue

        if image_entry.get("num_points", 0) != 16:
            continue

        H = np.array(image_entry["homography"], dtype=np.float64)

        if H.shape != (3, 3):
            continue

        H_list.append(H)
        used_images.append(image_entry["image_name"])

    if len(H_list) < 3:
        raise RuntimeError("Se necesitan al menos 3 homografías válidas para estimar K.")

    return H_list, used_images


# =========================================================
# MAIN
# =========================================================

def main():
    H_list, used_images = load_valid_homographies(INPUT_JSON)

    K, diagnostic = estimate_K_from_homographies(H_list)

    output = {
        "input_json": INPUT_JSON,
        "used_images": used_images,
        "K": K.tolist(),
        "diagnostic": diagnostic
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("====================================")
    print("MATRIZ K ESTIMADA")
    print(K)
    print("====================================")
    print(f"Homografías usadas: {len(used_images)}")
    print("Imágenes usadas:")
    for name in used_images:
        print(" -", name)
    print(f"Rango de V: {diagnostic['rank_V']}")
    print(f"Valores singulares de V: {diagnostic['V_singular_values']}")
    print(f"Resultado guardado en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()