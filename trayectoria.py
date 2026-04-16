import json
import numpy as np
import matplotlib.pyplot as plt


HOMOGRAPHIES_JSON = "homografias.json"
K_JSON = "K_resultado.json"

BOARD_WIDTH = 3000.0
BOARD_HEIGHT = 2000.0


def load_K(k_json_path):
    with open(k_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["K"], dtype=np.float64)


def load_homographies(h_json_path):
    with open(h_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    homographies = []
    for image_entry in data["images"]:
        if image_entry.get("status") != "ok":
            continue
        H = np.array(image_entry["homography"], dtype=np.float64)
        homographies.append((image_entry["image_name"], H))
    return homographies


def estimate_pose_from_H(K, H):
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
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt

    C = -R.T @ t

    if C[2] < 0:
        R = -R
        t = -t

    return R, t


def camera_center_from_pose(R, t):
    return -R.T @ t


def main():
    K = load_K(K_JSON)
    homographies = load_homographies(HOMOGRAPHIES_JSON)

    centers = []
    names = []

    print("====================================")
    print("CENTROS DE CÁMARA")
    print("====================================")

    for image_name, H in homographies:
        R, t = estimate_pose_from_H(K, H)
        C = camera_center_from_pose(R, t)

        centers.append(C)
        names.append(image_name)

        print(f"{image_name}: C = {C}")

    centers = np.array(centers, dtype=np.float64)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        marker="o",
        label="Trayectoria de la cámara"
    )

    for i, (x, y, z) in enumerate(centers):
        ax.text(x, y, z, str(i), fontsize=9)

    board_x = [0, BOARD_WIDTH, BOARD_WIDTH, 0, 0]
    board_y = [0, 0, BOARD_HEIGHT, BOARD_HEIGHT, 0]
    board_z = [0, 0, 0, 0, 0]

    ax.plot(board_x, board_y, board_z, linestyle="--", label="Tablero")

    ax.set_title("Trayectoria 3D de la cámara")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()