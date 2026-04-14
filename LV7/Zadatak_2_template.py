import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans


IMAGE_FILES = [
    "imgs/test_1.jpg",
    "imgs/test_2.jpg",
    "imgs/test_3.jpg",
    "imgs/test_4.jpg",
    "imgs/test_5.jpg",
    "imgs/test_6.jpg",
]
K = 5                          # broj grupa za osnovnu kvantizaciju
K_values = [2, 5, 10, 20, 50]  # vrijednosti K za vizualni eksperiment
K_elbow = list(range(1, 16))   # raspon K za elbow grafikon (1 do 15)


def quantize_image(img_array, k):
    """Primijeni K-means i vrati aproksimirano 2D polje piksela i km objekt."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(img_array)
    return km.cluster_centers_[km.labels_], km


def process_image(filepath):
    print(f"\n{'='*60}")
    print(f"Slika: {filepath}")
    print(f"{'='*60}")

    # --- ucitavanje i priprema ---
    img_raw = Image.imread(filepath)
    img = img_raw.astype(np.float64) / 255
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    # --- 1. Broj razlicitih boja ---
    img_uint8 = (img * 255).astype(np.uint8)
    unique_colors = np.unique(np.reshape(img_uint8, (w * h, d)), axis=0)
    print(f"Broj razlicitih boja: {len(unique_colors)}")

    # --- 2. & 3. K-means + zamjena piksela centrom ---
    img_array_aprox, kmeans = quantize_image(img_array, K)
    img_aprox = np.clip(np.reshape(img_array_aprox, (w, h, d)), 0, 1)

    centers = kmeans.cluster_centers_
    labels_2d = np.reshape(kmeans.labels_, (w, h))  # labele u obliku slike

    print(f"Centri klastera (K={K}):")
    for i, c in enumerate(centers):
        print(f"  Boja {i+1}: R={c[0]:.3f}  G={c[1]:.3f}  B={c[2]:.3f}")

    # --- 4a. Usporedba originalna vs kvantizirana ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Originalna slika")
    axes[0].axis("off")
    axes[1].imshow(img_aprox)
    axes[1].set_title(f"Kvantizirana slika (K={K})")
    axes[1].axis("off")
    plt.suptitle(f"{filepath}  —  K={K}", fontsize=13)
    plt.tight_layout()
    plt.show()


    # ------------------------------------------------------------------
    # 5. Elbow metoda: J(K) = inertia u ovisnosti o broju grupa K
    # ------------------------------------------------------------------
    inertias = []
    for k in K_elbow:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(img_array)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_elbow, inertias, marker='o', linewidth=2, markersize=7,
            color='steelblue', markerfacecolor='tomato', markeredgecolor='tomato')
    ax.set_xlabel("Broj grupa K", fontsize=12)
    ax.set_ylabel("J  (inertia)", fontsize=12)
    ax.set_title(f"Elbow metoda — {filepath}", fontsize=13)
    ax.set_xticks(K_elbow)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print(f"\nInertia po K:")
    for k, j in zip(K_elbow, inertias):
        print(f"  K={k:2d}  ->  J = {j:.6f}")

    # ------------------------------------------------------------------
    # 6. Binarne maske — svaka grupa kao zasebna crno-bijela slika
    # ------------------------------------------------------------------
    # Broj stupaca u gridu: prilagodi se broju grupa
    n_cols = K
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    for cluster_id in range(K):
        # Binarna maska: 1 (bijelo) gdje piksel pripada grupi, 0 (crno) inace
        binary_mask = (labels_2d == cluster_id).astype(np.uint8)

        # Boja centra tog klastera (za naslov)
        c = centers[cluster_id]
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
        )

        axes[cluster_id].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
        axes[cluster_id].set_title(f"Grupa {cluster_id + 1}\n{hex_color}", fontsize=9)
        axes[cluster_id].axis("off")

        # Mali kvadratic boje centra ispod naslova
        axes[cluster_id].add_patch(
            plt.Rectangle((0, 0), binary_mask.shape[1] * 0.12,
                           binary_mask.shape[0] * 0.08,
                           color=c[:3], transform=axes[cluster_id].transData)
        )

    plt.suptitle(f"Binarne maske po grupama — {filepath}  (K={K})", fontsize=12)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Obrada svih slika
# ------------------------------------------------------------------
for filepath in IMAGE_FILES:
    process_image(filepath)

