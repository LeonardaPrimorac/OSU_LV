import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers=4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

# 1. Generiranje podataka (mijenjajte drugi parametar od 1 do 5)
n_samples = 500
flag = 1
X = generate_data(n_samples, flag)

# 2. Primjena K-Means algoritma
# Promijenite n_clusters (K) ovisno o podacima
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, random_state=0)
y_km = km.fit_predict(X)

# 3. Prikaz rezultata
plt.figure(figsize=(8, 6))
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3')
# Ako imate K=4, otkomentirajte liniju ispod:
# plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='gray', marker='d', edgecolor='black', label='cluster 4')

# Prikaz centara grupa
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', edgecolor='black', label='centroids')

plt.title(f'K-means grupiranje (flagc={flag})')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()