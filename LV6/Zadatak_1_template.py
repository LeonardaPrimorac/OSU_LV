import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ======================
# KNN K=5
# ======================
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train_n, y_train)

y_train_pred5 = knn5.predict(X_train_n)
y_test_pred5 = knn5.predict(X_test_n)

print("\nKNN (K=5)")
print("Tocnost train:", round(accuracy_score(y_train, y_train_pred5), 3))
print("Tocnost test:", round(accuracy_score(y_test, y_test_pred5), 3))

plot_decision_regions(X_train_n, y_train, classifier=knn5)
plt.title("KNN K=5")
plt.show()


# ======================
# KNN K=1
# ======================
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train_n, y_train)

y_train_pred1 = knn1.predict(X_train_n)
y_test_pred1 = knn1.predict(X_test_n)

print("\nKNN (K=1)")
print("Tocnost train:", round(accuracy_score(y_train, y_train_pred1), 3))
print("Tocnost test:", round(accuracy_score(y_test, y_test_pred1), 3))

plot_decision_regions(X_train_n, y_train, classifier=knn1)
plt.title("KNN K=1")
plt.show()


# ======================
# KNN K=100
# ======================
knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train_n, y_train)

y_train_pred100 = knn100.predict(X_train_n)
y_test_pred100 = knn100.predict(X_test_n)

print("\nKNN (K=100)")
print("Tocnost train:", round(accuracy_score(y_train, y_train_pred100), 3))
print("Tocnost test:", round(accuracy_score(y_test, y_test_pred100), 3))

plot_decision_regions(X_train_n, y_train, classifier=knn100)
plt.title("KNN K=100")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# definiraj model
knn = KNeighborsClassifier()

# raspon K vrijednosti koje testiramo
param_grid = {
    'n_neighbors': list(range(1, 51))
}

# GridSearch s unakrsnom validacijom (npr. 5-fold)
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_n, y_train)

# najbolji K
print("Najbolji K:", grid.best_params_['n_neighbors'])

# najbolja točnost (CV)
print("Najbolja CV tocnost:", round(grid.best_score_, 3))

# evaluacija na test skupu
best_knn = grid.best_estimator_
y_test_pred = best_knn.predict(X_test_n)

from sklearn.metrics import accuracy_score
print("Test tocnost:", round(accuracy_score(y_test, y_test_pred), 3))

from sklearn import svm
from sklearn.metrics import accuracy_score

# SVM s RBF kernelom
svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_n, y_train)

# Evaluacija
y_train_svm = svm_model.predict(X_train_n)
y_test_svm = svm_model.predict(X_test_n)

print("\nSVM (RBF kernel)")
print("Tocnost train:", round(accuracy_score(y_train, y_train_svm), 3))
print("Tocnost test:", round(accuracy_score(y_test, y_test_svm), 3))

# Granica odluke
plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.title("SVM (RBF kernel)")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score

# definiraj model
svm_model = svm.SVC(kernel='rbf')

# mreža hiperparametara
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

# GridSearch s 5-fold CV
grid = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_n, y_train)

# najbolji parametri
print("Najbolji parametri:", grid.best_params_)
print("Najbolja CV tocnost:", round(grid.best_score_, 3))

# test evaluacija
best_svm = grid.best_estimator_
y_test_pred = best_svm.predict(X_test_n)

print("Test tocnost:", round(accuracy_score(y_test, y_test_pred), 3))

# granica odluke
plot_decision_regions(X_train_n, y_train, classifier=best_svm)
plt.title("SVM (optimalni C i gamma)")
plt.show()