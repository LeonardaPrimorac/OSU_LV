import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

X1_train = X_train[:, 0]
X2_train = X_train[:, 1]
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'magma', label = 'Train data')
plt.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap = 'pink', marker = 'X', label = 'Test data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

print(f'theta1..n: {LogRegression_model.coef_}, theta0: {LogRegression_model.intercept_}')
plot_decision_regions(X_train, y_train, LogRegression_model)
plt.scatter(X1_train, X2_train, c = y_train, cmap = 'Blues')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


y_test_p = LogRegression_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
print('Matrica zabune: ', cm)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()

print('Accuracy: ', accuracy_score(y_test, y_test_p))
print('Precision: ', precision_score(y_test, y_test_p))
print('Recall: ', recall_score(y_test, y_test_p))

correct = np.where(y_test_p == y_test)[0]
wrong = np.where(y_test_p != y_test)[0]
plt.scatter(X_test[correct, 0], X_test[correct, 1], c = 'green', label = 'Correct classification')
plt.scatter(X_test[wrong, 0], X_test[wrong, 1], c = 'black', label = 'Incorrect classification')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()