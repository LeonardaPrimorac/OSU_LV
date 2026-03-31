import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from zadatak_2_template import X_train, X_test, y_train, y_test, plot_decision_regions

y_train = y_train.ravel()
y_test = y_test.ravel()

classes_train, counts_train = np.unique(y_train, return_counts=True)
classes_test, counts_test = np.unique(y_test, return_counts=True)

plt.figure()

plt.bar(classes_train - 0.2, counts_train, width=0.4, label='Train')
plt.bar(classes_test + 0.2, counts_test, width=0.4, label='Test')

plt.xlabel('Klasa (0=Adelie, 1=Chinstrap, 2=Gentoo)')
plt.ylabel('Broj uzoraka')
plt.legend()
plt.title('Raspodjela klasa')
plt.show()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Koeficijenti:\n", model.coef_)

plot_decision_regions(X_train, y_train, model)
plt.title("Granice odluke (train skup)")
plt.show()

y_pred = model.predict(X_test)

print("Matrica zabune:")
print(confusion_matrix(y_test, y_pred))

print("\nTočnost:", accuracy_score(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))