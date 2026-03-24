import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("data_C02_emission.csv")

features = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
]

X = data[features]
y = data['CO2 Emissions (g/km)']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# b) Scatter plot
plt.figure()
plt.scatter(X_train['Engine Size (L)'], y_train, color='blue', label='Train')
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Test')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('CO2 vs Engine Size')
plt.legend()
plt.show()

# c) Standardizacija
scaler = StandardScaler()

# Histogram prije skaliranja
plt.figure()
plt.hist(X_train['Engine Size (L)'], bins=30)
plt.title('Prije skaliranja')
plt.show()

# Skaliranje
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Histogram nakon skaliranja
plt.figure()
plt.hist(X_train_scaled[:, 0], bins=30)
plt.title('Nakon skaliranja')
plt.show()

# d) Linearna regresija
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n--- Parametri modela ---")
print("Koeficijenti:", model.coef_)
print("Presjek (bias):", model.intercept_)


# e) Predikcija
y_pred = model.predict(X_test_scaled)

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('Stvarne vrijednosti')
plt.ylabel('Predviđene vrijednosti')
plt.title('Stvarno vs Predviđeno')
plt.show()

# f) Evaluacija
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluacija ---")
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)

# g) Napomena
print("\nNapomena:")
print("Povećanjem broja ulaznih varijabli model može bolje naučiti podatke,")
print("ali postoji rizik od overfittinga. Smanjenjem broja varijabli model")
print("postaje jednostavniji, ali može izgubiti važne informacije.")
