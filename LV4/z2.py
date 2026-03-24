import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

data = pd.read_csv("data_C02_emission.csv")

data_encoded = pd.get_dummies(data, columns=['Fuel Type'], drop_first=True)

features = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)'
] + [col for col in data_encoded.columns if 'Fuel Type_' in col]

X = data_encoded[features]
y = data_encoded['CO2 Emissions (g/km)']

# Podjela 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linearna regresija
model = LinearRegression()
model.fit(X_train, y_train)

# Predikcija
y_pred = model.predict(X_test)

# Evaluacija
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Evaluacija ---")
print("MSE:", mse)
print("MAE:", mae)
print("R2:", r2)

# Maksimalna pogreška
errors = np.abs(y_test - y_pred)
max_error = np.max(errors)

# indeks maksimalne pogreške
max_error_index = errors.idxmax()

print("\nMaksimalna pogreška:", max_error)

# dohvat modela vozila
vehicle = data.loc[max_error_index, ['Make', 'Model']]
print("Vozilo s najvećom pogreškom:")
print(vehicle)