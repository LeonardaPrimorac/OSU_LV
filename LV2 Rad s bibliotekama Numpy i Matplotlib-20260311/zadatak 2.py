import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

print(f"Broj osoba: {data.shape[0]}")


plt.scatter(data[:, 1], data[:, 2], label='Sve osobe', color='blue')
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Visina vs masa')
plt.show()


if data.shape[0] >= 50:
    plt.scatter(data[::50, 1], data[::50, 2], label='Svaka 50. osoba', color='red')
    plt.xlabel('Visina (cm)')
    plt.ylabel('Masa (kg)')
    plt.title('Visina vs masa (Svaka 50. osoba)')
    plt.show()


print(f"Min visina: {np.min(data[:, 1])} cm")
print(f"Max visina: {np.max(data[:, 1])} cm")
print(f"Prosječna visina: {np.mean(data[:, 1]):.2f} cm")


muskarci = data[data[:, 0] == 1]
zene = data[data[:, 0] == 0]

print(f"Min visina muškaraca: {np.min(muskarci[:, 1])} cm")
print(f"Max visina muškaraca: {np.max(muskarci[:, 1])} cm")
print(f"Prosječna visina muškaraca: {np.mean(muskarci[:, 1]):.2f} cm")
print(f"Min visina žena: {np.min(zene[:, 1])} cm")
print(f"Max visina žena: {np.max(zene[:, 1])} cm")
print(f"Prosječna visina žena: {np.mean(zene[:, 1]):.2f} cm")