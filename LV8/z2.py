import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import load_model

# 1. Učitaj model
model = load_model('model_zadatak_1.keras') # Pazi da se ime slaže s onim kako si ga spremila

# 2. Učitaj podatke (treba nam samo testni skup)
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# 3. Pripremi podatke (IDENTIČNO kao kod treniranja)
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# 4. Izvrši predikciju na SKALIRANIM podacima
predictions = model.predict(x_test_s)
y_test_p = np.argmax(predictions, axis=1)

# 5. Pronađi i prikaži nekoliko loše klasificiranih slika
brojac = 0
for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]: # Ako se stvarna oznaka ne slaže s predviđenom
        plt.imshow(x_test[i], cmap='gray') # Prikazujemo originalnu sliku
        plt.title(f'Stvarna: {y_test[i]}, Predviđena: {y_test_p[i]}')
        plt.show()
        
        brojac += 1
        if brojac == 5: # Zaustavi se nakon 5 prikazanih slika da ne otvaraš previše prozora
            break