import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image # Ispravan import za noviji Keras

# 1. Učitaj model
model = load_model('model_zadatak_1.keras')

# 2. Učitaj sliku i pretvori u grayscale, dimenzije 28x28
img_path = 'test.png' # Prilagodi ime datoteke prema zadatku
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')

# 3. Pretvori u polje i normaliziraj
img_array = image.img_to_array(img)
img_array = img_array.astype('float32') / 255

img_array = 1 - img_array 

# 4. PRIPREMA DIMENZIJA: Model očekuje (batch_size, height, width, channels)
# Naša slika je trenutno (28, 28, 1), moramo dodati batch dimenziju da bude (1, 28, 28, 1)
img_processed = np.expand_dims(img_array, axis=0)

# 5. Predikcija
prediction = model.predict(img_processed)
predicted_label = np.argmax(prediction)

# 6. Ispis rezultata u terminal (traženo u zadatku!)
print(f'Rezultat klasifikacije: Slika je znamenka {predicted_label}')

# 7. Vizualizacija
plt.imshow(img_array, cmap='gray')
plt.title(f'Predviđena oznaka: {predicted_label}')
plt.show()
