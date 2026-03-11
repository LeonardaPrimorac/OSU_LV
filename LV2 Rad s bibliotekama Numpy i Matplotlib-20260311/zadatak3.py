import numpy as np
import matplotlib.pyplot as plt



image = plt.imread("road.jpg")


osvijetli = np.clip(image + 90, image, 255)


plt.imshow(osvijetli)
plt.title('Posvijetljena slika')
plt.axis('off')
plt.show()


cetvrt = image[:, :image.shape[1] // 4]


plt.imshow(cetvrt)
plt.title('Četvrtina slike po širini')
plt.axis('off')
plt.show()


rotirana = np.rot90(image, k=-1)


plt.imshow(rotirana)
plt.title('Rotirana slika za 90°')
plt.axis('off')
plt.show()


obrnuta = np.flipud(image)

plt.imshow(obrnuta)
plt.title('Vertikalno zrcaljena slika')
plt.axis('off')
plt.show()