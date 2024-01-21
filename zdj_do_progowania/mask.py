import matplotlib.pyplot as plt
import numpy as np
import skimage as ski


def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


img = plt.imread("bolt4.jpg")
mean = get_mean_color(img[600:800, :800])
img_mask = ((distance(img, mean) > 40) * 255).astype(np.uint8)
img_mask = ski.morphology.closing(img_mask, ski.morphology.square(9))
img_mask = ski.morphology.opening(img_mask, ski.morphology.square(5))
plt.imshow(img_mask, cmap="gray")
plt.imsave("bolt4_mask.png", img_mask, cmap="gray")
plt.show()
