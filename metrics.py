import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

# Charger l'image originale et l'image super-résolue
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_or', type=str, required=True)
    parser.add_argument('--image_sr', type=str, required=True)
    args = parser.parse_args()

image_originale = cv2.imread(args.image_or)
image_superresolue = cv2.imread(args.image_sr)


# Redimensionner l'image super-résolue pour qu'elle ait la même résolution que l'image originale
image_superresolue = cv2.resize(image_superresolue, (image_originale.shape[1], image_originale.shape[0]))

# Calculer la différence entre les deux images
difference = image_originale - image_superresolue

# Calculer le carré de la différence
difference_squared = difference ** 2

# Calculer la moyenne des carrés des différences
mse = np.mean(difference_squared)

# Calculer le PSNR
max_pixel_value = 255  # Pour une image en niveaux de gris
psnr = 20 * np.log10(max_pixel_value) - 10 * np.log10(mse)

print(f"PSNR : {psnr} dB")

print(f"SSIM : {ssim(image_originale, image_superresolue, win_size=3)}")


