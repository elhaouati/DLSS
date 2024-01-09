import os
import cv2
import numpy as np
from PIL import Image




def compte_pixels_noirs(image_path):
    img = Image.open(image_path)#.convert('L')  # Convertir en niveaux de gris
    grayscale_image = np.mean(img, axis=2, dtype=np.uint8)
    img_array = np.array(grayscale_image)  # Convertir l'image en tableau NumPy
    count = np.sum(img_array < 10)  # Compter les pixels valant zéro
    return count

def supprimer_images_correspondantes(image_name, directory):
    h=0
    corresponding_folders = ['Depth', 'Segmentation', 'High_resolution']
    images_name=['depth', 'Segmentation', 'HR']
    image_num = image_name.split('-')[-1].split('.')[0]


    for folder in corresponding_folders:
        corresponding_path = os.path.join(directory, folder, f"Img_{images_name[h]}-{image_num}.png")
        if os.path.isfile(corresponding_path):
            os.remove(corresponding_path)
            print(f"            Img_{images_name[h]}-{image_num}.png")
        h=h+1

def parcourir_images(directory, seuil,image_supprimes):
    optical_flow_dir = os.path.join(directory, 'OpticalFlow')
    for filename in os.listdir(optical_flow_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(optical_flow_dir, filename)
            count = compte_pixels_noirs(image_path)
            if count > seuil:
                os.remove(image_path)
                print("i deleted: ",filename)
                supprimer_images_correspondantes(filename, directory)
                image_supprimes+=1
    return image_supprimes

# Spécifiez le répertoire où se trouvent vos fichiers
repertoire = "ImagesCarla\imgs_train"
# Spécifiez le seuil
seuil_pixels = 300_000
image_supprimes=0
print("les images supprimees :",parcourir_images(repertoire, seuil_pixels,image_supprimes))
