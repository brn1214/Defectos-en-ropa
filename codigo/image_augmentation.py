import cv2
import numpy as np
import os

# Ubicación de las imagenes
folder_path = "PATH"
save_path = "PATH"

# Aumentación de las imagenes
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Cargar imagenes
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        # Aqui definimos las transformaciones *Pseudocódigo*
        
        rotar = cv2.rotar  
        trasladar = cv2.trasladar  
        escalar = cv2.escalar  
        voltear = cv2.voltear  
        recortar = cv2.recortar  
        cambiar_espacio_color = cv2.cambiar_espacio_color  
        ajustar_brillo_contraste = cv2.ajustar_brillo_contraste  
        agregar_ruido = cv2.agregar_ruido  
        desenfocar = cv2.desenfocar  

        # Almacenamiento de las imágenes
        cv2.imwrite("path", imagen_rotada)
        ...



 
