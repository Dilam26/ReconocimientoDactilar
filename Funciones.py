
import numpy as np
import os
import cv2

# Obtenemos la imagen a buscar
imagen2 = cv2.imread('22.jpg')
# Sacamos los valores singulares
U2,s2,V2 = np.linalg.svd(imagen2, full_matrices=False)
# Con esto se pueden calgular los eigenvalores 
s2 = s2.reshape(600,1)
seigenval = sum(s2[:])

# Recorremos las carpetas y archivos de imagenes
for i in os.listdir('huellas'):
	for j in os.listdir(f'huellas/{i}'): 
		img1 = cv2.imread(f'huellas/{i}/{j}')
		# Obtenemos valores singulares y eigenvalores de las imagenes de la base de datos
		U,s,V = np.linalg.svd(img1, full_matrices=False)
		s = s.reshape(600,1)
		seigenval2 = sum(s[:])
		print("eigen: ", j, " == ", (seigenval2 - seigenval))
		prueba = ((seigenval2 - seigenval)/seigenval)
		if prueba <= 0:
			print("existe")
			pass
		error = (s - s2) / s
		if not error.any():
			print(f"La huella existe en la carpeta huellas/{i}/{j}")
			
		

print(f"La huella no existe ")