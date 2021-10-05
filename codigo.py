import os
import glob
import cv2
import numpy as npy

#Obtener direccion de las carpetas de las imagenes de entrenamiento
path = []
for i in os.listdir('huellas'): 
		path.append(os.getcwd()+"/huellas/"+i)

#Cantidad total de imagenes de entrenamiento
number = 0
for i in path:
	number += len(glob.glob(i+"/*.jpg"))

#Resolucion de las imagenes
m=200
n=200
resolution= m*n

#Definicion matriz de entrenamiento
#La funcion shape=(filas,columnas)
MTraining = npy.empty(shape=(resolution, number), dtype='float64')

#Bucle de lectura de imagenes de entrenamiento
column = 0
for dir in path:
	for filename in sorted(os.listdir(dir)):
		pathname = os.path.join(dir,filename)
		#Leer imagen en BN
		img = cv2.imread(pathname,0)
		#Convertir en array de tipo flotante y linealizarlo
		img_array = npy.array(img,dtype='float64').flatten()
		#Escribir los arreglos en columnas de la matriz de entrenamiento
		MTraining[:,column] = img_array[:]
		column += 1


#Calcular huella promedio por cada sub carpeta
medio = []
sum = 0
for i in range(49):
	for j in range(5):
		sum += npy.sum(MTraining[:,i*5+j])
	medio.append(sum/5)
	sum = 0