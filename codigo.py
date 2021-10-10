import os
import glob
import cv2
import numpy as npy
from matplotlib import pyplot as plt 
from scipy.optimize import minimize
from numpy import random

#Definicion de imagen de prueba
img = cv2.imread("11.jpg",0)
img_array = npy.array(img,dtype='float64').flatten()
testImageVec = img_array[:]

#Obtener direccion de las carpetas de las imagenes de entrenamiento
path = []
for i in os.listdir("huellas"): 
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

#Definicion de matriz de promedios
medio=npy.empty([resolution, 49], dtype = float)
sum = npy.zeros(shape =(resolution,1))

#Bucle para calcular los promedios de las 5 huellas por persona
for i in range(49):
	for j in range(5):
		sum[:,0] += MTraining[:,(i*5+j)]
		
	medio[:,i] = (sum/5)[:,0]
	sum = npy.zeros(shape=(resolution,1))

#SVD
#Para cada persona, el tamaño de U es de [40000, 5], cada cinco columnas consecutivas corresponden a una persona.
U = npy.zeros([resolution, 245], dtype = float) 

#Para cada persona, el tamaño de S es [5], cada columna corresponde a una persona.
S = npy.zeros([5, 49], dtype = float) 	

#Para cada persona, el tamaño de V es [5,5], cada cinco columnas consecutivas corresponden a una persona.
V = npy.zeros([5,245], dtype = float)  

for i in range(49):
	#U, S, V = npy.linalg.svd(MTraining[:, i*5: (i+1)*5], full_matrices = False)
	U[:,i*5:(i+1)*5], S[:,i], V[:,i*5:(i+1)*5] = npy.linalg.svd(MTraining[:, i*5: (i+1)*5], full_matrices = False)



startingX = npy.array([[(random.rand()*1.029961636281524) - 0.030018814508134515], 
						[(random.rand()*1.029961636281524) - 0.030018814508134515], 
						[(random.rand()*1.029961636281524) - 0.030018814508134515],
						[(random.rand()*1.029961636281524) - 0.030018814508134515],
						[(random.rand()*1.029961636281524) - 0.030018814508134515]]) 


def optimalRepresentation(startingX):
	x0 = startingX[0]
	x1 = startingX[1]
	x2 = startingX[2]
	x3 = startingX[3]
	x4 = startingX[4]
	dist = npy.linalg.norm(testImageVec-(x0*basis[:,0] + x1*basis[:,1] + x2*basis[:,2] + x3*basis[:,3] 
							+ x4*basis[:,4]))
	return dist

acceso = False
for i in range(49):
	basis = U[:,(i*5):(i+1)*5]
	result = minimize(optimalRepresentation, startingX, method = 'SLSQP')
	if (result.fun < 1):
		acceso = True
									
if (acceso):
	print('**************')
	print('¡¡¡¡EXISTE!!!!')
	print('**************')
else:
	print('*****************')
	print('¡¡¡¡NO EXISTE!!!!')
	print('*****************')