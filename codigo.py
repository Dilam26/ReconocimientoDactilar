import os
import glob
import cv2

import numpy as npy
from matplotlib import pyplot as plt 

from scipy.optimize import minimize

from numpy import random

#print(cv2.__version__)

#Obtener direccion de las carpetas de las imagenes de entrenamiento
path = []
for i in os.listdir("C:/CDocumentos/V.U/Univalle/IngDeSistemas/MateriasYTemas/Semestre4/MN/trabajoFinalMN/code/huellas"): 
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

#print(MTraining)




#medio = npy.zeros(shape =(resolution,5))
medio=npy.empty([resolution, 49], dtype = float)
sum = npy.zeros(shape =(resolution,1))
for i in range(49):
	for j in range(5):
		sum[:,0] += MTraining[:,(i*5+j)]
		#medio.append(sum)
	
	#print(sum[0][0])
	
	medio[:,i] = (sum/5)[:,0]
	sum = npy.zeros(shape=(resolution,1))
print("")	
#print(MTraining[0][0])
print("")
#print(medio[:,10])













#print(MTraining[155])
#print(medio.shape)
#print(npy.shape(medio))
#print(medio[0])
#print(MTraining[0][0])

"""
#Visualización del rostro promedio
print(medio.shape)
huellaPromedio = medio[:,1].reshape(n,m)
plt.imshow(huellaPromedio,cmap ='gray')
plt.title('HuellaPromedioP1')
plt.xticks([]),plt.yticks([])
plt.show()
"""

#Para cada persona, el tamaño de U es de [40000, 5], cada cinco columnas consecutivas corresponden a una persona.
U = npy.zeros([resolution, 245], dtype = float) 

#Para cada persona, el tamaño de S es [5], cada columna corresponde a una persona.
S = npy.zeros([5, 49], dtype = float) 	

#Para cada persona, el tamaño de V es [5,5], cada cinco columnas consecutivas corresponden a una persona.
V = npy.zeros([5,245], dtype = float)  

for i in range(49):
	#U, S, V = npy.linalg.svd(MTraining[:, i*5: (i+1)*5], full_matrices = False)
	U[:,i*5:(i+1)*5], S[:,i], V[:,i*5:(i+1)*5] = npy.linalg.svd(MTraining[:, i*5: (i+1)*5], full_matrices = False)

"""
#Visualización del rostro promedio
print(U.shape)
print(S.shape)
print(V.shape)
print("")
print(S[:,16])
print(S.shape)

#print(V[:,10:15])
"""



"""
#Visualización de las "huellas base" 
huellaBase = U[:,5].reshape(n,m)
plt.imshow(huellaBase,cmap ='gray')
plt.title('HuellaBase1')
plt.xticks([]),plt.yticks([])
plt.show()
"""




mExample = npy.zeros([5, 6], dtype = float)
#print(mExample)

m1 = npy.array([  [1, 1, 1, 1, 1, 1],      
				  [2, 2, 2, 2, 2, 2], 
				  [3, 6, 3, 3, 3, 3],
				  [4, 4, 4, 4, 4, 4], 
				  [5, 5, 5, 5, 5, 5]])

"""
x0*c0+x1*c1+x2*c2
				  [[2]
				  [3]
				  [18]
				  [15]
				  [2]]
"""
#print(m1)

col0 = npy.zeros([5, 1], dtype = float)
"""
mExample[:,2] = m1[:,1]
mExample[:,2] += m1[:,0]
mExample[:,2] = mExample[:,2]/2
print(mExample)
col0 = mExample[:,2]
col0 += mExample[:,2]
print(col0)
"""
#tv = npy.array([[2], [4], [6], [1], [3]])
#print(tv)

a0 = m1[:,0]
a1 = m1[:,1]
a2 = m1[:,2]

vecPrueba = npy.array([[2], [3], [18], [15], [2]])

def objective_fn(x):
	x0 = x[0]
	x1 = x[1]
	x2 = x[2]
	dist = npy.linalg.norm(vecPrueba-(x0*a0 + x1*a1 + x2*a2 ))
	return dist

"""
print(npy.max(U))#0.9999428217733894
print(npy.min(U))#-0.030018814508134515
print(npy.max(U) - npy.min(U))#1.029961636281524
print(255.0/(npy.max(U) - npy.min(U)))#247.58203705589253


startingX = npy.array([[(random.rand()*1.029961636281524) - 0.030018814508134515], 
						[(random.rand()*1.029961636281524) - 0.030018814508134515], 
						[(random.rand()*1.029961636281524) - 0.030018814508134515]]) 
print(startingX)
"""

startingJ = npy.array([[random.randint(10)], 
						[random.randint(10)], 
						[random.randint(10)]])
print(startingJ)

result = minimize(objective_fn, startingJ, method = 'SLSQP')
print(result)


