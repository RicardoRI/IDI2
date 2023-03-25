# -*- coding: utf-8 -*-
"""
TAREA 8: K-MEANS

@author: Ricardo Ramírez Islas
"""

import pandas as pd
from sklearn.cluster import KMeans
import random

#Excel file con datos en columnas A y B
dataset = pd.read_excel('kmeans_sample.xlsx')


X = dataset.iloc[: ,[0,1]].values

#Valores de entrada
k = 2 #Kmeans
x_dato = 29 #dato x para clasificar
y_dato = 95 #dato y para clasificar

#inicializacion de los centroides
x_min, x_max = min(X[:,0]), max(X[:,0])
y_min, y_max = min(X[:,1]), max(X[:,1])

centroides_iniciales = []
for i in range(k):
    cx = random.uniform(x_min, x_max)
    cy = random.uniform(y_min, y_max)
    centroides_iniciales.append([cx, cy])
    
#algoritmo K-means
k_means = KMeans(n_clusters = k, init= centroides_iniciales, max_iter=300, n_init=10, random_state=0)
y_kmeans = k_means.fit_predict(X)

#función para clasificar una pareja de datos
def clasificar(x_dato, y_dato, k_means):
    clase = k_means.predict([[x_dato, y_dato]])
    return clase[0]

#Clasificando pareja de datos definidos
clase = clasificar(x_dato, y_dato, k_means)


#RESPUESTAS
#centroides encontrados
print("Los centroides encontrados son:")
print(k_means.cluster_centers_)


#clasificacion de datos
print("Los datos: ({}, {}) pertenece a la clase: {}".format(x_dato, y_dato, clase))