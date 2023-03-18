# -*- coding: utf-8 -*-
"""


@author: Ricardo Ramirez Islas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Leyendo el archivo 
dataset = pd.read_excel('PercMultAplicado.xlsx')

#Normalizando las variables de entrada para el modelo
dataset['Monto sin sesgo'] = dataset['Monto'].apply(lambda x: x ** 0.5)
dataset['Monto Norm'] = (dataset['Monto sin sesgo']-dataset['Monto sin sesgo'].mean())/dataset['Monto sin sesgo'].std()
dataset['Carga Norm'] = dataset['Mensualidad']/dataset['Ingreso mensual']
dataset['Antigüedad laboral (meses) Norm'] = dataset["Antigüedad laboral (meses)"]/max(dataset["Antigüedad laboral (meses)"])

#Convirtiendo la columna Mora (variable de salida en 1 y 0)
mora_dict = {'SI': 1, 'NO': 0}
dataset['Mora'] = dataset['Mora'].map(mora_dict)

#Variables de entrada y salida
X = dataset.iloc[:, 9:].values
y = dataset.iloc[:, 7:8].values

#Dividiendo la data en entrenamiento y prueba
X_train, X_test, y_train, y_test_true = train_test_split(X, y, test_size=0.3)

#Valores de entrada 
N = X.shape[1] #Numero de entradas
L = 6 #Neuronas escondidas
M = y.shape[1] #Numero de salidas: Mora
alfa = 0.1 #aprendizaje
iteraciones_max = 10000

#np.random.seed(1)
w_h = np.random.uniform(-1, 1, size=(L, N)) #Pesos hidden layer valores al azar
w_o = np.random.uniform(-1, 1, size=(M, L)) #Pesos output layer valores al azar

# Definiendo la función de activación sigmoid
def activacion_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def threshold(x, threshold = 0.5):
    return np.where(x >= threshold, 1, 0)

# Función de entrenamiento
for i in range(iteraciones_max):
    # Fordward
    h = activacion_sigmoid(np.dot(w_h, X_train.T))
    y = activacion_sigmoid(np.dot(w_o, h))
    
    # Error
    error = y_train.T - y
    if np.mean(np.abs(error)) < 10**-3:
        print("Entrenamiento completado")
        break
    
    #Backward
    delta_o = error * y * (1 - y)
    delta_h = np.dot(w_o.T, delta_o) * h * (1 - h)
     
    # Actualizando los pesos
    w_o += alfa * np.dot(delta_o, h.T)
    w_h += alfa * np.dot(delta_h, X_train)

# Probando la red neuronal
h_test = activacion_sigmoid(np.dot(w_h, X_test.T))
y_test = activacion_sigmoid(np.dot(w_o, h_test))
y_pred_test = threshold(y_test, threshold = 0.5)

#Calculate the accuracy
y_test_pred = np.round(y_test.T)
accuracy = np.mean(y_test_pred == y_test_true)
print("Test accuracy:", accuracy)

#print("Test results:")
#print(y_test.T) 
#print("Valores de entrada para predecir la salida")
#print(X_test)
#print("Valores de salida")
#print(y_pred_test.T)   
