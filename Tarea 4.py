# -*- coding: utf-8 -*-
"""


@author: Ricardo Ramirez
"""
##CODIGO PARA UNA VARIABLE

import numpy as np

def funcion_costo(x):
    return x**4 - 3*x**3 + 2

def fun_derivada(x):
    return 4*x**3 - 9*x**2

def gradiente_descendente(fun, fun_der, x0, alpha, max_iter, tolerancia):
    x = x0
    hist = []
    J_prev = np.inf
    i = 0
    while i < max_iter:
        der = fun_der(x)
        x = x - alpha * der
        J = fun(x)
        hist.append(J)
        if abs(J - J_prev) < tolerancia:
            break
        J_prev = J
        i += 1
    return x, hist, i+1

# VARIABLES Y RESPUESTA PARA LOS EJERCICIOS
x0 = 2
alpha = 0.025
max_iter = 10000
tolerancia = 1e-3
xmin, local, iteraciones = gradiente_descendente(funcion_costo, fun_derivada, x0, alpha, max_iter, tolerancia)
print("El mínimo de la función es", local[-1], "y se encuentra en x =", xmin)
print("Con", iteraciones, "iteraciones.")