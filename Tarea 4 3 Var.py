# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 04:07:45 2023

@author: lenovo
"""
## CODIGO DOS O TRES VARIABLES
import numpy as np

def funcion_costo(x):
    x, y, z = x[0], x[1], x[2]
    return x**4 + y**4 + z**4 + x*y*z

def funcion_derivada(x):
    x, y, z = x[0], x[1], x[2]
    return np.array([4*x**3 + y*z, x*z + 4*y**3, x*y + 4*z**3 ])

def gradiente_descendente(fun, fun_der, x0, alpha, max_iter, tolerance):
    x = x0
    history = []
    J_prev = np.inf
    i = 0
    while i < max_iter:
        der = fun_der(x)
        x = x - alpha * der
        J = fun(x)
        history.append(J)
        if abs(J - J_prev) < tolerance:
            break
        J_prev = J
        i += 1
    return x, history, i+1

# VARIABLES POR EJERCICIO
x0 = np.array([1, 1, -1])
alpha = .25
max_iter = 1000
tolerance = 1e-3
x_min, J_history, iteraciones = gradiente_descendente(funcion_costo, funcion_derivada, x0, alpha, max_iter, tolerance)
print("El mínimo de la función es", J_history[-1], "y se encuentra en x =", x_min[0], " y =", x_min[1]," z =", x_min[2])
print("Con", iteraciones, "iteraciones.")