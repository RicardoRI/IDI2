# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 02:04:23 2023

@author: lenovo
"""

import numpy as np

def funcion_costo(x):
    x, y = x[0], x[1]
    return x**2 - 24*x + y**2 - 10*y

def funcion_derivada(x):
    x, y = x[0], x[1]
    return np.array([2*x - 24, 2*y - 10])

def gradiente_descendente(func, grad_func, x0, alpha, max_iter, tolerance):
    x = x0
    history = []
    J_prev = np.inf
    i = 0
    while i < max_iter:
        grad = grad_func(x)
        x = x - alpha * grad
        J = func(x)
        history.append(J)
        if abs(J - J_prev) < tolerance:
            break
        J_prev = J
        i += 1
    return x, history, i+1

# Ejemplo de uso
x0 = np.array([10, 5])
alpha = 0.5
max_iter = 1000
tolerance = 1e-3
x_min, J_history, iteraciones = gradiente_descendente(funcion_costo, funcion_derivada, x0, alpha, max_iter, tolerance)
print("El mínimo de la función es", J_history[-1], "y se encuentra en x1 =", x_min[0], "y x2 =", x_min[1])
print("Con", iteraciones, "iteraciones.")
