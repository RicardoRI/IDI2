# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 02:09:55 2023

@author: lenovo
"""

import sympy as sp

# Definimos las variables y la función a derivar
x, y, z = sp.symbols('x y z')
f = x**2 - 24*x + y**2 - 10*y

# Derivamos parcialmente con respecto a x1
df_dx = sp.diff(f, x)
print("La derivada parcial de f con respecto a x es:", df_dx)

# Derivamos parcialmente con respecto a x2
df_dy = sp.diff(f, y)
print("La derivada parcial de f con respecto a y es:", df_dy)

# Derivamos parcialmente con respecto a x2
df_dz = sp.diff(f, z)
print("La derivada parcial de f con respecto a z es:", df_dz)