# -*- coding: utf-8 -*-
"""


@author: Ricardo Ramirez Islas
"""

import autograd as ad
from autograd import grad, jacobian
import autograd.numpy as np


#PARA 3 VARIABLES



jac_func1 = jacobian(func1)
jac_func2 = jacobian(func2)
jac_func3 = jacobian(func3)


i = 0
error = 100
tol = 0.0001
M = 3
N = 3



while np.any(abs(error) > tol):
    fun_evaluate = np.array([func1(x_0), func2(x_0), func3(x_0)]).reshape(M,1)
    flat_x_0 = x_0.flatten()
    jac = np.array([jac_func1(flat_x_0), jac_func2(flat_x_0), jac_func3(flat_x_0)])
    jac = jac.reshape(N,M)
    x_new = x_0 - np.linalg.inv(jac)@fun_evaluate
    error = x_new -x_0
    x_0 = x_new
    i = i + 1
    
print("la solución es:")
print(x_new)
print("Las iteraciones fueron: ", i)

#%%
#PARA 2 VARIABLES

jac_func1 = jacobian(func1)
jac_func2 = jacobian(func2)

i = 0
error = 100
tol = 0.0001
M = 2
N = 2


while np.any(abs(error) > tol):
    fun_evaluate = np.array([func1(x_0), func2(x_0)]).reshape(M,1)
    flat_x_0 = x_0.flatten()
    jac = np.array([jac_func1(flat_x_0), jac_func2(flat_x_0)])
    jac = jac.reshape(N,M)
    x_new = x_0 - np.linalg.inv(jac)@fun_evaluate
    error = x_new -x_0
    x_0 = x_new
    i = i + 1
    
print("La solución es:")
print(x_new)
print("Las iteraciones fueron: ", i)
    
#%%   
    
# Ejercicio 1:
func1 = lambda x: (x[0]**2) + x[1] - 1
func2 = lambda x: x[0] - 2*(x[1]**2)
x_0 = np.array([1,1],dtype=float).reshape(N,1)
x_0 = np.array([1,-1],dtype=float).reshape(N,1)
    
# Ejercicio 2
func1 = lambda x: (x[0]**2) - 10*x[0] + x[1]**2 + 5
func2 = lambda x: (x[0])*x[1]**2 + x[0] - 10*x[1] + 8
x_0 = np.array([2,4],dtype=float).reshape(N,1)
x_0 = np.array([1,1],dtype=float).reshape(N,1)


# Ejercicio 3
func1 = lambda x: x[0]*np.sin(x[1]) - 1
func2 = lambda x: x[0]**2 + x[1]**2 - 4
x_0 = np.array([2,0.5],dtype=float).reshape(N,1)
x_0 = np.array([1,2],dtype=float).reshape(N,1)
x_0 = np.array([-2,-0.5],dtype=float).reshape(N,1)
x_0 = np.array([-1,-2],dtype=float).reshape(N,1)

# Ejercicio 4
func1 = lambda x: (x[1]**2)*(np.log(x[0])) - 3
func2 = lambda x: x[1] - x[0]**2
x_0 = np.array([1.5,2.5],dtype=float).reshape(N,1)

# Ejercicio 5
func1 = lambda x: x[0] + x[1] - x[2] + 2
func2 = lambda x: x[0]**2 + x[1] 
func3 = lambda x: x[2] - x[1]**2 - 1
x_0 = np.array([1,1,1],dtype=float).reshape(N,1)
x_0 = np.array([1,-1,2],dtype=float).reshape(N,1)


