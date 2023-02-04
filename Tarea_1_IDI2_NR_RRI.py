# -*- coding: utf-8 -*-
"""

@author: Ricardo Ramírez Islas

IDI2
"""

import sympy as sp
x = sp.symbols('x')
def metodo_newton_rapson(f, x0, tol):
    
    n = 1 #contador
    df = sp.diff(f) #obteniendo la derviada
    x1 = x0 - (f.evalf(subs={x: x0}) / df.evalf(subs={x: x0}))
    
    while abs(x1-x0) > tol:
        n = n+1
        x0 = x1
        x1 = x0 - (f.evalf(subs={x: x0}) / df.evalf(subs={x: x0}))
           
    print('{} es una buena aproximación, con {} iteraciones'.format(x1, n))
      
## Ejercicio 1:
metodo_newton_rapson(x ** 3 - 2 * x ** 2 - 5, 2, 0.0001)
## Ejercicio 2
metodo_newton_rapson(x - sp.cos(x), 1, 0.0001)
## Ejercicio 3
metodo_newton_rapson(x - 0.8 - 0.2*sp.sin(x), 1, 0.0001)
## Ejercicio 4
metodo_newton_rapson(sp.ln(x-1)+sp.cos(x-1), 1.2, 0.0001)
## Ejercicio 5
metodo_newton_rapson(3*x**2 - sp.exp(x), -0.5, 0.0001)
## Ejercicio 6
metodo_newton_rapson(sp.sqrt(5) - x, 2.2, 0.0001)
## Ejercicio 7
metodo_newton_rapson(sp.ln(x**2+1)-sp.exp(0.4*x)*sp.cos(sp.pi*x), -1, 10**-6)


