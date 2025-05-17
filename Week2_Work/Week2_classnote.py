import numpy as np 
import matplotlib.pyplot as plt 

def bisection(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    c = a
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c

def f(x):
    return x**3 - 0.2*x - 40

root = bisection(f, 3, 4, 0.01)
print("Root:", root)


# Plotting the function
x_vals = np. linspace(0, 4, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="x**3 -0.2*x - 40") 
plt.axhline(0, color='black',linewidth=0.5) 
plt.axvline(root, color='r', linestyle='--', label=f'Root at x = {root:.2f}')
plt. xlabel('x')
plt.ylabel('f(x)')
plt. title( 'Bisection Method - Root Finding')
plt. legend()
plt. grid (True)
plt. show( )


# %%
import numpy as np 
import matplotlib.pyplot as plt 

def my_bisection(f, a, b, tol):
    if (f(a)) * (f(b)) > 0:
        raise Exception("The scalars a and b do not bound a root")
    
    m = (a + b) / 2
    
    if abs(f(m)) < tol:
        return m
    
    if f(a) * f(m) > 0:
        return my_bisection(f, m, b, tol)
    else:
        return my_bisection(f, a, m, tol)

def f(x):
    return x**4 - 2*x - 40

# Bisection method
root = my_bisection(f, 1, 3, 0.01)
print("Root:", root)

# Plotting the function
x_vals = np. linspace(0, 4, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="x**3 -0.2*x - 40") 
plt.axhline(0, color='black',linewidth=0.5) 
plt.axvline(root, color='r', linestyle='--', label=f'Root at x = {root:.2f}')
plt. xlabel('x')
plt. title( 'Bisection Method - Root Finding')
plt. legend()
plt. grid (True)
plt. show( )

