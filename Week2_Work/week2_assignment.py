import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-0.5 * x) * (4 - x) - 2

def df(x):
    return -0.5 * np.exp(-0.5 * x) * (4 - x) - np.exp(-0.5 * x)

def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    xn = x0
    for n in range(max_iter):
        fxn = f(xn)
        if abs(fxn) < tol:
            return xn
        dfxn = df(xn)
        if dfxn == 0:
            print("Zero derivative. No solution found.")
            return None
        xn = xn - fxn / dfxn
    print("Exceeded maximum iterations.")
    return None

root = newton_raphson(f, df, x0=2)
print("Root (Newton-Raphson):", root)

# Plotting
x_vals = np.linspace(0, 6, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root, color='r', linestyle='--', label=f'Root at x = {root:.2f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Newton-Raphson Method - Root Finding')
plt.legend()
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + np.cos(x) - np.exp(-x) - 2

def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    for n in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        if abs(f_x1 - f_x0) < 1e-12:
            print("Small denominator. No solution found.")
            return None
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        if abs(f(x2)) < tol:
            return x2
        x0, x1 = x1, x2
    print("Exceeded maximum iterations.")
    return None

root = secant_method(f, x0=1, x1=2)
print("Root (Secant Method):", root)

# Plotting
x_vals = np.linspace(0, 3, 400)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals, label="f(x)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(root, color='g', linestyle='--', label=f'Root at x = {root:.2f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Secant Method - Root Finding')
plt.legend()
plt.grid(True)
plt.show()









