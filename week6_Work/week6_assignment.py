#%%
import numpy as np
import matplotlib.pyplot as plt

# Analytical solution function derived:
# dy/dt = y*(t - 1.5)
# => y(t) = y0 * exp( (t^2)/2 - 1.5*t )

def y_analytical(t):
    return np.exp( (t**2)/2 - 1.5*t )

# Euler's method function (same as sample)
def euler_method(f, x0, y0, h, x_target):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    n = int((x_target - x0) / h)
    for i in range(n):
        y = y + h * f(x, y)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# RK4 method function (not in sample, so implementing simply here)
def rk4_method(f, x0, y0, h, x_target):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    n = int((x_target - x0) / h)
    for i in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + k1*h/2)
        k3 = f(x + h/2, y + k2*h/2)
        k4 = f(x + h, y + k3*h)
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# Define differential equation
def f(t, y):
    return y * (t - 1.5)

# Initial conditions and interval
t0 = 0
y0 = 1
t_end = 2

# Analytical solution points (fine grid for smooth curve)
t_analytical = np.linspace(t0, t_end, 100)
y_analytical_vals = y_analytical(t_analytical)

# Euler with h=0.5
h1 = 0.5
t_euler1, y_euler1 = euler_method(f, t0, y0, h1, t_end)
print("Euler's method with h=0.5:")
for i in range(len(t_euler1)):
    print(f"t = {t_euler1[i]:.2f}, y = {y_euler1[i]:.4f}")

# Euler with h=0.25
h2 = 0.25
t_euler2, y_euler2 = euler_method(f, t0, y0, h2, t_end)
print("\nEuler's method with h=0.25:")
for i in range(len(t_euler2)):
    print(f"t = {t_euler2[i]:.2f}, y = {y_euler2[i]:.4f}")

# RK4 with h=0.5
t_rk4, y_rk4 = rk4_method(f, t0, y0, h1, t_end)
print("\nRK4 method with h=0.5:")
for i in range(len(t_rk4)):
    print(f"t = {t_rk4[i]:.2f}, y = {y_rk4[i]:.4f}")

# Plot all results on the same graph
plt.plot(t_analytical, y_analytical_vals, 'k-', label='Analytical')
plt.plot(t_euler1, y_euler1, 'ro-', label="Euler h=0.5")
plt.plot(t_euler2, y_euler2, 'bs-', label="Euler h=0.25")
plt.plot(t_rk4, y_rk4, 'g^-', label="RK4 h=0.5")
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Problem 1: dy/dt = y*t - 1.5*y, y(0)=1')
plt.legend()
plt.grid(True)
plt.show()







# %%
import numpy as np
import matplotlib.pyplot as plt

# Analytical solution:
# dy/dx = (1 + 2y) x
# Solve analytically (by separation or integrating factor):
# The solution is y = (1/2)(exp(x^2) - 1)

def y_analytical(x):
    return 0.5 * (np.exp(x**2) - 1)

# Euler's method function (same as sample)
def euler_method(f, x0, y0, h, x_target):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    n = int((x_target - x0) / h)
    for i in range(n):
        y = y + h * f(x, y)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# RK4 method function (same as previous)
def rk4_method(f, x0, y0, h, x_target):
    x_values = [x0]
    y_values = [y0]
    x = x0
    y = y0
    n = int((x_target - x0) / h)
    for i in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + k1*h/2)
        k3 = f(x + h/2, y + k2*h/2)
        k4 = f(x + h, y + k3*h)
        y = y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x = x + h
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)

# Differential equation function
def f(x, y):
    return (1 + 2*y)*x

# Initial condition and interval
x0 = 0
y0 = 1
x_end = 1
h = 0.25

# Analytical solution points (smooth curve)
x_analytical = np.linspace(x0, x_end, 100)
y_analytical_vals = y_analytical(x_analytical)

# Euler method
x_euler, y_euler = euler_method(f, x0, y0, h, x_end)
print("Euler's method:")
for i in range(len(x_euler)):
    print(f"x = {x_euler[i]:.2f}, y = {y_euler[i]:.4f}")

# RK4 method
x_rk4, y_rk4 = rk4_method(f, x0, y0, h, x_end)
print("\nRK4 method:")
for i in range(len(x_rk4)):
    print(f"x = {x_rk4[i]:.2f}, y = {y_rk4[i]:.4f}")

# Plot all results
plt.plot(x_analytical, y_analytical_vals, 'k-', label='Analytical')
plt.plot(x_euler, y_euler, 'ro-', label='Euler method')
plt.plot(x_rk4, y_rk4, 'bs-', label='RK4 method')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Problem 2: dy/dx = (1 + 2y)x, y(0)=1')
plt.legend()
plt.grid(True)
plt.show()




# %%
