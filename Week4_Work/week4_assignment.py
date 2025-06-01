import numpy as np

# Lagrange interpolation function
def lagrange_interpolation(x, y, x_point):
    total = 0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (x_point - x[j]) / (x[i] - x[j])
        total += term
    return total

# Newton interpolation function
def newton_interpolation(x, y, x_point):
    n = len(x)
    # Calculate divided difference coefficients
    coef = [y[0]]
    divided_diff = y.copy()
    for level in range(1, n):
        for i in range(n - level):
            divided_diff[i] = (divided_diff[i + 1] - divided_diff[i]) / (x[i + level] - x[i])
        coef.append(divided_diff[0])
    
    # Calculate interpolation value at x_point
    result = coef[0]
    product = 1
    for i in range(1, n):
        product *= (x_point - x[i - 1])
        result += coef[i] * product
    return result

# Dataset
times = [8, 10, 12, 14, 16, 18]
temps = [20, 22, 24, 23, 21, 19]

# Time to estimate temperature
estimate_time = 17  # 5 PM

# Calculate estimates
temp_lagrange = lagrange_interpolation(times, temps, estimate_time)
temp_newton = newton_interpolation(times, temps, estimate_time)

print(f"Estimated temperature at {estimate_time}:00 using Lagrange: {temp_lagrange:.2f}°C")
print(f"Estimated temperature at {estimate_time}:00 using Newton: {temp_newton:.2f}°C")
