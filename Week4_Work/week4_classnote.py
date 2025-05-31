import numpy as np

# Function to compute Lagrange interpolation
def lagrange_interpolation(x_values, y_values, x_value):
    n = len(x_values)
    result = 0
    for i in range(n):
        # Calculate Lagrange basis polynomial L_i(x)
        term = y_values[i]
        for j in range(n):
            if j != i:
                term = term * (x_value - x_values[j]) / (x_values[i] - x_values[j])
        result = result + term
    return result

# Example Test Case
x_values = np.array([1, 2, 3, 4, 5])        # Given x-values
y_values = np.array([1, 4, 9, 16, 20])      # Given y-values

# Value to interpolate
x_value = 4.5

# Call the interpolation function
interpolated_value = lagrange_interpolation(x_values, y_values, x_value)
print(f"The interpolated value at x = {x_value} is: {interpolated_value}")
