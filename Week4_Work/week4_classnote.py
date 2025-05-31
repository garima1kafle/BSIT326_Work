

#%%
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

#%%
import numpy as np

# Function to compute Newton's Divided Difference interpolation
def newton_divided_difference(x_values, y_values, x_value):
    n = len(x_values)

    # Create a divided difference table
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y_values

    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x_values[i + j] - x_values[i])

    # Optional: Print the divided difference table rounded to 3 decimal places
    print("Divided Difference Table:")
    print(np.round(divided_diff, 3))

    # Calculate the interpolated value
    result = divided_diff[0, 0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_value - x_values[i - 1])
        result += divided_diff[0][i] * product_term

    return result

# Example Test Case
x_values = np.array([1, 2, 3, 4, 5])       # Given x-values
y_values = np.array([1, 4, 9, 16, 20])     # Given y-values
x_value = 4.5                              # Value to interpolate

# Call the interpolation function
interpolated_value = newton_divided_difference(x_values, y_values, x_value)
print(f"The interpolated value at x = {x_value} is: {interpolated_value}")

# %%
