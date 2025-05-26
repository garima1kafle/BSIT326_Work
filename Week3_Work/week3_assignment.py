#%%
import numpy as np

def gauss_elimination_simple(A, b):
    n = len(b)
    Aug = np.hstack((A.astype(float), b.reshape(-1, 1)))  # Augmented matrix

    # Forward elimination (no pivoting)
    for i in range(n):
        pivot = Aug[i, i]
        Aug[i] = Aug[i] / pivot  # Make the pivot 1

        for j in range(i + 1, n):
            factor = Aug[j, i]
            Aug[j] = Aug[j] - factor * Aug[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Aug[i, -1] - np.dot(Aug[i, i+1:n], x[i+1:n])

    print("Augmented Matrix after elimination:\n", Aug)
    return x

# Example input
A = np.array([[2, -6, -1],
              [-3, -1, 7],
              [-8, 1, -2]], dtype=float)

b = np.array([-38, -34, -20], dtype=float)

solution = gauss_elimination_simple(A, b)
print("Solution:", solution)

# %%
import numpy as np

def gauss_jordan(A, b):
    n = len(b)
    # Form the augmented matrix [A | b]
    Aug = np.hstack((A, b.reshape(-1, 1)))

    print("Initial Augmented Matrix:\n", Aug)

    for i in range(n):
        # Make the diagonal element 1
        Aug[i] = Aug[i] / Aug[i][i]

        # Make all other elements in column i zero
        for j in range(n):
            if i != j:
                Aug[j] = Aug[j] - Aug[j][i] * Aug[i]

        print(f"\nAfter step {i+1}:\n", Aug)

    # Extract solution from last column
    return Aug[:, -1]

# Given equations
A = np.array([[2, 3, -1],
              [1, 1, 1],
              [-2, 5, 2]], dtype=float)

b = np.array([54, 6, -3], dtype=float)

solution = gauss_jordan(A, b)
print("\nFinal Solution:\n", solution)

# %%
import numpy as np

def partial_pivot(A, b):
    """Make matrix diagonally dominant by row swapping if possible"""
    n = len(b)
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
    return A, b

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=100):
    n = len(b)
    x = np.copy(x0)

    for k in range(max_iter):
        x_old = np.copy(x)
        print(f"\nIteration {k+1}:")
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
            print(f"  x[{i}] = {x[i]:.6f}")

        # Print solution after each iteration
        print(f"Solution after iteration {k+1}: {x}")

        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"\nConverged after {k+1} iterations")
            return x

    print("\nMax iterations reached")
    return x

# Original system
A = np.array([[-3, 1, 15],
              [6, -2, 1],
              [5, 10, 1]], dtype=float)

b = np.array([44, 5, 28], dtype=float)
x0 = np.zeros_like(b)

# Partial pivot to improve diagonal dominance
A, b = partial_pivot(A, b)

print("Matrix A after partial pivot:\n", A)
print("Vector b after partial pivot:\n", b)

# Run Gauss-Seidel with max 100 iterations but print all iterations
solution = gauss_seidel(A, b, x0)

print("\nFinal Solution:", solution)