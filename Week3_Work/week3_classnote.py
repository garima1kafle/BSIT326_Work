
#%%
import numpy as np

def gauss_elimination_partial_pivoting(A, b):
    n = len(b)
    AugmentedMatrix = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial Pivoting
        max_row = np.argmax(np.abs(AugmentedMatrix[i:n, i])) + i
        AugmentedMatrix[[i, max_row]] = AugmentedMatrix[[max_row, i]]
        
        for j in range(i + 1, n):
            factor = AugmentedMatrix[j, i] / AugmentedMatrix[i, i]
            AugmentedMatrix[j, i:] -= factor * AugmentedMatrix[i, i:]

    # Backward substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (AugmentedMatrix[i, -1] - np.dot(AugmentedMatrix[i, i + 1:n], x[i + 1:n])) / AugmentedMatrix[i, i]

    print("\nAugmented Matrix after elimination:\n", AugmentedMatrix)
    return x

# Example matrix
A = np.array([[3, -2, 5],
              [1, 1, -3],
              [2, 3, 1]], dtype=float)

b = np.array([3, 3, 4], dtype=float)

solution = gauss_elimination_partial_pivoting(A, b)
print("Solution:", solution)

#%%
import numpy as np

def gauss_jordan(A, b):
    n = len(b)
    AugmentedMatrix = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        # Find the row with the maximum element in the current column
        max_row = np.argmax(np.abs(AugmentedMatrix[i:n, i])) + i
        # Swap the current row with the max_row
        AugmentedMatrix[[i, max_row]] = AugmentedMatrix[[max_row, i]]

        # Make the diagonal element 1
        AugmentedMatrix[i] = AugmentedMatrix[i] / AugmentedMatrix[i, i]

        # Make the other elements in current column zero
        for j in range(n):
            if i != j:
                AugmentedMatrix[j] = AugmentedMatrix[j] - AugmentedMatrix[j, i] * AugmentedMatrix[i]

        print("\nAfter step", i+1, ":\n", AugmentedMatrix)

    return AugmentedMatrix[:, -1]  # return only the solution column

# Example usage
A = np.array([[2, -1, 1],
              [4, 3, 2],
              [1, 1, 1]], dtype=float)

b = np.array([2, 12, 5], dtype=float)

solution = gauss_jordan(A, b)
print("\nSolution:", solution)

# %%
import numpy as np

def gauss_seidel(A, b, x0, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.copy(x0)

    for k in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Converged after {k+1} iterations")
            return x

    print("Max iterations reached")
    return x

# Example usage
A = np.array([[4, -1, 0, 0],
              [-1, 4, -1, 0],
              [0, -1, 4, -1],
              [0, 0, -1, 3]], dtype=float)

b = np.array([15, 10, 10, 10], dtype=float)
x0 = np.zeros_like(b)

solution = gauss_seidel(A, b, x0)
print("Solution:", solution)

# %%
