import numpy as np

# Original matrix A
A = np.array([[1, 2], [3, 4]])

# Perform SVD
u, s, vT = np.linalg.svd(A)

# Create a diagonal matrix Sigma with the singular values
sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(sigma, s)

# Reconstruct the matrix B
B = u.dot(sigma.dot(vT))

print("Original matrix A:")
print(A)

print("\nMatrix U:")
print(u)

print("\nSingular value:")
print(s)

print("\nMatrix sigma:")
print(sigma)

print("\nReconstructed matrix B:")
print(B)
                                                                                            
                                                                                                            
