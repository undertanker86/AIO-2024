import numpy as np
import math


def solve_quadratic_equation(a, b, c):
    delta = b**2 - 4*a*c
    x1 = (-b + math.sqrt(delta)) / (2*a)
    x2 = (-b - math.sqrt(delta)) / (2*a)
    return x1, x2


def compute_eigenvalues_eigenvectors(matrix):
    a = 1
    b = -(matrix[0, 0] + matrix[1, 1])
    c = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

    z1, z2 = solve_quadratic_equation(a, b, c)

    eigenvalues = [z1, z2]
    eigenvectors = []

    for eigenvalue in eigenvalues:

        A_lambda_I = matrix - np.eye(matrix.shape[0]) * eigenvalue

        u, s, vh = np.linalg.svd(A_lambda_I)
        null_space = np.compress(s <= 1e-10, vh, axis=0)
        eigenvector = null_space[0]

        eigenvector = eigenvector / np.linalg.norm(eigenvector)

        eigenvectors.append(eigenvector)

    return eigenvalues, np.array(eigenvectors).T


if __name__ == "__main__":
    matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(matrix)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)
