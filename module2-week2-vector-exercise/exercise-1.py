import numpy as np


def compute_vector_length(vector):
    vector_pow = vector ** 2
    vector_sum = np.sum(vector_pow)
    result = np.sqrt(vector_sum)
    return result


def dot_product(vector1, vector2):
    mul_vector = vector1 * vector2
    result = np.sum(mul_vector)
    return result


def matrix_multi_vector(matrix, vector):
    return np.dot(matrix, vector)


def matrix_multi_matrix(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def inverse_matrix(matrix):
    det_a = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    matrix_b = np.array([[matrix[1][1], -matrix[0][1]],
                        [-matrix[1][0], matrix[0][0]]])
    if det_a == 0:
        return None
    else:
        return (1 / det_a) * matrix_b


if __name__ == "__main__":
    vector_a = np.arange(1, 6)

    matrix_a = np.arange(1, 11)

    matrix_a = matrix_a.reshape(2, 5)

    matrix_b = np.array([[2, 6], [1, 4]])

    vector = np.array([-2, 4, 9, 21])
    print(compute_vector_length(vector))

    v1 = np.array([0, 1, -1, 2])
    v2 = np.array([2, 5, 1, 0])

    print(dot_product(v1, v2))

    x = np.array([[1, 2], [3, 4]])
    k = np.array([1, 2])
    print("result \n", x.dot(k))
    print()

    m = np.array([[-1, 1, 1], [0, -4, 9]])
    v = np.array([0, 2, 1])
    result = matrix_multi_vector(m, v)
    print(result)

    print()
    # m1 = np.array([[0, 1, 2], [2, -3, 1]])
    # m2 = np. array([[1, -3], [6, 1], [0, -1]])
    # result = matrix_multi_matrix(m1, m2)
    # print(result)

    print()

    m1 = np.array([[-2, 6], [8, -4]])
    result = inverse_matrix(m1)
    print(result)

    # print(compute_vector_length(vector_a))

    # print(dot_product(vector_a, vector_a))

    # print(matrix_multi_vector(matrix_a, vector_a))

    # print(inverse_matrix(matrix_b))
