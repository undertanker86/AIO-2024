import numpy as np


def compute_cosine(v1, v2):
    v1_pow = v1 ** 2
    v2_pow = v2 ** 2
    v1_v2_sum = np.sum(v1 * v2)

    v1_square = np.sqrt(np.sum(v1_pow))
    v2_square = np.sqrt(np.sum(v2_pow))

    result = v1_v2_sum / (v1_square * v2_square)

    return result


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 0, 3, 0])
    result = compute_cosine(x, y)
    print(round(result, 3))
