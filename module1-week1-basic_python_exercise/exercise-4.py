import math
def factorial(number):
    if number == 0: return 1
    result = 1
    for i in range(1, number + 1):
        result *= i
    return result

def approx_sin(x, n):
    result = 0
    for i in range(n + 1):
       result += math.pow(-1, i) * (math.pow(x, 2 * i + 1) / factorial(2 * i + 1))
    return result

def approx_cos(x, n):
    result = 0
    for i in range(n + 1):
       result += math.pow(-1, i) * (math.pow(x, 2 * i) / factorial(2 * i))
    return result

def approx_sinh(x, n):
    result = 0
    for i in range(n + 1):
       result += math.pow(x, 2 * i + 1) / factorial(2 * i + 1)
    return result

def approx_cosh(x, n):
    result = 0
    for i in range(n + 1):
       result += math.pow(x, 2 * i) / factorial(2 * i)
    return result

if __name__ == "__main__":
    print(approx_sin(3.14, 10))
    print(approx_cos(3.14, 10))
    print(approx_sinh(3.14, 10))
    print(approx_cosh(3.14, 10))