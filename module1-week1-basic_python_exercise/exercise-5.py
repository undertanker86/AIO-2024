import math
def md_nre_single_sample(y, y_hat, n, p):
    result = ((y**(1/n)) - (y_hat**(1/n)))**p
    return result

if __name__ == "__main__":
    print(md_nre_single_sample(y =50 , y_hat =49.5 , n =2 , p =1))