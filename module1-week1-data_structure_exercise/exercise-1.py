# Vì không đề cập đến k  có thể không hợp lệ  nên sẽ code với k là hợp lệ
def max_kernel(num_list, k):
    result_list = []
    for i in range(len(num_list)):
        if i <= len(num_list) - k:
            num_max = -1e9
            for j in range(i, i + k):
                if num_list[j] > num_max:
                    num_max = num_list[j]
            result_list.append(num_max)
                
    return result_list
                
            

if __name__=='__main__':
    print(max_kernel([3 , 4 , 5 , 1 , -44 , 5 ,10 , 12 ,33 , 1], 3))