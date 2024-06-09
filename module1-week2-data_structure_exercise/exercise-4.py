def levenshtein_distance(string_source, string_target):
    
    length_string_source = len(string_source)
    length_string_target = len(string_target)
    
    string_source = "#" + string_source
    string_target = "#" + string_target
    
    D = [[0 for _ in range(length_string_target + 1)] for _ in range(length_string_source + 1)]
    
    D[0][0] = 0
    
    for i in range(1, length_string_source + 1):
        D[i][0] = D[i-1][0] + 1
        
    for i in range(1, length_string_target + 1):
        D[0][i] = D[0][i-1] + 1
        
    for i in range(1, length_string_source + 1):
        for j in range(1, length_string_target + 1):
            if string_source[i] == string_target[j]:
                D[i][j] = min(D[i-1][j] +1, D[i][j-1] +1, D[i-1][j-1] + 0)
            else:
                
                D[i][j] = min(D[i-1][j] +1, D[i][j-1] +1, D[i-1][j-1] + 1)
                
    return D[length_string_source][length_string_target]

if __name__=='__main__':
    print(levenshtein_distance("hola", "hello"))