def count_chars(string):
    dic_characters = {}
    
    for i in range(len(string)):
        if string[i] in dic_characters:
            dic_characters[string[i]] += 1
        else:
            dic_characters[string[i]] = 1
    return dic_characters

if __name__=='__main__':
    print(count_chars("smiles"))