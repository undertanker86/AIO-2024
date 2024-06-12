import os


def read_file(file_path):
    a_file = open(file_path, 'r')

    data = a_file.read()

    a_file.close()

    return data


def word_count(data):
    data = data.lower()

    data_list = data.split()

    dic_word = {}

    for i in range(len(data_list)):
        if data_list[i] in dic_word:
            dic_word[data_list[i]] += 1
        else:
            dic_word[data_list[i]] = 1
    return dic_word


if __name__ == '__main__':
    file_path = 'D:/AIO-2024-WORK/AIO-2024/module1-week1-data_structure_exercise/assets/P1_data.txt'

    # print(word_count(data = read_file(file_path)))
    data = read_file(file_path)
    dic_word = word_count(data)
    print(dic_word['man'])
