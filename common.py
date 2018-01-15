# coding=utf-8
import collections
import re

import xlrd
import numpy as np

# 字符表的维数
character_size = 3500
data_index = 0

with open('..\comment.txt', 'rb') as fr:
    comment_text = fr.readlines()
    character_list = []
    for line in comment_text:
        line = line.decode('utf-8')
        character_list += list(line)[0:-1]

        # 将文本数据转变为数值数据（0-character_size-1)
        # 取频率最高的character_size-1个字符组成字符表，之外的都当做未知字符，编号为0

count = [['UNK', -1]]
count.extend(collections.Counter(
    character_list).most_common(character_size - 1))
dictionary = {}
for word, _ in count:
    dictionary[word] = len(dictionary)
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


# 建立word2vec的数据集，包括词频


def build_word2vec_dataset():
    data = []
    unk_count = 0
    for word in character_list:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    return data, count, dictionary, reverse_dictionary


def load_data_set():
    neg_data_set = []
    pos_data_set = []
    neg_data = xlrd.open_workbook('neg.xls').sheets()[0]
    pos_data = xlrd.open_workbook('pos.xls').sheets()[0]
    text_dict = {}
    for i in range(neg_data.nrows):
        text = neg_data.row_values(i)[0]
        if text not in text_dict:
            text_dict[text] = 1
        else:
            continue
        neg_data_set.append([0, text])

    for i in range(pos_data.nrows):
        text = pos_data.row_values(i)[0]
        if text not in text_dict:
            text_dict[text] = 1
        else:
            continue
        pos_data_set.append([1, text])

    with open('comment_score.txt', 'rb') as fr:
        comment_text = fr.readlines()
        for line in comment_text:
            line = line.decode('utf-8')[0:-1]
            try:
                star, comment = re.split(r"\t+", line)
            except:
                continue
            if comment not in text_dict:
                text_dict[comment] = 1
            else:
                continue
            if int(star) == 5:
                pos_data_set.append([1, comment])
            elif int(star) == 1:
                neg_data_set.append([0, comment])
            else:
                continue
    np.random.shuffle(pos_data_set)
    np.random.shuffle(neg_data_set)
    num = min(len(pos_data_set), len(neg_data_set))
    neg_data_set = neg_data_set[0:num]
    pos_data_set = pos_data_set[0:num]
    np.random.shuffle(pos_data_set)
    np.random.shuffle(neg_data_set)
    dev_proportion = 0.1
    dev_data = neg_data_set[0:int(
        num * dev_proportion)] + pos_data_set[0:int(num * dev_proportion)]
    train_data = neg_data_set[int(
        num * dev_proportion):] + pos_data_set[int(num * dev_proportion):]
    return train_data, dev_data
