import collections

# 字符表的维数
character_size = 3500
data_index = 0

with open('/home/wuchao/paper/sentiment_analysis/comment.txt', 'r') as fr:
    comment_text = fr.readlines()
    character_list = []
    for line in comment_text:
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
