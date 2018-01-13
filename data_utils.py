import numpy as np
from common import dictionary, reverse_dictionary
import xlrd
import config


class Data(object):
    def __init__(self, max_length=200, batch_size=128, no_of_classes=1):
        self.max_length = max_length
        self.no_of_classes = no_of_classes
        self.reverse_dictionary = reverse_dictionary
        self.model_dataset_index = []
        self.batch_size = batch_size
        self.data = []

    # 模型需要数据的处理（训练和验证）

    def build_model_dataset(self, data_set_name):
        neg_data = xlrd.open_workbook('neg.xls').sheets()[0]
        pos_data = xlrd.open_workbook('pos.xls').sheets()[0]

        def data_to_index(data, label):
            for i in range(data.nrows):
                line_character_list = list(data.row_values(i)[0])
                length = min(len(line_character_list), self.max_length)
                line_dataset = np.zeros(self.max_length, dtype='int64')
                for i in range(length):
                    if line_character_list[i] in dictionary:
                        index = dictionary[line_character_list[i]]
                    else:
                        index = 0
                    line_dataset[i] = index
                self.data.append((int(label), line_dataset))

        data_to_index(neg_data, 0)
        data_to_index(pos_data, 1)
        neg_num = 1000
        pos_num = 1000
        dev_data = []
        train_data = []
        for i in range(len(self.data)):
            if (self.data[i][0] == 0) and (neg_num >= 0):
                dev_data.append(self.data[i])
                neg_num -= 1
            elif (self.data[i][0] == 1) and (pos_num >= 0):
                dev_data.append(self.data[i])
                pos_num -= 1
            else:
                train_data.append(self.data[i])
        if data_set_name == 'train':
            self.data = train_data
        else:
            self.data = dev_data

    # 数据混乱化
    def shuffle_data(self):
        np.random.shuffle(self.data)

    # 获得批处理的数据(返回的是已经index化的数据）
    def get_batch_data(self, batch_num=0):
        data_size = len(self.data)
        start_index = batch_num * self.batch_size
        end_index = data_size if self.batch_size == 0 else min(
            (batch_num + 1) * self.batch_size, data_size)

        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        classes = []
        for c, s in batch_texts:
            batch_indices.append(s)
            if self.no_of_classes > 2:
                c = int(c) - 1
                one_hot = np.eye(self.no_of_classes, dtype='int64')
                classes.append(one_hot[c])
            else:
                one_hot = np.eye(self.no_of_classes, dtype='int64')
                classes.append(one_hot[c])
        return np.asarray(batch_indices, dtype='int64'), classes

    def get_length(self):
        return len(self.data)


if __name__ == '__main__':
    exec(open('config.py').read())
    train_data = Data(max_length=config.max_length,
                      batch_size=config.batch_size, no_of_classes=config.no_of_classes)
    train_data.build_model_dataset()
    train_data.shuffle_data()
    batch_indices, classes = train_data.get_batch_data(batch_num=5)
    print(batch_indices, classes)
