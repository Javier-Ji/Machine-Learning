#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/5
# @Author  : jingwei_ji
# @Software: PyCharm
# @File    : test.py

import numpy as np


def creat_dataset(file_name, delimiter):
    dataset = np.genfromtxt(file_name, delimiter=delimiter, dtype=str)
    dataset_attributes = dataset[0, 1:-1]
    dataset_X = dataset[1:, 1:-1]
    dataset_Y = dataset[1:, -1]
    # print(dataset_attributes)
    # print(dataset_X)
    # print(dataset_Y)
    return dataset_X, dataset_Y, dataset_attributes


def calculate_information_entropy(label, instances_num=None):
    dict_label = {}
    if instances_num is None:
        instances_num = len(label)
    information_entropy = 0
    for l in label:
        if l not in dict_label.keys():
            dict_label[l] = 0
        dict_label[l] += 1
    for k in dict_label.keys():
        if dict_label[k] != 0:  # fix log2 of 0 byg
            information_entropy -= dict_label[k] / instances_num * np.log2(dict_label[k] / instances_num)

    return information_entropy


def is_continuous(sample):
    try:
        float(sample)
        return True
    except ValueError:
        return False

watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('watermelon_3', ', ')
# for i in range(len(watermelon_label)):
for j in range(len(watermelon_attributes)):
    base_entropy = calculate_information_entropy(watermelon_label)
    print('base_entropy: ', base_entropy)
    attribute_values_list = list(set(watermelon_dataset[:, j]))
    dict_attributes = {}

    # print(attribute_values_list)
    if is_continuous(attribute_values_list[0]):
        print(watermelon_attributes[j])
        attribute_values_list.sort()
        attribute_values_list_continuous = []
        for i in range(len(attribute_values_list)-1):
            attribute_values_list_continuous.append(round((float(attribute_values_list[i])+float(attribute_values_list[i+1]))/2, 3))
        print(attribute_values_list)
        print('attribute_values: ', attribute_values_list_continuous)
        for i in attribute_values_list_continuous:
            sub_label = {}
            for d in range(len(watermelon_dataset)):
                if float(watermelon_dataset[d, j]) <= i:
                    if '<=%s' % i not in sub_label.keys():
                        sub_label['<=%s' % i] = []
                    sub_label['<=%s' % i].append(watermelon_label[d])
                    # if watermelon_label[d] not in sub_label['<=%s' % i].keys():
                    #     sub_label['<=%s' % i][watermelon_label[d]] = 0
                    # sub_label['<=%s' % i][watermelon_label[d]] += 1
                else:
                    if '>%s' % i not in sub_label.keys():
                        sub_label['>%s' % i] = []
                    sub_label['>%s' % i].append(watermelon_label[d])
                    # if watermelon_label[d] not in sub_label['>%s' % i].keys():
                    #     sub_label['>%s' % i][watermelon_label[d]] = 0
                    # sub_label['>%s' % i][watermelon_label[d]] += 1
            print(sub_label)
            if watermelon_attributes[j] not in dict_attributes:
                dict_attributes[watermelon_attributes[j]] = {}
            dict_attributes[watermelon_attributes[j]][i] = base_entropy
            # print(dict_attributes)
            for s in sub_label.keys():
                # print('s', sub_label[s])

                instances_num = len(watermelon_label)
                # sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value)[1]
                # dict_attributes[watermelon_attributes[j]][i] = calculate_information_entropy(sub_label, instances_num)
                dict_attributes[watermelon_attributes[j]][i] -= len(sub_label[s]) / instances_num * calculate_information_entropy(sub_label[s])
    # else:
    #     instances_num = len(watermelon_label)
    #     sub_label = split_dataset(dataset, label, attribute_index, attribute_value)[1]
    #     dict_attributes[attributes[attribute_index]][attribute_value] = self.calculate_information_entropy(sub_label)
    #     dict_attributes[attributes[attribute_index]]['info_gain'] -= len(sub_label) / instances_num * \
    #                                                                  dict_attributes[attributes[attribute_index]][
    #                                                                      attribute_value]

                # dict_attributes[watermelon_attributes[j]][i] -= calculate_information_entropy(sub_label[s])
    print(dict_attributes)




