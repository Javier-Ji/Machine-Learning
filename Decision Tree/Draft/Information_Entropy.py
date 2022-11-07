#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/10/29
# @Author  : jingwei_ji
# @Software: PyCharm
# @File    : Information_Entropy.py


import numpy as np
from sklearn import tree
import time


# =============================================================================
# Create dataset from the txt file
# =============================================================================
def creat_dataset(file_name, delimiter):
    dataset = np.genfromtxt(file_name, delimiter=delimiter, dtype=str)
    dataset_attributes = dataset[0, 1:-1]
    dataset_X = dataset[1:, 1:-1]
    dataset_Y = dataset[1:, -1]
    # print(dataset_attributes)
    # print(dataset_X)
    # print(dataset_Y)
    return dataset_X, dataset_Y, dataset_attributes


# def Tree_Generate(dataset, label, attributes):
class TreeNode:
    def __init__(self, instance_id=None, children=None, entropy=0, depth=0):
        if children is None:
            children = {}
        # if instance_id is None:
        #     instance_id = {}
        self._split_attribute = None            # 结点的属性名
        self._split_attribute_index = None
        self._instances_id = {}       # 结点中
        self.__attribute_value = None
        self.__entropy = entropy                # If use ID3 or C4.5 to build the decision tree
        # self.gini = 0                   # If use CART to build the decision tree
        self._children = children
        self.__label = None
        self.__depth = depth
        self.__details = {}
        self.__is_leaf = None
        # self.label = {}

    def set_attribute_value(self, attribute_value):
        self.__attribute_value = attribute_value

    @property
    def get_attribute_value(self):
        return self.__attribute_value

    def set_instances_id(self, attribute_value, instances_id):
        if attribute_value not in self._instances_id.keys():
            self._instances_id[attribute_value] = instances_id

    def set_node(self, split_attribute=None, split_attribute_index=None, instances_id=None, details=None, depth=0):
        if split_attribute is not None:
            self._split_attribute = split_attribute
        if split_attribute_index is not None:
            self._split_attribute_index = split_attribute_index
        if instances_id is not None:
            self._instances_id = instances_id
        if details is not None:
            self.__details = details
        if depth != 0:
            self.__depth = depth

    def get_node(self):
        return self._split_attribute, self.__details, self.__depth


    def set_split_attribute(self, attribute):
        self._split_attribute = attribute

    @property
    def get_split_attribute(self):
        return self._split_attribute

    def set_split_attribute_index(self, attribute_index):
        self._split_attribute_index = attribute_index

    @property
    def get_split_attribute_index(self):
        return self._split_attribute_index

    @property
    def get_instances_id(self):
        return self._instances_id

    def set_entropy(self, entropy):
        self.__entropy = entropy

    def get_entropy(self):
        return self.__entropy

    def set_children(self, attribute_value, node):
        # if (attribute_value and node) is not None:
        self._children[attribute_value] = node

    def get_children(self, key=None):
        # if key is None:
        #     return
        return self._children[key]

    def get_children_keys(self):
        return self._children.keys()

    def set_label(self, label):
        self.__label = label

    def get_label(self):
        return self.__label

    def set_leaf(self):
        self.__is_leaf = 'Y'

    def whether_leaf(self):
        if self.__is_leaf == 'Y':
            return True
        return False

    def print_tree(self):
        print('attribute: ', self._split_attribute, '\nattribute index: ', self._split_attribute_index,
              '\nattribute value: ', self.__attribute_value,
              '\nis_leaf: : ', self.__is_leaf, '\ninstances_id: : ', self._instances_id,
              '\ndetails: ', self.__details, '\ndepth: ', self.__depth, '\nentropy: ', self.__entropy,
              '\nlabel: ', self.__label, '\nchildren:', self._children)

    # @property
    def set_depth(self, depth):
        self.__depth = depth
        # return self.__depth

    def get_depth(self):
        return self.__depth


class DecisionTreeGenerate:
    def __init__(self, max_depth=10):
        self.__root = TreeNode()
        self.max_depth = max_depth
        self.dataset = None
        self.label = None
        self.attributes = None
        self.continuous_attributes = {}

    def print_root(self):
        queue = [self.__root]
        blank = ' '
        # split_attribute = ''
        while queue:
            node = queue.pop()
            if node.get_attribute_value is None:
                print('***************** Root *****************')
                node.print_tree()
                # print('%s' % node.get_split_attribute, end='')
            else:
                # print(' --(%s)--> ' % node.get_attribute_value, end='')
                # if node.whether_leaf():
                #     print(node.get_label())
                # else:
                #     print('%s' % node.get_split_attribute, end='')
                print('***************** %s *****************' % node.get_attribute_value)
                node.print_tree()
            # print('(%s)--> ' % node.get_attribute_value, end='')
            if not node.whether_leaf():
                # split_attribute = node.get_split_attribute
                # print('--> ', end='')
                for i in node.get_children_keys():
                    queue.append(node.get_children(i))
            # else:
            #     print(node.get_label())

    # def draw(self, node, v):
    #     if node.get_attribute_value is None:
    #         print('%s' % node.get_split_attribute, end='')
    #         v += len(node.get_split_attribute)
    #     else:
    #         print(' --(%s)--> ' % node.get_attribute_value, end='')
    #         if node.whether_leaf():
    #             print(node.get_label())
    #             v += len(' --(%s)--> ' % node.get_attribute_value)
    #             # v += 10
    #             # print(' ' * v + '|')
    #         else:
    #             print('%s' % node.get_split_attribute, end='')
    #             v += len(node.get_split_attribute)
    #     if not node.whether_leaf():
    #         for i in node.get_children_keys():
    #             # print(' ' * v + '|')
    #
    #             self.draw(node.get_children(i), v)
    #             v += len(' --(%s)--> ' % node.get_attribute_value)
    #             print(' ' * v + ' '*len(node.get_split_attribute) + '|')


    def fit(self, dataset, label, attributes):
        self.dataset = dataset
        self.label = label
        self.attributes = attributes
        self.tree_generate_new(self.__root)
        self.print_root()
        # self.draw(self.__root, 0)

    def tree_generate_new(self, node, attribute_value=None, parent_node=None):
        sub_dataset = []
        sub_label = []
        sub_instance_id = []
        if node != self.__root:
            parent_node.print_tree()
            sub_instance_id = parent_node.get_instances_id[attribute_value]
            for i in sub_instance_id:
                sub_dataset.append(list(self.dataset[i]))
                sub_label.append(self.label[i])
            sub_dataset = np.array(sub_dataset)
        else:
            sub_dataset = self.dataset
            sub_label = self.label
        attribute_index, info_gain, continuous_value = self.ID3_choose_optimal_attribute(sub_dataset, sub_label, self.attributes)
        node.set_entropy(self.calculate_information_entropy(sub_label))
        # is_continuous = False
        # if self.attributes[attribute_index] in self.continuous_attributes.keys():
        #     keys = self.continuous_attributes[self.attributes[attribute_index]]
        #     is_continuous = True
        # else:
        # print(continuous_value)
        if not continuous_value:
            print('zouzhe1')

            keys = list(set(self.dataset[:, attribute_index]))
            # instances_id = {}
            # instances_label = {}
            details = {}
            most_node_label = {}
            for i in keys:
                split_sub_dataset, split_sub_label, split_sub_dataset_id = self.split_dataset(sub_dataset,
                                                                                              sub_label,
                                                                                              attribute_index,
                                                                                              i,
                                                                                              sub_instance_id)
                # instances_id[i] = split_sub_dataset_id
                # instances_label[i] = split_sub_label
                if i not in details.keys():
                    details[i] = {}
                details[i]['sub_dataset'] = split_sub_dataset
                details[i]['sub_label'] = split_sub_label
                node.set_instances_id(i, split_sub_dataset_id)
                deduplication_label = set(split_sub_label)
                depth = node.get_depth()+1
                if depth <= self.max_depth:
                    new_node = TreeNode()
                    new_node.set_node(split_attribute=self.attributes[attribute_index],
                                      split_attribute_index=attribute_index,
                                      details={i: details[i]}, depth=depth)
                    new_node.set_attribute_value(i)

                    if len(deduplication_label) == 1:
                        # new_node.set_attribute_value(i)
                        print('zouzhe8')

                        current_node_label = deduplication_label.pop()
                        new_node.set_leaf()
                        new_node.set_label(current_node_label)
                    elif len(deduplication_label) == 0 or depth == self.max_depth:
                        # new_node.set_attribute_value(i)
                        print('zouzhe8')

                        for j in sub_label:
                            if j not in most_node_label.keys():
                                most_node_label[j] = 0
                            most_node_label[j] += 1
                        # print('most_node_label: ', most_node_label)
                        most_frequent_label = max(most_node_label, key=most_node_label.get)
                        new_node.set_leaf()
                        new_node.set_label(most_frequent_label)
                    else:
                        print('zouzhe8')
                        self.tree_generate_new(new_node, i, node)
                    node.set_children(i, new_node)
                details['info_gain'] = info_gain
                node.set_node(self.attributes[attribute_index], attribute_index, details=details)
        else:

            print('zouzhe')
            # pass
            details = {}
            most_node_label = {}
            split_sub_dataset, split_sub_label, split_sub_dataset_id = {}, {}, {}
            # for i in sub_dataset:
            for d in range(len(sub_dataset)):
                # print(sub_dataset[d, attribute_index])
                # print(sub_dataset)
                # print(continuous_value)
                if float(sub_dataset[d, attribute_index]) <= continuous_value:
                    if '<=%s' % continuous_value not in split_sub_dataset.keys():
                        split_sub_dataset['<=%s' % continuous_value] = []
                        split_sub_label['<=%s' % continuous_value] = []
                        split_sub_dataset_id['<=%s' % continuous_value] = []
                        details['<=%s' % continuous_value] = {}
                    split_sub_dataset['<=%s' % continuous_value].append(list(sub_dataset[d, :]))
                    split_sub_label['<=%s' % continuous_value].append(sub_label[d])
                    split_sub_dataset_id['<=%s' % continuous_value].append(d)
                    # if '<=%s' % continuous_value not in details.keys():
                    # details['<=%s' % continuous_value]['sub_dataset'] = split_sub_dataset[]
                    # sub_label['<=%s' % continuous_value].append(label[d])
                    # if watermelon_label[d] not in sub_label['<=%s' % i].keys():
                    #     sub_label['<=%s' % i][watermelon_label[d]] = 0
                    # sub_label['<=%s' % i][watermelon_label[d]] += 1
                else:
                    if '>%s' % continuous_value not in split_sub_dataset.keys():
                        # sub_label['>%s' % continuous_value] = []
                        split_sub_dataset['>%s' % continuous_value] = []
                        split_sub_label['>%s' % continuous_value] = []
                        split_sub_dataset_id['>%s' % continuous_value] = []
                        details['>%s' % continuous_value] = {}
                    # sub_label['>%s' % continuous_value].append(label[d])
                    split_sub_dataset['>%s' % continuous_value].append(list(sub_dataset[d, :]))
                    split_sub_label['>%s' % continuous_value].append(sub_label[d])
                    split_sub_dataset_id['>%s' % continuous_value].append(d)
            split_sub_dataset['<=%s' % continuous_value] = np.array(split_sub_dataset['<=%s' % continuous_value])
            split_sub_dataset['>%s' % continuous_value] = np.array(split_sub_dataset['>%s' % continuous_value])
            for i in split_sub_dataset.keys():
                details[i]['sub_dataset'] = split_sub_dataset[i]
                details[i]['sub_label'] = split_sub_label[i]
                node.set_instances_id(i, split_sub_dataset_id[i])
                new_node = TreeNode()
                new_node.set_node(split_attribute=self.attributes[attribute_index],
                                  split_attribute_index=attribute_index,
                                  details={i: details[i]})
                new_node.set_attribute_value(i)
                deduplication_label = set(split_sub_label[i])
                if len(deduplication_label) == 1:
                    # new_node.set_attribute_value(i)

                    current_node_label = deduplication_label.pop()
                    new_node.set_leaf()
                    new_node.set_label(current_node_label)
                elif len(deduplication_label) == 0:
                    # new_node.set_attribute_value(i)

                    for j in sub_label:
                        if j not in most_node_label.keys():
                            most_node_label[j] = 0
                        most_node_label[j] += 1
                    # print('most_node_label: ', most_node_label)
                    most_frequent_label = max(most_node_label, key=most_node_label.get)
                    new_node.set_leaf()
                    new_node.set_label(most_frequent_label)
                else:
                    print('zouzhe2')
                    self.tree_generate_new(new_node, i, node)
                node.set_children(i, new_node)
            node.set_node(self.attributes[attribute_index], attribute_index, details=details)



        # details['info_gain'] = info_gain
        # node.set_node(self.attributes[attribute_index], attribute_index, details=details)


    # =============================================================================
    # Calculate the information entropy
    # info_entropy(D) = -\sum_{k=1}^{the number of classes}{p_k*\log_2{p_k}}
    #
    # NOTE: 信息熵越小，纯度越高
    # =============================================================================
    @staticmethod
    def calculate_information_entropy(label):
        dict_label = {}
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

    # =============================================================================
    # Calculate the gini
    # gini = \sum_{k=1}^{number of classes}{p_k * (1-p_k)}
    #      = 1 - \sum_{k=1}^{number of classes}{p_k^2}
    #
    # NOTE: 基尼值越小，纯度越高
    # =============================================================================
    @staticmethod
    def calculate_gini(label):
        dict_label = {}
        instances_num = len(label)
        gini = 0
        for l in label:
            if l not in dict_label.keys():
                dict_label[l] = 0
            dict_label[l] += 1
        for k in dict_label.keys():
            p_k = dict_label[k] / instances_num
            p_not_k = 1 - p_k
            gini += p_k * p_not_k
        return gini

    # =============================================================================
    # Split dataset based on the attribute value to get sub label
    # =============================================================================
    @staticmethod
    def split_dataset(dataset, label, attribute_index, split_attribute_value, dataset_index=None):
        if dataset_index is None:
            dataset_index = []
        rest_label = []
        rest_dataset = []
        rest_dataset_id = []
        for i in range(len(label)):
            if dataset[i, attribute_index] == split_attribute_value:
                if len(dataset_index) == 0:

                    rest_label.append(label[i])
                    # instance = []
                    # instance.extend(list(dataset[i, :]))
                    rest_dataset_id.append(i)
                    # instance.extend(list(dataset[i, :attribute_index]))
                    # instance.extend(list(dataset[i, attribute_index + 1:]))
                    rest_dataset.append(list(dataset[i, :]))
                    # rest_dataset.append(instance)
                else:
                    rest_label.append(label[i])
                    rest_dataset_id.append(dataset_index[i])
                    rest_dataset.append(list(dataset[i, :]))
        rest_dataset = np.array(rest_dataset)
        # rest_dataset_id = instance_id
        return rest_dataset, rest_label, rest_dataset_id

    def calculate_information_gain(self, dataset, label, attributes, attribute_index,
                                   attribute_value, dict_attributes):
        instances_num = len(label)
        sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value)[1]
        dict_attributes[attributes[attribute_index]][attribute_value] = self.calculate_information_entropy(sub_label)
        dict_attributes[attributes[attribute_index]]['info_gain'] -= len(sub_label) / instances_num * \
                                                                     dict_attributes[attributes[attribute_index]][
                                                                         attribute_value]

        return sub_label, dict_attributes

    def is_continuous(self, sample):
        try:
            float(sample)
            return True
        except ValueError:
            return False


    # =============================================================================
    # Choose the optimal attribute to split the dataset using the information gain based on ID3
    # The information gain prefers attributes with more possible values!
    # info_gain(D,attribute)=info_entropy(D)-\sum_{v=1}^{number of attribute values}{|D^{v}|/|D| * info_entropy(D^{v})}
    #
    # NOTE: the larger the information gain, the best attribute to select
    #       信息增益越大，意味着使用该属性划分所获得的"纯度提升"越大，所以选该属性进行划分。
    # =============================================================================
    def ID3_choose_optimal_attribute(self, dataset, label, attributes):
        base_entropy = self.calculate_information_entropy(label)
        best_attribute_information_gain = 0
        best_split_attribute_index = -1
        dict_attributes = {}
        continuous_value = []
        max_value = False
        for attribute_index in range(len(attributes)):
            """
            Create attributes dictionary which contains:
            1) 'info_gain': the information gain;
            2) 'attribute value': the information entropy of the attribute value
            """
            dict_attributes[attributes[attribute_index]] = {}
            # dict_attributes[attributes[attribute_index]]['info_gain'] = base_entropy
            attribute_values_list = list(set(dataset[:, attribute_index]))

            if self.is_continuous(attribute_values_list[0]):
                # print(watermelon_attributes[j])
                attribute_values_list.sort()
                attribute_values_list_continuous = []
                for i in range(len(attribute_values_list) - 1):
                    attribute_values_list_continuous.append(
                        round((float(attribute_values_list[i]) + float(attribute_values_list[i + 1])) / 2, 3))
                # print(attribute_values_list)
                # print('attribute_values: ', attribute_values_list_continuous)
                for i in attribute_values_list_continuous:
                    sub_label = {}
                    for d in range(len(dataset)):
                        if float(dataset[d, attribute_index]) <= i:
                            if '<=%s' % i not in sub_label.keys():
                                sub_label['<=%s' % i] = []
                            sub_label['<=%s' % i].append(label[d])
                            # if watermelon_label[d] not in sub_label['<=%s' % i].keys():
                            #     sub_label['<=%s' % i][watermelon_label[d]] = 0
                            # sub_label['<=%s' % i][watermelon_label[d]] += 1
                        else:
                            if '>%s' % i not in sub_label.keys():
                                sub_label['>%s' % i] = []
                            sub_label['>%s' % i].append(label[d])
                            # if watermelon_label[d] not in sub_label['>%s' % i].keys():
                            #     sub_label['>%s' % i][watermelon_label[d]] = 0
                            # sub_label['>%s' % i][watermelon_label[d]] += 1
                    # print(sub_label)
                    if attributes[attribute_index] not in dict_attributes:
                        dict_attributes[attributes[attribute_index]] = {}
                    dict_attributes[attributes[attribute_index]][i] = base_entropy
                    # print(dict_attributes)
                    for s in sub_label.keys():
                        # print('s', sub_label[s])

                        instances_num = len(label)

                        # dict_attributes[attributes[attribute_index]][i] -= self.calculate_information_entropy(sub_label[s])

                        # sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value)[1]
                        # dict_attributes[watermelon_attributes[j]][i] = calculate_information_entropy(sub_label, instances_num)
                        dict_attributes[attributes[attribute_index]][i] -= len(
                            sub_label[s]) / instances_num * self.calculate_information_entropy(sub_label[s])
                # dict_attributes[attributes[attribute_index]][i]
                info_gain = max(dict_attributes[attributes[attribute_index]].values())
                continuous_value.append(max(dict_attributes[attributes[attribute_index]], key=dict_attributes[attributes[attribute_index]].get))
                dict_attributes[attributes[attribute_index]]['info_gain'] = info_gain

            else:

                dict_attributes[attributes[attribute_index]]['info_gain'] = base_entropy

                for attribute_value in attribute_values_list:
                    dict_attributes = self.calculate_information_gain(dataset, label, attributes, attribute_index,
                                                                      attribute_value, dict_attributes)[1]
                # print(continuous_value)
                continuous_value.append(False)
            if dict_attributes[attributes[attribute_index]]['info_gain'] > best_attribute_information_gain:
                best_attribute_information_gain = dict_attributes[attributes[attribute_index]]['info_gain']
                best_split_attribute_index = attribute_index
                max_value = continuous_value[attribute_index]
        # print(dict_attributes)
        best_split_attribute = attributes[best_split_attribute_index]
        print(dict_attributes)
        print('%s: info_gain = %.4f' % (best_split_attribute, best_attribute_information_gain), max_value)
        return best_split_attribute_index, best_attribute_information_gain, max_value

    def explanation_of_information_gain(self, dataset, label, attributes):
        instances_num = len(label)
        base_entropy = self.calculate_information_entropy(label)
        dict_attributes = {}
        for f in range(len(attributes)):
            """
            create empty attributes dictionary which contains:
                1) the information gain for each attribute;
                2) the attribute value space, whose keys are:
                    a) 'class_name': the number of instances of each class
                    b) 'instances_num': the number of instances belonging to the attribute value
                    c) 'info_ent': the information entropy of the attribute value
            """
            dict_attributes[attributes[f]] = {}
            dict_attributes[attributes[f]]['info_gain'] = base_entropy
            for d in set(dataset[:, f]):
                dict_attributes[attributes[f]][d] = {l: 0 for l in set(label)}
                dict_attributes[attributes[f]][d]['info_ent'] = 0
                dict_attributes[attributes[f]][d]['instances_num'] = 0
        for l in range(len(label)):
            for f in range(len(attributes)):
                dict_attributes[attributes[f]][dataset[l, f]][label[l]] += 1
                dict_attributes[attributes[f]][dataset[l, f]]['instances_num'] += 1
        for f in range(len(attributes)):
            for i in set(dataset[:, f]):
                separate_attribute = dict_attributes[attributes[f]][i]
                for l in set(label):
                    if separate_attribute[l] != 0:  # fix log2 of 0 byg
                        separate_attribute['info_ent'] -= separate_attribute[l] / separate_attribute[
                            'instances_num'] * np.log2(separate_attribute[l] / separate_attribute['instances_num'])
        best_attribute_information_gain = 0
        best_split_attribute_index = -1
        for f in range(len(attributes)):
            for i in set(dataset[:, f]):
                separate_attribute = dict_attributes[attributes[f]][i]
                dict_attributes[attributes[f]]['info_gain'] -= separate_attribute['instances_num'] / instances_num * \
                                                               separate_attribute['info_ent']
            if dict_attributes[attributes[f]]['info_gain'] > best_attribute_information_gain:
                best_attribute_information_gain = dict_attributes[attributes[f]]['info_gain']
                best_split_attribute_index = f
        best_split_attribute = attributes[best_split_attribute_index]
        # print(dict_attributes)
        # print('%s: %.4f' % (best_split_attribute, best_attribute_information_gain))
        return best_split_attribute, best_attribute_information_gain

    # =============================================================================
    # Choose the optimal attribute to split the dataset using the gain ratio based on C4.5
    # The gain ratio prefers attributes with fewer possible values!
    #
    # Intrinsic value (IV) = -\sum_{v=1}^{number of attribute values}{|D^v|/|D| * \log_2{|D^v|/|D|}}
    # gain_ratio(D, a) = info_gain(D, a) / IV(a)
    #
    # Note: The larger the number of possible values of attribute 'a', the larger the IV is usually.
    #
    # Rule: choose the attribute whose information gain is greater than the average information gain of the dataset,
    #       and then select the attribute with the largest gain ratio from them.
    # =============================================================================
    def C45_choose_optimal_attribute(self, dataset, label, attributes):
        instances_num = len(label)
        base_entropy = self.calculate_information_entropy(label)
        dict_attributes = {}
        total_info_gain = 0
        best_attribute_gain_ratio = 0
        best_split_attribute_index = -1
        for attribute_index in range(len(attributes)):
            """
            Create attributes dictionary which contains:
            1) 'info_gain': the information gain;
            2) 'attribute value': the information entropy of the attribute value;
            3) 'gain_ratio': the gain ratio.
            """
            dict_attributes[attributes[attribute_index]] = {}
            dict_attributes[attributes[attribute_index]]['info_gain'] = base_entropy
            IV = 0  # intrinsic value 固有值，属性a的可能取值数目越大，IV通常越大
            for attribute_value in set(dataset[:, attribute_index]):
                sub_label, dict_attributes = self.calculate_information_gain(dataset, label, attributes, attribute_index,
                                                                             attribute_value, dict_attributes)
                # sub_label = split_dataset(dataset, label, attribute_index, attribute_value)[1]
                # dict_attributes[attributes[attribute_index]][attribute_value] = calculate_information_entropy(sub_label)
                # dict_attributes[attributes[attribute_index]]['info_gain'] -= \
                #     len(sub_label) / instances_num * dict_attributes[attributes[attribute_index]][attribute_value]
                if len(sub_label) != 0:  # fix log2 of 0 byg
                    IV -= len(sub_label) / instances_num * np.log2(len(sub_label) / instances_num)
            # print('%s: %.4f' % (attributes[attribute_index], IV))
            total_info_gain += dict_attributes[attributes[attribute_index]]['info_gain']
            dict_attributes[attributes[attribute_index]]['gain_ratio'] = dict_attributes[attributes[attribute_index]][
                                                                             'info_gain'] / IV
        print(dict_attributes)
        mean_info_gain = total_info_gain / len(attributes)
        # print('mean_info_gain: %.4f' % mean_info_gain)
        for attribute_index in range(len(attributes)):
            if dict_attributes[attributes[attribute_index]]['info_gain'] > mean_info_gain:
                if dict_attributes[attributes[attribute_index]]['gain_ratio'] > best_attribute_gain_ratio:
                    best_attribute_gain_ratio = dict_attributes[attributes[attribute_index]]['gain_ratio']
                    best_split_attribute_index = attribute_index
        print('%s: gain_ratio = %.4f' % (attributes[best_split_attribute_index], best_attribute_gain_ratio))
        return best_split_attribute_index, best_attribute_gain_ratio

    # =============================================================================
    # Choose the optimal attribute to split the dataset using the gini index based on CART
    # =============================================================================
    def CART_choose_optimal_attribute(self, dataset, label, attributes):
        instances_num = len(label)
        # gini_index = 0
        best_attribute_gini_index = 0
        best_split_attribute_index = -1
        dict_attributes = {}
        for attribute_index in range(len(attributes)):
            dict_attributes[attributes[attribute_index]] = {}
            dict_attributes[attributes[attribute_index]]['gini_index'] = 0
            for attribute_value in set(dataset[:, attribute_index]):
                sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value)[1]
                dict_attributes[attributes[attribute_index]][attribute_value] = self.calculate_gini(sub_label)
                dict_attributes[attributes[attribute_index]]['gini_index'] += len(sub_label) / instances_num * \
                                                                              dict_attributes[
                                                                                  attributes[attribute_index]][
                                                                                  attribute_value]
            if attribute_index == 0:
                best_attribute_gini_index = dict_attributes[attributes[attribute_index]]['gini_index']
                best_split_attribute_index = attribute_index
            if dict_attributes[attributes[attribute_index]]['gini_index'] < best_attribute_gini_index:
                best_attribute_gini_index = dict_attributes[attributes[attribute_index]]['gini_index']
                best_split_attribute_index = attribute_index
        print(dict_attributes)
        print('%s: gini_index = %.4f' % (attributes[best_split_attribute_index], best_attribute_gini_index))
        return best_split_attribute_index, best_attribute_gini_index

    # def predict(self, dataset):
    #     for i in range(len(dataset)):
            # for a in range(len(dataset[i])):







if __name__ == '__main__':
    # watermelon_2 = creat_dataset('watermelon_2')
    # calculate_information_gain(watermelon_2[0], watermelon_2[1], watermelon_2[2])
    watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('watermelon_2', ', ')
    watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('watermelon_3', ', ')
    # watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('watermelon_3_alpha', ', ')
    # calculate_information_entropy(watermelon_label)
    # explanation_of_information_gain(watermelon_dataset, watermelon_label, watermelon_attributes)
    # ID3_choose_optimal_attribute(watermelon_dataset, watermelon_label, watermelon_attributes)
    # # print(calculate_information_entropy(watermelon_label))
    # C45_choose_optimal_attribute(watermelon_dataset, watermelon_label, watermelon_attributes)
    # CART_choose_optimal_attribute(watermelon_dataset, watermelon_label, watermelon_attributes)
    decision_tree = DecisionTreeGenerate()
    decision_tree.fit(watermelon_dataset, watermelon_label, watermelon_attributes)
    # a = {'1': 100, '2': 100, '3': 1}
    # print(max(a, key=a.get))
    # weather_dataset, weather_label, weather_attributes = creat_dataset('weather', '\t')
    # decision_tree = DecisionTreeGenerate()
    # decision_tree.fit(weather_dataset, weather_label, weather_attributes)

