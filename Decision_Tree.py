#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/6
# @Author  : javier_ji
# @Software: PyCharm
# @File    : Decision_Tree.py
import numpy as np


# =============================================================================
# Create dataset from the txt file
# =============================================================================
def creat_dataset(file_name, delimiter):
    dataset = np.genfromtxt(file_name, delimiter=delimiter, dtype=str)
    dataset_attributes = dataset[0, 1:-1]
    dataset_X = dataset[1:, 1:-1]
    dataset_Y = dataset[1:, -1]
    return dataset_X, dataset_Y, dataset_attributes


class TreeNode:
    def __init__(self, entropy=0, depth=0):
        self.__parent_attribute = None
        self.__parent_attribute_value = None
        self.__current_attribute = None            # 结点的属性名
        self.__current_attribute_index = None
        self.__current_attribute_value = None
        self.__instances_id = {}       # 结点中
        self.__entropy = entropy                # If use ID3 or C4.5 to build the decision tree
        self.__children = {}
        self.__is_leaf = 'N'               # 是否为叶结点
        self.__label = None
        self.__depth = depth                # 离root的距离
        self.__details = {}

    def set_parent_attribute(self, parent_attribute):
        self.__parent_attribute = parent_attribute

    @property
    def get_parent_attribute(self):
        return self.__parent_attribute

    def set_parent_attribute_value(self, parent_attribute_value):
        self.__parent_attribute_value = parent_attribute_value

    @property
    def get_parent_attribute_value(self):
        return self.__parent_attribute_value

    def set_current_attribute(self, attribute):
        self.__current_attribute = attribute

    @property
    def get_current_attribute(self):
        return self.__current_attribute

    def set_current_attribute_index(self, attribute_index):
        self.__current_attribute_index = attribute_index

    @property
    def get_current_attribute_index(self):
        return self.__current_attribute_index

    def set_current_attribute_value(self, attribute_value):
        self.__current_attribute_value = attribute_value

    @property
    def get_current_attribute_value(self):
        return self.__current_attribute_value

    def set_instances_id(self, attribute_value, instances_id):
        if attribute_value not in self.__instances_id.keys():
            self.__instances_id[attribute_value] = instances_id

    def get_instances_id(self, attribute_value):
        return self.__instances_id[attribute_value]

    def set_entropy(self, entropy):
        self.__entropy = entropy

    def get_entropy(self):
        return self.__entropy

    def set_children(self, attribute_value, node):
        # if (attribute_value and node) is not None:
        self.__children[attribute_value] = node

    def get_children(self, key=None):
        # if key is None:
        #     return
        return self.__children[key]

    def get_children_keys(self):
        return self.__children.keys()

    def set_leaf(self):
        self.__is_leaf = 'Y'

    @property
    def whether_leaf(self):
        if self.__is_leaf == 'Y':
            return True
        return False

    def set_label(self, label):
        self.__label = label

    def get_label(self):
        return self.__label

    def set_depth(self, depth):
        self.__depth = depth

    @property
    def get_depth(self):
        return self.__depth

    def set_node(self, parent_attribute=None, parent_attribute_value=None, current_attribute=None,
                 current_attribute_index=None, current_attribute_value=None, depth=0, details=None):
        if parent_attribute is not None:
            self.__parent_attribute = parent_attribute
        if parent_attribute_value is not None:
            self.__parent_attribute_value = parent_attribute_value
        if current_attribute is not None:
            self.__current_attribute = current_attribute
        if current_attribute_index is not None:
            self.__current_attribute_index = current_attribute_index
        if current_attribute_value is not None:
            self.__current_attribute_value = current_attribute_value
        if depth != 0:
            self.__depth = depth
        if details is not None:
            self.__details = details

    # def get_node(self):
    #     return self._split_attribute, self.__details, self.__depth

    def print_tree(self):
        print('parent attribute: ', self.__parent_attribute,
              '\nparent attribute value: ', self.__parent_attribute_value,
              '\ncurrent attribute: ', self.__current_attribute,
              '\ncurrent attribute index: ', self.__current_attribute_index,
              '\ncurrent attribute value: ', self.__current_attribute_value,
              '\ninstances id: ', self.__instances_id,
              '\nentropy: ', self.__entropy,
              '\nchildren:', self.__children,
              '\nis_leaf: : ', self.__is_leaf,
              '\nlabel: ', self.__label,
              '\ndepth: ', self.__depth,
              '\ndetails: ', self.__details)


class DecisionTreeGenerate:
    def __init__(self, max_depth=10):
        self.__root = TreeNode()
        self.max_depth = max_depth
        self.dataset = None
        self.label = None
        self.attributes = None

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
    # Split dataset based on the attribute value to get sub label
    # =============================================================================
    @staticmethod
    def split_dataset(dataset, label, attribute_index, split_attribute_value, dataset_index=None, is_continuous=False):
        if dataset_index is None:
            dataset_index = []
        if not is_continuous:
            rest_label = []
            rest_dataset = []
            rest_dataset_id = []
            for i in range(len(label)):
                if dataset[i, attribute_index] == split_attribute_value:
                    if len(dataset_index) == 0:

                        rest_label.append(label[i])
                        rest_dataset_id.append(i)
                        rest_dataset.append(list(dataset[i, :]))
                    else:
                        rest_label.append(label[i])
                        rest_dataset_id.append(dataset_index[i])
                        rest_dataset.append(list(dataset[i, :]))
            rest_dataset = np.array(rest_dataset)

        else:
            rest_label = {'<=%s' % split_attribute_value: [], '>%s' % split_attribute_value: []}
            rest_dataset = {'<=%s' % split_attribute_value: [], '>%s' % split_attribute_value: []}
            rest_dataset_id = {'<=%s' % split_attribute_value: [], '>%s' % split_attribute_value: []}
            for i in range(len(label)):
                if float(dataset[i, attribute_index]) <= split_attribute_value:
                    if len(dataset_index) == 0:
                        rest_label['<=%s' % split_attribute_value].append(label[i])
                        rest_dataset_id['<=%s' % split_attribute_value].append(i)
                        rest_dataset['<=%s' % split_attribute_value].append(list(dataset[i, :]))
                    else:
                        rest_label['<=%s' % split_attribute_value].append(label[i])
                        rest_dataset_id['<=%s' % split_attribute_value].append(dataset_index[i])
                        rest_dataset['<=%s' % split_attribute_value].append(list(dataset[i, :]))
                else:
                    if len(dataset_index) == 0:
                        rest_label['>%s' % split_attribute_value].append(label[i])
                        rest_dataset_id['>%s' % split_attribute_value].append(i)
                        rest_dataset['>%s' % split_attribute_value].append(list(dataset[i, :]))
                    else:
                        rest_label['>%s' % split_attribute_value].append(label[i])
                        rest_dataset_id['>%s' % split_attribute_value].append(dataset_index[i])
                        rest_dataset['>%s' % split_attribute_value].append(list(dataset[i, :]))
            for i in rest_dataset.keys():
                rest_dataset[i] = np.array(rest_dataset[i])
        return rest_dataset, rest_label, rest_dataset_id

    def calculate_information_gain(self, dataset, label, attribute_index,
                                   attribute_value, dict_attributes, is_continuous=False):
        sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value, is_continuous=is_continuous)[1]
        if not is_continuous:
            instances_num = len(label)
            # sub_label = self.split_dataset(dataset, label, attribute_index, attribute_value)[1]
            dict_attributes[self.attributes[attribute_index]][attribute_value] = self.calculate_information_entropy(sub_label)
            dict_attributes[self.attributes[attribute_index]]['info_gain'] -= len(sub_label) / instances_num * dict_attributes[self.attributes[attribute_index]][attribute_value]
        else:
            instances_num = 0
            for k in sub_label.keys():
                instances_num += len(sub_label[k])
            if attribute_value not in dict_attributes[self.attributes[attribute_index]].keys():
                dict_attributes[self.attributes[attribute_index]][attribute_value] = {}
                dict_attributes[self.attributes[attribute_index]][attribute_value]['info_gain'] = self.calculate_information_entropy(label)
            # for k in label.keys():
            for k in sub_label.keys():

                dict_attributes[self.attributes[attribute_index]][attribute_value][k] = self.calculate_information_entropy(sub_label[k])
                dict_attributes[self.attributes[attribute_index]][attribute_value]['info_gain'] -= \
                    len(sub_label[k]) / instances_num * dict_attributes[self.attributes[attribute_index]][attribute_value][k]
        return sub_label, dict_attributes

    @staticmethod
    def is_continuous(sample):
        try:
            float(sample)
            return True
        except ValueError:
            return False

    def ID3_choose_optimal_attribute(self, dataset, label):
        base_entropy = self.calculate_information_entropy(label)
        optimal_attribute_information_gain = 0
        optimal_split_attribute_index = -1
        dict_attributes = {}
        whether_continuous_value = []
        optimal_whether_continuous = False
        for attribute_index in range(len(self.attributes)):
            """
            Create attributes dictionary which contains:
            1) 'info_gain': the information gain;
            2) 'attribute value': the information entropy of the attribute value
            """
            dict_attributes[self.attributes[attribute_index]] = {}
            dict_attributes[self.attributes[attribute_index]]['info_gain'] = base_entropy
            attribute_values_list = list(set(dataset[:, attribute_index]))

            if self.is_continuous(attribute_values_list[0]):
                attribute_values_list.sort()
                attribute_values_list_continuous = []
                for i in range(len(attribute_values_list) - 1):
                    attribute_values_list_continuous.append(
                        round((float(attribute_values_list[i]) + float(attribute_values_list[i + 1])) / 2, 3))
                for attribute_value in attribute_values_list_continuous:
                    dict_attributes = self.calculate_information_gain(dataset, label, attribute_index,
                                                                      attribute_value, dict_attributes, True)[1]
                max_info_gain = -1
                max_value_inner = 0
                for i in dict_attributes[self.attributes[attribute_index]].keys():
                    if i != 'info_gain':
                        if dict_attributes[self.attributes[attribute_index]][i]['info_gain'] > max_info_gain:
                            max_info_gain = dict_attributes[self.attributes[attribute_index]][i]['info_gain']
                            max_value_inner = i
                dict_attributes[self.attributes[attribute_index]]['info_gain'] = max_info_gain
                whether_continuous_value.append(max_value_inner)
            else:
                for attribute_value in attribute_values_list:
                    dict_attributes = self.calculate_information_gain(dataset, label, attribute_index,
                                                                      attribute_value, dict_attributes)[1]
                whether_continuous_value.append(False)
            if dict_attributes[self.attributes[attribute_index]]['info_gain'] > optimal_attribute_information_gain:
                optimal_attribute_information_gain = dict_attributes[self.attributes[attribute_index]]['info_gain']
                optimal_split_attribute_index = attribute_index
                optimal_whether_continuous = whether_continuous_value[attribute_index]
        # best_split_attribute = self.attributes[optimal_split_attribute_index]
        # print(dict_attributes)
        # print('%s: info_gain = %.4f' % (best_split_attribute, optimal_attribute_information_gain), optimal_whether_continuous)
        return optimal_split_attribute_index, optimal_attribute_information_gain, optimal_whether_continuous

    def fit(self, dataset, label, attributes):
        self.dataset = dataset
        self.label = label
        self.attributes = attributes
        self.tree_generate_new(self.__root)
        self.print_root()

    def tree_generate_new(self, node, attribute_value=None, parent_node=None):
        sub_dataset = []
        sub_label = []
        sub_instance_id = []
        if node != self.__root:
            sub_instance_id = parent_node.get_instances_id(attribute_value)
            for i in sub_instance_id:
                sub_dataset.append(list(self.dataset[i]))
                sub_label.append(self.label[i])
            sub_dataset = np.array(sub_dataset)

        else:
            sub_dataset = self.dataset
            sub_label = self.label
        optimal_attribute_index, optimal_info_gain, optimal_whether_continuous = \
            self.ID3_choose_optimal_attribute(sub_dataset, sub_label)
        node.set_entropy(self.calculate_information_entropy(sub_label))
        split_sub_dataset, split_sub_label, split_sub_dataset_id = {}, {}, {}

        if not optimal_whether_continuous:
            keys = list(set(self.dataset[:, optimal_attribute_index]))
        else:
            split_sub_dataset, split_sub_label, split_sub_dataset_id = \
                self.split_dataset(sub_dataset, sub_label, optimal_attribute_index, optimal_whether_continuous,
                                   dataset_index=sub_instance_id, is_continuous=True)
            keys = list(set(split_sub_dataset.keys()))
        node.set_node(current_attribute=self.attributes[optimal_attribute_index],
                      current_attribute_index=optimal_attribute_index)
        details = {}
        for i in keys:
            if i not in details.keys():
                details[i] = {}
            if not optimal_whether_continuous:
                split_sub_dataset, split_sub_label, split_sub_dataset_id = self.split_dataset(sub_dataset,
                                                                                              sub_label,
                                                                                              optimal_attribute_index,
                                                                                              i,
                                                                                              sub_instance_id)
                details[i]['sub_dataset'] = split_sub_dataset
                details[i]['sub_label'] = split_sub_label
                node.set_instances_id(i, split_sub_dataset_id)
                deduplication_label = set(split_sub_label)
            else:
                details[i]['sub_dataset'] = split_sub_dataset[i]
                details[i]['sub_label'] = split_sub_label[i]
                node.set_instances_id(i, split_sub_dataset_id[i])
                deduplication_label = set(split_sub_label[i])
            depth = node.get_depth + 1

            if depth <= self.max_depth:
                new_node = TreeNode(entropy=self.calculate_information_entropy(details[i]['sub_label']), depth=depth)
                new_node.set_node(parent_attribute=node.get_current_attribute, parent_attribute_value=i,
                                  details={i: details[i]})
                # new_node.set_entropy(self.calculate_information_entropy(details[i]['sub_label']))
                if len(deduplication_label) == 1:                                       # 生成叶结点
                    current_node_label = deduplication_label.pop()
                    new_node.set_leaf()
                    new_node.set_label(current_node_label)
                    new_node.set_instances_id(i, node.get_instances_id(i))
                elif len(deduplication_label) == 0 or depth == self.max_depth:          # 生成叶结点
                    most_node_label = {}
                    for j in sub_label:
                        if j not in most_node_label.keys():
                            most_node_label[j] = 0
                        most_node_label[j] += 1
                    most_frequent_label = max(most_node_label, key=most_node_label.get)
                    new_node.set_leaf()
                    new_node.set_label(most_frequent_label)
                    new_node.set_instances_id(i, node.get_instances_id(i))
                else:
                    self.tree_generate_new(new_node, i, node)                           # 递归生成叶结点
                # new_node.print_tree()
                node.set_children(i, new_node)
                node.set_current_attribute_value(keys)

        details['info_gain'] = optimal_info_gain                        # 只有内部节点存在information gain
        node.set_node(details=details)

    def print_root(self):
        queue = [self.__root]
        while queue:
            node = queue.pop()
            if node.get_current_attribute_value is None:
                print('***************** Leaf Node *****************')
                node.print_tree()
            else:
                if node.get_parent_attribute is None:
                    print('***************** Root *****************')
                else:
                    print('***************** Internal Node: %s *****************' % node.get_current_attribute_value)
                node.print_tree()
            if not node.whether_leaf:
                for i in node.get_children_keys():
                    queue.append(node.get_children(i))


if __name__ == '__main__':
    # watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('./Data/watermelon_2', ', ')
    watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('./Data/watermelon_3', ', ')
    # watermelon_dataset, watermelon_label, watermelon_attributes = creat_dataset('./Data/watermelon_3_alpha', ', ')
    decision_tree_watermelon = DecisionTreeGenerate()
    # decision_tree_watermelon_limit_depth = DecisionTreeGenerate(2)
    decision_tree_watermelon.fit(watermelon_dataset, watermelon_label, watermelon_attributes)

    # loan_dataset, loan_label, loan_attributes = creat_dataset('./Data/loan_data', ', ')
    # decision_tree_loan = DecisionTreeGenerate()
    # decision_tree_loan.fit(loan_dataset, loan_label, loan_attributes)

    # weather_dataset, weather_label, weather_attributes = creat_dataset('./Data/weather', '\t')
    # decision_tree_weather = DecisionTreeGenerate()
    # decision_tree_weather.fit(weather_dataset, weather_label, weather_attributes)


