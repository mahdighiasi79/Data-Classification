import math

import pandas as pd

import pickle


class Node:

    def __init__(self, id, feature):
        self.id = id
        self.feature = feature
        self.children = {}

    def ask(self, data):
        if self.feature == "0":
            return -1
        if self.feature == "1":
            return -2
        if self.feature == "-1":
            return -3

        attribute = data[self.feature]
        if self.feature == "age":
            if attribute < 20:
                category = "teen"
            elif 20 <= attribute < 30:
                category = "adult"
            elif 30 <= attribute < 40:
                category = "young"
            elif 40 <= attribute < 50:
                category = "middle_aged"
            else:
                category = "old"

        elif self.feature == "fnlwgt":
            x25 = 50341
            x50 = 194897
            x75 = 233955
            if attribute <= x25:
                category = "first_q"
            elif x25 < attribute <= x50:
                category = "second_q"
            elif x50 < attribute <= x75:
                category = "third_q"
            else:
                category = "fourth_q"

        elif self.feature == "capitalGain":
            if attribute == 0:
                category = "no"
            else:
                category = "yes"

        elif self.feature == "capitalLoss":
            if attribute == 0:
                category = "no"
            else:
                category = "yes"

        elif self.feature == "hoursPerWeek":
            if attribute < 40:
                category = "low"
            elif attribute == 40:
                category = "normal"
            else:
                category = "high"

        elif self.feature == "race":
            if attribute == "White":
                category = "White"
            else:
                category = "other"

        elif self.feature == "nativeCountry":
            if attribute == "United-States":
                category = "US"
            else:
                category = "other"

        else:
            category = attribute

        for key in self.children.keys():
            if category == key:
                return self.children[key]
        return -3


class DT:

    def __init__(self):
        self.ids = 0
        self.nodes = {}

    def classify(self, data):
        id = 0
        while id != -1 and id != -2 and id != -3:
            id = self.nodes[id].ask(data)

        if id == -1:
            return 0
        elif id == -2:
            return 1
        else:
            return -1

    def id_generator(self):
        self.ids += 1
        return self.ids


class BuildDT:

    def __init__(self, address, criterion):
        self.data_set = pd.read_csv(address)
        self.criterion = criterion
        self.decision_tree = DT()

    def impurity_node(self, node):
        if len(node) == 0:
            return 0

        class0 = 0
        for i in node:
            data = self.data_set.iloc[i]
            if data[len(data) - 1] == "<=50K":
                class0 += 1
        class1 = len(node) - class0
        p0 = class0 / len(node)
        p1 = class1 / len(node)

        if self.criterion == "GINI":
            impurity = pow(p0, 2)
            impurity += pow(p1, 2)
            impurity = 1 - impurity
            return impurity
        else:
            if p0 == 0:
                impurity = 0
            else:
                impurity = p0 * math.log(p0)
            if p1 != 0:
                impurity += p1 * math.log(p1)
            impurity *= -1
            return impurity

    def gain_split(self, parent, nodes):
        gain = 0
        num = 0
        for node in nodes:
            gain += len(node) * self.impurity_node(node)
            num += len(node)
        gain /= num
        gain = self.impurity_node(parent) - gain
        return gain

    def split(self, node, feature):
        split = {}

        if feature == "age":
            column = self.data_set.age

            split['teen'] = []
            split['adult'] = []
            split['young'] = []
            split['middle_aged'] = []
            split['old'] = []

            for i in node:
                if column[i] < 20:
                    split['teen'].append(i)
                elif 20 <= column[i] < 30:
                    split['adult'].append(i)
                elif 30 <= column[i] < 40:
                    split['young'].append(i)
                elif 40 <= column[i] < 50:
                    split['middle_aged'].append(i)
                else:
                    split['old'].append(i)

        elif feature == "workclass":
            column = self.data_set.workclass

            split['State-gov'] = []
            split['Self-emp-not-inc'] = []
            split['Private'] = []
            split['Federal-gov'] = []
            split['Local-gov'] = []
            split['Self-emp-inc'] = []
            split['Without-pay'] = []
            split['Never-worked'] = []
            split['?'] = []

            for i in node:
                if column[i] == "State-gov":
                    split['State-gov'].append(i)
                elif column[i] == "Self-emp-not-inc":
                    split['Self-emp-not-inc'].append(i)
                elif column[i] == "Private":
                    split['Private'].append(i)
                elif column[i] == "Federal-gov":
                    split['Federal-gov'].append(i)
                elif column[i] == "Local-gov":
                    split['Local-gov'].append(i)
                elif column[i] == "Self-emp-inc":
                    split['Self-emp-inc'].append(i)
                elif column[i] == "Without-pay":
                    split['Without-pay'].append(i)
                elif column[i] == "Never-worked":
                    split['Never-worked'].append(i)
                else:
                    split['?'].append(i)

        elif feature == "fnlwgt":
            column = self.data_set.fnlwgt

            split['first_q'] = []
            split['second_q'] = []
            split['third_q'] = []
            split['fourth_q'] = []

            x25 = 50341
            x50 = 194897
            x75 = 233955

            for i in node:
                if column[i] <= x25:
                    split['first_q'].append(i)
                elif x25 < column[i] <= x50:
                    split['second_q'].append(i)
                elif x50 < column[i] <= x75:
                    split['third_q'].append(i)
                else:
                    split['fourth_q'].append(i)

        elif feature == "education":
            column = self.data_set.education

            split['Bachelors'] = []
            split['11th'] = []
            split['Masters'] = []
            split['9th'] = []
            split['HS-grad'] = []
            split['Some-college'] = []
            split['Assoc-acdm'] = []
            split['Assoc-voc'] = []
            split['7th-8th'] = []
            split['Doctorate'] = []
            split['Prof-school'] = []
            split['5th-6th'] = []
            split['10th'] = []
            split['1st-4th'] = []
            split['Preschool'] = []
            split['12th'] = []

            for i in node:
                if column[i] == "Bachelors":
                    split['Bachelors'].append(i)
                elif column[i] == "11th":
                    split['11th'].append(i)
                elif column[i] == "Assoc-acdm":
                    split['Assoc-acdm'].append(i)
                elif column[i] == "Masters":
                    split['Masters'].append(i)
                elif column[i] == "9th":
                    split['9th'].append(i)
                elif column[i] == "HS-grad":
                    split['HS-grad'].append(i)
                elif column[i] == "Some-college":
                    split['Some-college'].append(i)
                elif column[i] == "Assoc-voc":
                    split['Assoc-voc'].append(i)
                elif column[i] == "7th-8th":
                    split['7th-8th'].append(i)
                elif column[i] == "Doctorate":
                    split['Doctorate'].append(i)
                elif column[i] == "Prof-school":
                    split['Prof-school'].append(i)
                elif column[i] == "5th-6th":
                    split['5th-6th'].append(i)
                elif column[i] == "10th":
                    split['10th'].append(i)
                elif column[i] == "1st-4th":
                    split['1st-4th'].append(i)
                elif column[i] == "Preschool":
                    split['Preschool'].append(i)
                else:
                    split['12th'].append(i)

        elif feature == "maritalStatus":
            column = self.data_set.maritalStatus

            split['Never-married'] = []
            split['Married-civ-spouse'] = []
            split['Married-spouse-absent'] = []
            split['Divorced'] = []
            split['Separated'] = []
            split['Married-AF-spouse'] = []
            split['Widowed'] = []

            for i in node:
                if column[i] == "Never-married":
                    split['Never-married'].append(i)
                elif column[i] == "Married-civ-spouse":
                    split['Married-civ-spouse'].append(i)
                elif column[i] == "Married-spouse-absent":
                    split['Married-spouse-absent'].append(i)
                elif column[i] == "Divorced":
                    split['Divorced'].append(i)
                elif column[i] == "Separated":
                    split['Separated'].append(i)
                elif column[i] == "Married-AF-spouse":
                    split['Married-AF-spouse'].append(i)
                else:
                    split['Widowed'].append(i)

        elif feature == "occupation":
            column = self.data_set.occupation

            split['Adm-clerical'] = []
            split['Exec-managerial'] = []
            split['Handlers-cleaners'] = []
            split['Prof-specialty'] = []
            split['Other-service'] = []
            split['Sales'] = []
            split['Craft-repair'] = []
            split['Transport-moving'] = []
            split['Farming-fishing'] = []
            split['Machine-op-inspct'] = []
            split['Tech-support'] = []
            split['Protective-serv'] = []
            split['Armed-Forces'] = []
            split['Priv-house-serv'] = []
            split['?'] = []

            for i in node:
                if column[i] == "Adm-clerical":
                    split['Adm-clerical'].append(i)
                elif column[i] == "Exec-managerial":
                    split['Exec-managerial'].append(i)
                elif column[i] == "Handlers-cleaners":
                    split['Handlers-cleaners'].append(i)
                elif column[i] == "Prof-specialty":
                    split['Prof-specialty'].append(i)
                elif column[i] == "Other-service":
                    split['Other-service'].append(i)
                elif column[i] == "Sales":
                    split['Sales'].append(i)
                elif column[i] == "Craft-repair":
                    split['Craft-repair'].append(i)
                elif column[i] == "Transport-moving":
                    split['Transport-moving'].append(i)
                elif column[i] == "Farming-fishing":
                    split['Farming-fishing'].append(i)
                elif column[i] == "Machine-op-inspct":
                    split['Machine-op-inspct'].append(i)
                elif column[i] == "Tech-support":
                    split['Tech-support'].append(i)
                elif column[i] == "Protective-serv":
                    split['Protective-serv'].append(i)
                elif column[i] == "Armed-Forces":
                    split['Armed-Forces'].append(i)
                elif column[i] == "Priv-house-serv":
                    split['Priv-house-serv'].append(i)
                else:
                    split['?'].append(i)

        elif feature == "relationship":
            column = self.data_set.relationship

            split['Not-in-family'] = []
            split['Husband'] = []
            split['Wife'] = []
            split['Own-child'] = []
            split['Unmarried'] = []
            split['Other-relative'] = []

            for i in node:
                if column[i] == "Not-in-family":
                    split['Not-in-family'].append(i)
                elif column[i] == "Husband":
                    split['Husband'].append(i)
                elif column[i] == "Wife":
                    split['Wife'].append(i)
                elif column[i] == "Own-child":
                    split['Own-child'].append(i)
                elif column[i] == "Unmarried":
                    split['Unmarried'].append(i)
                else:
                    split['Other-relative'].append(i)

        elif feature == "race":
            column = self.data_set.race

            split['White'] = []
            split['other'] = []

            for i in node:
                if column[i] == "White":
                    split['White'].append(i)
                else:
                    split['other'].append(i)

        elif feature == "sex":
            column = self.data_set.sex

            split['Male'] = []
            split['Female'] = []

            for i in node:
                if column[i] == "Male":
                    split['Male'].append(i)
                else:
                    split['Female'].append(i)

        elif feature == "capitalGain":
            column = self.data_set.capitalGain

            split['yes'] = []
            split['no'] = []

            for i in node:
                if column[i] == 0:
                    split['no'].append(i)
                else:
                    split['yes'].append(i)

        elif feature == "hoursPerWeek":
            column = self.data_set.hoursPerWeek

            split['low'] = []
            split['normal'] = []
            split['high'] = []

            for i in node:
                if column[i] < 40:
                    split['low'].append(i)
                elif column[i] == 40:
                    split['normal'].append(i)
                else:
                    split['high'].append(i)

        elif feature == "capitalLoss":
            column = self.data_set.capitalLoss

            split['yes'] = []
            split['no'] = []

            for i in node:
                if column[i] == 0:
                    split['no'].append(i)
                else:
                    split['yes'].append(i)

        elif feature == "nativeCountry":
            column = self.data_set.nativeCountry

            split['US'] = []
            split['other'] = []

            for i in node:
                if column[i] == "United-States":
                    split['US'].append(i)
                else:
                    split['other'].append(i)

        return split

    def build_tree(self, node, features, id):
        income = self.data_set.income

        if len(node) == 0:
            leaf = Node(id, "-1")
            self.decision_tree.nodes[id] = leaf
            return

        if len(features) == 0:
            class0 = 0
            for row in node:
                if income[row] == "<=50K":
                    class0 += 1
            class1 = len(node) - class0
            if class1 > class0:
                feature = "1"
            else:
                feature = "0"
            leaf = Node(id, feature)
            self.decision_tree.nodes[id] = leaf
            return

        leaf_node = True
        for row in node:
            if income[row] != income[node[0]]:
                leaf_node = False
        if leaf_node:
            if income[node[0]] == "<=50K":
                feature = "0"
            else:
                feature = "1"
            leaf = Node(id, feature)
            self.decision_tree.nodes[id] = leaf
            return

        maximum_gain = -1
        expand = ""
        for feature in features:
            split = self.split(node, feature)
            nodes = []
            for key in split.keys():
                nodes.append(split[key])
            gain_split = self.gain_split(node, nodes)
            if gain_split > maximum_gain:
                maximum_gain = gain_split
                expand = feature

        split = self.split(node, expand)
        parent = Node(id, expand)
        features.remove(expand)
        children = {}
        for key in split.keys():
            child_id = self.decision_tree.id_generator()
            children[key] = child_id
            self.build_tree(split[key], features.copy(), child_id)
        parent.children = children
        self.decision_tree.nodes[id] = parent
        return


if __name__ == "__main__":
    # dt = BuildDT("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset1.csv", "Entropy")
    # features = ['age', 'workclass', 'fnlwgt', 'education',
    #             'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
    #             'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry'
    #             ]
    # root = []
    # for i in range(int(0.8 * len(dt.data_set.index))):
    #     root.append(i)
    # dt.build_tree(root, features, 0)
    #
    # with open("D:\\computer\\DataMining\\HW2\\DT_Entropy", "wb") as writer:
    #     pickle.dump(dt.decision_tree, writer)
    #
    # print("entropy finished")
    #
    # dt = BuildDT("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset1.csv", "GINI")
    # features = ['age', 'workclass', 'fnlwgt', 'education',
    #             'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
    #             'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry'
    #             ]
    # root = []
    # for i in range(int(0.8 * len(dt.data_set.index))):
    #     root.append(i)
    # dt.build_tree(root, features, 0)
    #
    # with open("D:\\computer\\DataMining\\HW2\\DT_GINI", "wb") as writer:
    #     pickle.dump(dt.decision_tree, writer)

    with open("D:\\computer\\DataMining\\HW2\\DT_Entropy", "rb") as reader:
        dt = pickle.load(reader)

    # test_data = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset1_Unknown.csv")
    #
    # results = []
    # for i in range(len(test_data.index)):
    #     results.append(dt.classify(test_data.iloc[i]))
    # with open("D:\\computer\\DataMining\\HW2\\unknown1_clf_dt_Entropy", "wb") as writer:
    #     pickle.dump(results, writer)
    #
    # with open("D:\\computer\\DataMining\\HW2\\unknown1_clf_dt_Entropy", "rb") as reader:
    #     vector = pickle.load(reader)
    # print(vector)

    data_set = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset1.csv")
    income = data_set.income

    total_predictions = 0
    true_predictions = 0
    for i in range(int(0.8 * len(data_set.index)), len(data_set.index)):
        predicted_class = dt.classify(data_set.iloc[i])
        total_predictions += 1
        if income[i] == ">50K":
            real_class = 1
        elif income[i] == "<=50K":
            real_class = 0
        else:
            real_class = -1
        if predicted_class == real_class:
            true_predictions += 1

    accuracy = (true_predictions / total_predictions) * 100
    print(accuracy)
