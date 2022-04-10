import pandas as pd
import pickle
import math


class Information:

    def __init__(self, address):
        self.data_set = pd.read_csv(address)
        self.data_set = self.data_set.head(int(0.8 * len(self.data_set.index)))
        self.info0 = {}
        self.info1 = {}
        self.p0 = 0
        self.p1 = 0
        self.gain_info()

    def variance(self, array):
        mean = 0
        for element in array:
            mean += element
        mean /= len(array)

        variance = 0
        for element in array:
            variance += pow(element - mean, 2)
        variance /= len(array) - 1
        return [mean, variance]

    def continuous_feature(self, feature):
        classes = self.data_set.disease
        features = self.data_set[feature]
        feature0 = []
        feature1 = []
        for i in range(len(classes)):
            if classes[i] == 1:
                feature1.append(features[i])
            else:
                feature0.append(features[i])
        self.info0[feature] = self.variance(feature0)
        self.info1[feature] = self.variance(feature1)
        return

    def binary_feature(self, feature, class0):
        classes = self.data_set.disease
        rows = len(classes)
        class1 = rows - class0
        features = self.data_set[feature]
        feature0 = 0
        feature1 = 0
        for i in range(rows):
            if classes[i] == 0 and features[i] == 0:
                feature0 += 1
            elif classes[i] == 1 and features[i] == 1:
                feature1 += 1
        p = feature0 / class0
        self.info0[feature] = [p, 1 - p]
        p = feature1 / class1
        self.info1[feature] = [p, 1 - p]

    def gain_info(self):
        classes = self.data_set.disease
        rows = len(classes)
        class0 = 0
        class1 = 0
        for element in classes:
            if element == 1:
                class1 += 1
            else:
                class0 += 1
        self.p0 = class0 / rows
        self.p1 = class1 / rows

        self.continuous_feature("age")

        self.binary_feature("sex", class0)

        cp = self.data_set.cp
        self.info1['cp'] = []
        self.info0['cp'] = []
        cp_count = [[0] * 2] * 4
        for i in range(rows):
            if classes[i] == 0:
                if cp[i] == 0:
                    cp_count[0][0] += 1
                elif cp[i] == 1:
                    cp_count[1][0] += 1
                elif cp[i] == 2:
                    cp_count[2][0] += 1
                else:
                    cp_count[3][0] += 1
            else:
                if cp[i] == 0:
                    cp_count[0][1] += 1
                elif cp[i] == 1:
                    cp_count[1][1] += 1
                elif cp[i] == 2:
                    cp_count[2][1] += 1
                else:
                    cp_count[3][1] += 1
        for i in range(4):
            self.info0['cp'].append(cp_count[i][0] / class0)
            self.info1['cp'].append(cp_count[i][1] / class1)

        self.continuous_feature("trestbps")

        self.continuous_feature("chol")

        self.binary_feature("fbs", class0)

        restecg = self.data_set.restecg
        self.info1['restecg'] = []
        self.info0['restecg'] = []
        restecg_count = [[0] * 2] * 3
        for i in range(rows):
            if classes[i] == 0:
                if restecg[i] == 0:
                    restecg_count[0][0] += 1
                elif restecg[i] == 1:
                    restecg_count[1][0] += 1
                else:
                    restecg_count[2][0] += 1
            else:
                if restecg[i] == 0:
                    restecg_count[0][1] += 1
                elif restecg[i] == 1:
                    restecg_count[1][1] += 1
                else:
                    restecg_count[2][1] += 1
        for i in range(3):
            self.info0['restecg'].append(restecg_count[i][0] / class0)
            self.info1['restecg'].append(restecg_count[i][1] / class1)

        self.continuous_feature("thalach")

        self.binary_feature("exang", class0)

        self.continuous_feature("oldpeak")

        slope = self.data_set.slope
        self.info1['slope'] = []
        self.info0['slope'] = []
        slope_count = [[0] * 2] * 3
        for i in range(rows):
            if classes[i] == 0:
                if slope[i] == 0:
                    slope_count[0][0] += 1
                elif slope[i] == 1:
                    slope_count[1][0] += 1
                else:
                    slope_count[2][0] += 1
            else:
                if slope[i] == 0:
                    slope_count[0][1] += 1
                elif slope[i] == 1:
                    slope_count[1][1] += 1
                else:
                    slope_count[2][1] += 1
        for i in range(3):
            self.info0['slope'].append(slope_count[i][0] / class0)
            self.info1['slope'].append(slope_count[i][1] / class1)

        ca = self.data_set.ca
        self.info0['ca'] = []
        self.info1['ca'] = []
        ca_count = [[0] * 2] * 5
        for i in range(rows):
            if classes[i] == 0:
                if ca[i] == 0:
                    ca_count[0][0] += 1
                elif ca[i] == 1:
                    ca_count[1][0] += 1
                elif ca[i] == 2:
                    ca_count[2][0] += 1
                elif ca[i] == 3:
                    ca_count[3][0] += 1
                else:
                    ca_count[4][0] += 1
            else:
                if ca[i] == 0:
                    ca_count[0][1] += 1
                elif ca[i] == 1:
                    ca_count[1][1] += 1
                elif ca[i] == 2:
                    ca_count[2][1] += 1
                elif ca[i] == 3:
                    ca_count[3][1] += 1
                else:
                    ca_count[4][1] += 1
        for i in range(5):
            self.info0['ca'].append(ca_count[i][0] / class0)
            self.info1['ca'].append(ca_count[i][1] / class1)

        thal = self.data_set.thal
        self.info1['thal'] = []
        self.info0['thal'] = []
        thal_count = [[0] * 2] * 4
        for i in range(rows):
            if classes[i] == 0:
                if thal[i] == 0:
                    thal_count[0][0] += 1
                elif thal[i] == 1:
                    thal_count[1][0] += 1
                elif thal[i] == 2:
                    thal_count[2][0] += 1
                else:
                    thal_count[3][0] += 1
            else:
                if thal[i] == 0:
                    thal_count[0][1] += 1
                elif thal[i] == 1:
                    thal_count[1][1] += 1
                elif thal[i] == 2:
                    thal_count[2][1] += 1
                else:
                    thal_count[3][1] += 1
        for i in range(4):
            self.info0['thal'].append(thal_count[i][0] / class0)
            self.info1['thal'].append(thal_count[i][1] / class1)


class Classification:

    def __init__(self, information, features):
        self.information = information
        self.features = features

    def normal_distribution(self, x, feature, y):
        if y == 0:
            mean = self.information.info0[feature][0]
            variance = self.information.info0[feature][1]
        else:
            mean = self.information.info1[feature][0]
            variance = self.information.info1[feature][1]

        p = (1 / pow(2 * math.pi * variance, 0.5)) * math.exp(-pow(x - mean, 2) / (2 * variance))
        return p

    def classify(self, data):
        p0 = 1
        p1 = 1
        for i in range(len(self.features)):
            if self.features[i] == "age" or self.features[i] == "trestbps" or self.features[i] == "chol" or \
                    self.features[i] == "thalach" or self.features[i] == "oldpeak":
                p0 *= self.normal_distribution(data[i], self.features[i], 0)
                p1 *= self.normal_distribution(data[i], self.features[i], 1)
            else:
                p0 *= self.information.info0[self.features[i]][int(data[i])]
                p1 *= self.information.info1[self.features[i]][int(data[i])]
        p0 *= self.information.p0
        p1 *= self.information.p1
        if p1 > p0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    # information = Information("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset3.csv")
    # with open("D:\\computer\\DataMining\\HW2\\Naive_Bayes_information", "wb") as writer:
    #     pickle.dump(information, writer)

    with open("D:\\computer\\DataMining\\HW2\\Naive_Bayes_information", "rb") as reader:
        information = pickle.load(reader)

    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
    classification = Classification(information, features)

    # data_set = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset3.csv")
    # total_predictions = 0
    # true_predictions = 0
    # for i in range(int(0.8 * len(data_set.index)), len(data_set.index)):
    #     data = data_set.iloc[i]
    #     real_class = data['disease']
    #     predicted_class = classification.classify(data)
    #     total_predictions += 1
    #     if real_class == predicted_class:
    #         true_predictions += 1
    # accuracy = (true_predictions / total_predictions) * 100
    # print("accuracy: ", accuracy)

    test_data = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset3_Unknown.csv")
    result = []
    for i in range(len(test_data.index)):
        data = test_data.iloc[i]
        result.append(classification.classify(data))

    with open("D:\\computer\\DataMining\\HW2\\unknown3_clf_naive_bayes", "wb") as writer:
        pickle.dump(result, writer)

    with open("D:\\computer\\DataMining\\HW2\\unknown3_clf_naive_bayes", "rb") as reader:
        result = pickle.load(reader)
    print(result)
