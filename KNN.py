import pickle

import pandas as pd
from sklearn import preprocessing
import numpy as np


class Point:

    def __init__(self, distance, poisonous):
        self.distance = distance
        self.poisonous = poisonous


class Heap:

    def __init__(self, points):
        self.points = points
        self.points.append(Point(float('inf'), True))
        self.build_min_heap()

    def swap(self, i, j):
        point = self.points[i]
        self.points[i] = self.points[j]
        self.points[j] = point
        return

    def min_heapify(self, index):
        elements = len(self.points)

        if (index * 2) + 2 < elements:
            smallest = index
            if self.points[smallest].distance > self.points[(index * 2) + 2].distance:
                smallest = (index * 2) + 2
            if self.points[smallest].distance > self.points[(index * 2) + 1].distance:
                smallest = (index * 2) + 1
            if smallest != index:
                self.swap(index, smallest)
                self.min_heapify(smallest)
        elif (index * 2) + 1 < elements:
            if self.points[index].distance > self.points[(index * 2) + 1].distance:
                self.swap(index, (index * 2) + 1)
        return

    def build_min_heap(self):
        element = len(self.points)
        for i in range(len(self.points)):
            index = element - i - 1
            self.min_heapify(index)
        return

    def min_heap(self):
        result = self.points[0]
        self.points[0] = Point(float('inf'), True)
        self.min_heapify(0)
        return result


class KNN:

    def __init__(self, data_set, k):
        self.data_set = data_set
        self.k = k

    def d(self, x, y):
        x = np.array(x)
        y = np.array(y)
        distance = np.linalg.norm(x - y, 2)
        return distance

    def calculate_distances(self, data):
        # del data['poisonous']
        points = []
        for i in range(len(self.data_set.index)):
            row = self.data_set.iloc[i]
            if row[0] == 1:
                poisonous = True
            else:
                poisonous = False
            del row['poisonous']
            distance = self.d(row, data)
            point = Point(distance, poisonous)
            points.append(point)
        return points

    def classify(self, data):
        points = self.calculate_distances(data)
        heap = Heap(points)
        nearest_neighbors = []
        for i in range(self.k):
            point = heap.min_heap()
            if point.distance == float('inf'):
                return -1
            nearest_neighbors.append(point)

        count1 = 0
        count0 = 0
        for neighbor in nearest_neighbors:
            if neighbor.poisonous:
                count1 += 1
            else:
                count0 += 1
        if count1 > count0:
            return 1
        else:
            return 0


if __name__ == "__main__":
    x = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset2.csv")
    le = preprocessing.LabelEncoder()
    data_set = x.apply(le.fit_transform)
    data_set1 = data_set.head(int(0.8 * len(data_set.index)))

    # for i in range(1, 4):
    #     knn = KNN(data_set1, i)
    #     total_predictions = 0
    #     true_predictions = 0
    #     for j in range(int(0.8 * len(data_set.index)), len(data_set.index)):
    #         data = data_set.iloc[j]
    #         real_class = data['poisonous']
    #         predicted_class = knn.classify(data)
    #         total_predictions += 1
    #
    #         if predicted_class == real_class:
    #             true_predictions += 1
    #     accuracy = (true_predictions / total_predictions) * 100
    #     print("K:", i, "   accuracy:", accuracy)

    y = pd.read_csv("D:\\computer\\DataMining\\HW2\\Dataset\\Dataset2_Unknown.csv")
    le = preprocessing.LabelEncoder()
    test_data = y.apply(le.fit_transform)

    knn = KNN(data_set1, 1)
    result = []
    for i in range(len(test_data.index)):
        data = test_data.iloc[i]
        result.append(knn.classify(data))

    with open("D:\\computer\\DataMining\\HW2\\Dataset\\unknown2_clf_knn", "wb") as writer:
        pickle.dump(result, writer)

    with open("D:\\computer\\DataMining\\HW2\\Dataset\\unknown2_clf_knn", "rb") as reader:
        result = pickle.load(reader)
    print(result)
