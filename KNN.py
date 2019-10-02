import math
import operator
import numpy as np

class KNN(object):
    def __init__(self, k, inputs, data_set):
        self.k = k
        self.inputs = inputs
        self.data_set = data_set
        self.prepare_data(self.data_set)

    def prepare_data(self, data_set):
        np.random.shuffle(data_set)
        train_size = int(0.8 * len(data_set))
        self.train_set, self.test_set = data_set[:train_size], data_set[train_size:]
 
    def get_neighbors(self, data):
        distances, neighbors = [], []
        for x in range(len(self.train_set)):
            dist = self.euclidean_distance(data, self.train_set[x])
            distances.append((self.train_set[x], dist))
        distances.sort(key=operator.itemgetter(1))
        for x in range(self.k):
            neighbors.append(distances[x][0])
        return neighbors
    
    def predict(self, data):
        class_votes = {}
        neighbors = self.get_neighbors(data)
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]
    
    def get_accuracy(self, predictions):
        correct = 0
        for x in range(len(self.test_set)):
            if self.test_set[x][-1] == predictions[x]:
                correct += 1
        return (correct/float(len(self.test_set))) * 100.0

    def euclidean_distance(self, instance1, instance2):
        distance = 0
        for x in range(len(self.inputs)):
            distance += pow((instance1[self.inputs[x]] - instance2[self.inputs[x]]), 2)
        return math.sqrt(distance)
