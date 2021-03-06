import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import numpy as np

colors = 10 * ["g", "r", "c", "b", "k"]

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.mem_number = {}

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
                self.mem_number[i] = 0

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    # print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                # count member of each cluster
                for classification in self.classifications:
                    self.mem_number[classification] = np.count_nonzero(np.sum(self.classifications[classification], axis=1))
                break

    def predict(self, data):
        classes = []
        for d in data:
            distances = [np.linalg.norm(d - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            classes.append(classification)
        return np.array(classes)

    def update(self, new_data, delta):
        for featureset in new_data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]

            if min(distances) < delta:
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
                self.mem_number[classification] += 1
            else:
                self.centroids[self.k] = featureset
                self.classifications[self.k] = []
                self.mem_number[self.k] = 1
                self.classifications[self.k].append(featureset)
                self.k = self.k + 1

if __name__ == '__main__':
    X = np.array([[1],
                [1.5],
                [5]])

    clf = K_Means()
    clf.fit(X)
    print(clf.centroids)
    # X1 = np.array([[6, 8],
    #             [7, 10],
    #             [6, 4],
    #             [2, 2],
    #             [2, 3]])

    # #Updating the model with X1 and threshold of 4 
    # clf.update(X1, 4)

    # for centroid in clf.centroids:
    #     plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
    #                 marker="o", color="k", s=150, linewidths=5)

    # for classification in clf.classifications:
    #     color = colors[classification]
    #     for featureset in clf.classifications[classification]:
    #         plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

    # plt.show()