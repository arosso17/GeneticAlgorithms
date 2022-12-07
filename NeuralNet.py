import numpy as np
from numpy import array


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NeuralNetwork:
    def __init__(self, sizes, weights, biases):
        self.sizes = sizes
        if weights is not None and biases is not None:
            self.weights = weights[:]
            self.biases = biases[:]
        else:
            self.biases = [1 * np.random.rand(y, 1) - 0.5 for y in self.sizes[1:]]
            self.weights = [1 * np.random.rand(y, x) - 0.5 for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        # self.biases = [array([[-0.23067609],                                         # pretrained brain for testing
        #                       [-0.0230654], [0.00978982], [-0.14309116], [-0.33265655]]),
        #                array([[-0.26302269], [-0.42288439], [-0.30961607], [-0.02421273]])]
        # self.weights = [[array([0.41507994, -0.32873068, 0.07733694, 0.00333398, 0.19125972,
        #                         -0.24051878]), array([0.23031882, 0.34517107, 0.14855657, 0.44964635, 0.24338571,
        #                                               -0.09383268]),
        #                  array([-0.03772682, 0.35443, -0.01275211, -0.42030167, 0.22951438,
        #                         0.4527125]), array([0.07270966, -0.44618837, -0.49055737, 0.20304056, 0.24776172,
        #                                             0.40268698]),
        #                  array([-0.31794025, -0.35517426, -0.22939854, 0.39535277, 0.34899746,
        #                         0.23695801])], [array([0.27823309, 0.05322112, 0.47998052, 0.19268973, -0.26001732]),
        #                                         array([-0.04114979, -0.31362213, 0.00616192, -0.07710409, 0.27169233]),
        #                                         array([-0.31157883, 0.13238014, 0.40708475, 0.03098394, -0.46753936]),
        #                                         array([-0.08518676, -0.4545325, 0.234086, 0.13396382, 0.08156383])]]

    def think(self, dists, vel):  # takes the 5 distances and speed and outputs if it should turn or accelerate
        values = [[dists[0]], [dists[1]], [dists[2]], [dists[3]], [dists[4]], [vel]]
        for b, w in zip(self.biases, self.weights):
            values = sigmoid(np.dot(w, values) + b)
        return values

    def mutate(self):
        w = [[i + 0 for i in x] for x in self.weights]
        b = [b + 0 for b in self.biases]
        for i in range(len(b)):
            for j in range(len(b[i])):
                if np.random.random() > 0.2:
                    b[i][j][0] = 1 * np.random.rand() - 0.5
        for i in range(len(w)):
            for j in range(len(w[i])):
                for k in range(len(w[i][j])):
                    if np.random.random() > 0.2:
                        w[i][j][k] = 1 * np.random.rand() - 0.5
        return w, b

    def variant(self):
        n = np.random.random()
        if n <= 0.66:
            w = [[0.025 * np.random.random() - 0.0125 + i for i in x] for x in self.weights]
            b = [0.025 * np.random.random() - 0.0125 + b for b in self.biases]
        elif n <= 0.33:
            w = [[0.05 * np.random.random() - 0.025 + i for i in x] for x in self.weights]
            b = [0.05 * np.random.random() - 0.025 + b for b in self.biases]
        else:
            w = [[0.5 * np.random.random() - 0.25 + i for i in x] for x in self.weights]
            b = [0.5 * np.random.random() - 0.25 + b for b in self.biases]
        return w, b
