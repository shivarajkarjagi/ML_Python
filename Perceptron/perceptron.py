import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # Activation function (step function)
    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    # Training function
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)

                # Update rule
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    # Prediction function
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)


# 🚀 Run the model
if __name__ == "__main__":
    # Example dataset (AND gate)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iters=10)
    model.fit(X, y)

    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("Weights:", model.weights)
    print("Bias:", model.bias)