"""
MLP (Multi-Layer Perceptron) — Built from Scratch using NumPy
=============================================================
No PyTorch, no TensorFlow — just math + NumPy.
Perfect for interviews: you understand EVERY line.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# ===========================================================
# ACTIVATION FUNCTIONS
# Each neuron applies an activation function to decide
# how much signal to pass forward.
# ===========================================================

def sigmoid(z):
    """
    Sigmoid: squashes any value into (0, 1).
    Used in output layer for binary classification.
    Formula: 1 / (1 + e^-z)
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip for numerical stability

def sigmoid_derivative(z):
    """Gradient of sigmoid — needed during backpropagation."""
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    """
    ReLU (Rectified Linear Unit): max(0, z)
    Most popular activation for hidden layers.
    Fast to compute, avoids vanishing gradient.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """Gradient of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)

def softmax(z):
    """
    Softmax: converts raw scores into probabilities that sum to 1.
    Used in output layer for multi-class classification.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # subtract max for stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ===========================================================
# LOSS FUNCTIONS
# Measure how wrong our predictions are.
# ===========================================================

def binary_cross_entropy(y_true, y_pred):
    """
    Loss for binary classification.
    Formula: -mean(y*log(p) + (1-y)*log(1-p))
    """
    eps = 1e-9  # avoid log(0)
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

def categorical_cross_entropy(y_true, y_pred):
    """
    Loss for multi-class classification.
    y_true should be one-hot encoded.
    """
    eps = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


# ===========================================================
# THE MLP CLASS
# ===========================================================

class MLP:
    """
    Multi-Layer Perceptron (Feedforward Neural Network)

    Architecture:
        Input Layer → Hidden Layer(s) → Output Layer

    Each layer has:
        - Weights (W): how much each input matters
        - Biases (b): offset for each neuron
        - Activation: non-linearity (ReLU for hidden, Sigmoid/Softmax for output)

    Training uses:
        - Forward Pass: compute predictions
        - Loss: measure error
        - Backward Pass: compute gradients via chain rule
        - Gradient Descent: update W and b to reduce loss
    """

    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000,
                 hidden_activation='relu', output_activation='sigmoid',
                 verbose=True):
        """
        Args:
            layer_sizes   : list like [input_dim, hidden1, hidden2, ..., output_dim]
                            e.g. [8, 16, 8, 1] means 8 inputs → 16 → 8 → 1 output
            learning_rate : step size for gradient descent
            epochs        : number of full passes over the training data
            hidden_activation : 'relu' or 'sigmoid'
            output_activation : 'sigmoid' (binary) or 'softmax' (multiclass)
            verbose       : print loss every 100 epochs
        """
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.verbose = verbose
        self.loss_history = []

        # Initialize weights and biases
        self.params = self._initialize_parameters()

    def _initialize_parameters(self):
        """
        He Initialization for ReLU — prevents vanishing/exploding gradients.
        W ~ N(0, sqrt(2/fan_in))
        b = zeros
        """
        params = {}
        n_layers = len(self.layer_sizes)

        for l in range(1, n_layers):
            fan_in = self.layer_sizes[l - 1]
            fan_out = self.layer_sizes[l]

            # He init for ReLU, Xavier-like for others
            scale = np.sqrt(2.0 / fan_in) if self.hidden_act == 'relu' else np.sqrt(1.0 / fan_in)

            params[f'W{l}'] = np.random.randn(fan_in, fan_out) * scale
            params[f'b{l}'] = np.zeros((1, fan_out))

        return params

    def _activate(self, Z, layer_type='hidden'):
        """Apply the correct activation function."""
        if layer_type == 'hidden':
            if self.hidden_act == 'relu':
                return relu(Z)
            else:
                return sigmoid(Z)
        else:  # output layer
            if self.output_act == 'softmax':
                return softmax(Z)
            else:
                return sigmoid(Z)

    def _activate_derivative(self, Z):
        """Derivative of hidden layer activation (for backprop)."""
        if self.hidden_act == 'relu':
            return relu_derivative(Z)
        else:
            return sigmoid_derivative(Z)

    def forward(self, X):
        """
        FORWARD PASS
        ============
        Data flows: X → Layer1 → Layer2 → ... → Output

        For each layer l:
            Z[l] = A[l-1] @ W[l] + b[l]   (linear transformation)
            A[l] = activation(Z[l])         (non-linearity)

        Returns:
            cache: all intermediate values needed for backprop
        """
        cache = {'A0': X}
        n_layers = len(self.layer_sizes) - 1

        A = X
        for l in range(1, n_layers + 1):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']

            Z = A @ W + b  # Linear step: (n_samples, fan_in) @ (fan_in, fan_out)

            layer_type = 'output' if l == n_layers else 'hidden'
            A = self._activate(Z, layer_type)

            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A

        return A, cache  # A is the final prediction

    def compute_loss(self, y_true, y_pred):
        """Calculate how wrong we are."""
        if self.output_act == 'softmax':
            return categorical_cross_entropy(y_true, y_pred)
        else:
            return binary_cross_entropy(y_true, y_pred)

    def backward(self, y_true, cache):
        """
        BACKWARD PASS (Backpropagation)
        ================================
        Uses the chain rule to compute how much each weight
        contributed to the loss.

        dL/dW[l] = A[l-1].T @ dZ[l]
        dL/db[l] = mean(dZ[l], axis=0)
        dL/dA[l-1] = dZ[l] @ W[l].T

        Then dZ[l-1] = dL/dA[l-1] * activation'(Z[l-1])
        """
        grads = {}
        n_layers = len(self.layer_sizes) - 1
        m = y_true.shape[0]  # number of samples

        # ---- Output layer gradient ----
        # For sigmoid + binary cross-entropy OR softmax + categorical cross-entropy,
        # the gradient simplifies nicely to: dZ = A_out - y_true
        A_out = cache[f'A{n_layers}']
        dZ = A_out - y_true  # shape: (m, output_dim)

        # ---- Propagate backwards through each layer ----
        for l in range(n_layers, 0, -1):
            A_prev = cache[f'A{l-1}']
            W = self.params[f'W{l}']

            grads[f'dW{l}'] = (A_prev.T @ dZ) / m
            grads[f'db{l}'] = np.mean(dZ, axis=0, keepdims=True)

            if l > 1:  # Don't need dA for the input layer
                dA_prev = dZ @ W.T
                dZ = dA_prev * self._activate_derivative(cache[f'Z{l-1}'])

        return grads

    def update_parameters(self, grads):
        """
        GRADIENT DESCENT UPDATE
        ========================
        W = W - lr * dW
        b = b - lr * db

        We move weights in the opposite direction of the gradient
        (downhill on the loss surface).
        """
        n_layers = len(self.layer_sizes) - 1
        for l in range(1, n_layers + 1):
            self.params[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= self.lr * grads[f'db{l}']

    def fit(self, X_train, y_train):
        """
        TRAINING LOOP
        =============
        Repeat for each epoch:
            1. Forward pass → get predictions
            2. Compute loss
            3. Backward pass → get gradients
            4. Update weights
        """
        for epoch in range(self.epochs):
            # Step 1: Forward pass
            y_pred, cache = self.forward(X_train)

            # Step 2: Compute loss
            loss = self.compute_loss(y_train, y_pred)
            self.loss_history.append(loss)

            # Step 3: Backward pass
            grads = self.backward(y_train, cache)

            # Step 4: Update weights
            self.update_parameters(grads)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    def predict_proba(self, X):
        """Get probability scores."""
        proba, _ = self.forward(X)
        return proba

    def predict(self, X, threshold=0.5):
        """Get binary class predictions."""
        proba = self.predict_proba(X)
        if self.output_act == 'softmax':
            return np.argmax(proba, axis=1)
        else:
            return (proba >= threshold).astype(int).flatten()

    def plot_loss(self):
        """Visualize training loss over epochs."""
        plt.figure(figsize=(8, 4))
        plt.plot(self.loss_history, color='steelblue', linewidth=2)
        plt.title('Training Loss over Epochs', fontsize=14)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('loss_curve.png', dpi=150)
        plt.show()
        print("Loss curve saved as loss_curve.png")

    def summary(self):
        """Print architecture summary."""
        print("\n" + "="*50)
        print("      MLP Architecture Summary")
        print("="*50)
        total_params = 0
        for l in range(1, len(self.layer_sizes)):
            W = self.params[f'W{l}']
            b = self.params[f'b{l}']
            n_params = W.size + b.size
            total_params += n_params
            layer_type = "Output" if l == len(self.layer_sizes) - 1 else "Hidden"
            print(f"Layer {l} ({layer_type}): {self.layer_sizes[l-1]} → {self.layer_sizes[l]} | "
                  f"W: {W.shape} | b: {b.shape} | Params: {n_params}")
        print(f"\nTotal trainable parameters: {total_params}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Hidden activation: {self.hidden_act}")
        print(f"Output activation: {self.output_act}")
        print("="*50 + "\n")


# ===========================================================
# MAIN: Run the full pipeline
# ===========================================================

if __name__ == "__main__":
    print("\n🚀 Building and Training a Multi-Layer Perceptron from Scratch\n")

    # ----- 1. Generate a synthetic dataset -----
    print("Step 1: Generating dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    y = y.reshape(-1, 1)  # Make y a column vector: (1000, 1)

    # ----- 2. Split into train/test -----
    print("Step 2: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- 3. Normalize the features -----
    # Neural networks train much better when inputs are on the same scale.
    print("Step 3: Normalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ----- 4. Build the MLP -----
    # Architecture: 8 inputs → 16 neurons → 8 neurons → 1 output (binary)
    print("Step 4: Building MLP...")
    mlp = MLP(
        layer_sizes=[8, 16, 8, 1],  # [input, hidden1, hidden2, output]
        learning_rate=0.05,
        epochs=500,
        hidden_activation='relu',
        output_activation='sigmoid',
        verbose=True
    )

    mlp.summary()

    # ----- 5. Train -----
    print("Step 5: Training...\n")
    mlp.fit(X_train, y_train)

    # ----- 6. Evaluate -----
    print("\nStep 6: Evaluating on test set...")
    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ----- 7. Plot loss -----
    print("Step 7: Plotting loss curve...")
    mlp.plot_loss()

    print("\n✅ Done! MLP trained successfully.\n")
