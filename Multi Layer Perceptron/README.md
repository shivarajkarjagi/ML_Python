# 🧠 Multi-Layer Perceptron (MLP) from Scratch

Built using **pure NumPy** — no PyTorch, no TensorFlow. Every line of math is visible and explained.

---

## 📁 Project Structure

```
mlp_project/
├── mlp.py            ← Main MLP code (fully commented)
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mlp_project.git
cd mlp_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the MLP
python mlp.py
```

---

## 🧠 What is an MLP?

A **Multi-Layer Perceptron** is the simplest form of a deep neural network. It consists of:

```
Input Layer → Hidden Layer(s) → Output Layer
```

Each layer is **fully connected**: every neuron in one layer connects to every neuron in the next.

---

## 🔄 How Training Works (Step by Step)

### 1️⃣ Forward Pass
Data flows from input to output.

```
Z = X @ W + b        ← Linear transformation
A = activation(Z)    ← Non-linearity
```

### 2️⃣ Loss Calculation
Measures how wrong the predictions are.

- Binary classification → **Binary Cross-Entropy**
- Multi-class → **Categorical Cross-Entropy**

### 3️⃣ Backward Pass (Backpropagation)
Uses the **chain rule** to compute how much each weight contributed to the error.

```
dZ = A_out - y_true        ← Output layer gradient
dW = A_prev.T @ dZ / m    ← Weight gradient
db = mean(dZ)              ← Bias gradient
dZ_prev = dZ @ W.T * activation'(Z_prev)   ← Propagate back
```

### 4️⃣ Gradient Descent
Update weights to reduce loss.

```
W = W - lr * dW
b = b - lr * db
```

---

## ⚙️ Architecture Used

```
Input (8) → Hidden1 (16, ReLU) → Hidden2 (8, ReLU) → Output (1, Sigmoid)
```

| Layer | Neurons | Activation |
|-------|---------|------------|
| Input | 8 | — |
| Hidden 1 | 16 | ReLU |
| Hidden 2 | 8 | ReLU |
| Output | 1 | Sigmoid |

---

## 🎯 Key Concepts for Interviews

| Concept | What to Say |
|---|---|
| **Activation Function** | Adds non-linearity so the network can learn complex patterns |
| **ReLU** | `max(0, z)` — fast, avoids vanishing gradient |
| **Sigmoid** | `1/(1+e^-z)` — squashes to (0,1), used in binary output |
| **Backpropagation** | Chain rule applied backwards to compute weight gradients |
| **Learning Rate** | Controls step size in gradient descent — too high = diverge, too low = slow |
| **He Initialization** | Initialize weights with `sqrt(2/fan_in)` — prevents dead ReLUs |
| **Vanishing Gradient** | Gradients shrink as they go back through layers — ReLU helps |
| **Overfitting** | Model memorizes training data — fix with dropout, regularization, more data |

---

## 📊 Expected Output

```
Epoch   0 | Loss: 0.6931
Epoch 100 | Loss: 0.4521
Epoch 200 | Loss: 0.3102
...
✅ Test Accuracy: ~90%
```

---

## 📤 Push to GitHub

```bash
git init
git add .
git commit -m "Add MLP from scratch using NumPy"
git remote add origin https://github.com/YOUR_USERNAME/mlp_project.git
git push -u origin main
```

---

## 🧪 Customize It

```python
# Change architecture
mlp = MLP(layer_sizes=[8, 32, 16, 8, 1], ...)

# Change learning rate or epochs
mlp = MLP(..., learning_rate=0.01, epochs=1000)

# Use softmax for multi-class
mlp = MLP(..., output_activation='softmax')
```
