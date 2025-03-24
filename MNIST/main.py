import numpy as np
from sklearn.datasets import fetch_openml

# Fetch MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract images and labels
images = mnist.data.astype('float32')
labels = mnist.target.astype('int32')

# Normalize images to the range [0, 1]
images /= 255.0

# Combine images and labels into one array and shuffle
data = np.column_stack((images, labels))
np.random.shuffle(data)

# Split data into development and training sets
dev_size = 1000
x_dev = data[:dev_size, :-1].T
y_dev = data[:dev_size, -1]

x_train = data[dev_size:, :-1].T
y_train = data[dev_size:, -1].astype(int)



# Initialize parameters with He initialization
def init_params():
    w1 = np.random.randn(16, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((16, 1))
    w2 = np.random.randn(16, 16) * np.sqrt(2. / 16)
    b2 = np.zeros((16, 1))
    w3 = np.random.randn(10, 16) * np.sqrt(2. / 16)
    b3 = np.zeros((10, 1))
    return w1, b1, w2, b2, w3, b3

# Activation functions
def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Forward propagation
def forward_prop(w1, b1, w2, b2, w3, b3, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

# One-hot encoding
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

# Derivative of ReLU
def deriv_ReLU(z):
    return z > 0

# Back propagation
def back_prop(z1, a1, z2, a2, z3, a3, w2, w3, x, y):
    m = y.size
    one_hot_y = one_hot(y)
    dz3 = a3 - one_hot_y
    dw3 = (1 / m) * dz3.dot(a2.T)
    db3 = (1 / m) * np.sum(dz3, axis=1, keepdims=True)
    dz2 = w3.T.dot(dz3) * deriv_ReLU(z2)
    dw2 = (1 / m) * dz2.dot(a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * deriv_ReLU(z1)
    dw1 = (1 / m) * dz1.dot(x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    return dw1, db1, dw2, db2, dw3, db3

# Update parameters
def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3

# Get predictions
def get_predictions(a3):
    return np.argmax(a3, axis=0)

# Calculate accuracy
def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

# Gradient descent function
def gradient_descent(x, y, iterations, alpha):
    w1, b1, w2, b2, w3, b3 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_prop(w1, b1, w2, b2, w3, b3, x)
        dw1, db1, dw2, db2, dw3, db3 = back_prop(z1, a1, z2, a2, z3, a3, w2, w3, x, y)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)
        if i % 10 == 0:
            predictions = get_predictions(a3)
            accuracy = get_accuracy(predictions, y)
            print(f"Iteration {i}: Accuracy = {accuracy:.4f}")
    return w1, b1, w2, b2, w3, b3

# Perform gradient descent
w1, b1, w2, b2, w3, b3 = gradient_descent(x_train, y_train, iterations=100, alpha=0.1)
