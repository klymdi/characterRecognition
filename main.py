import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

image_dir = 'C:/PyProjects/neural_networks/CharacterRecognition/DmDataset/'

images = []
labels = []

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    if img_path.endswith('.jpg'):
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        images.append(img_array)
        label = filename.split('-')[0]
        labels.append(label)

images = np.array(images)
labels = np.array(labels)
labels = np.unique(labels)
labels_for_prediction = labels
encoded_labels = np.arange(labels.shape[0])
one_hot_labels = np.eye(labels.shape[0])[encoded_labels]


images = images / 255.0
images = images.reshape(images.shape[0], -1)
ic(images.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(y, y_pred):
    m = y.shape[0]
    y_int = y.astype(int)
    log_prob = -np.log(y_pred[range(m), y_int])
    loss = np.sum(log_prob) / m
    return loss


def backward_propagation(X, y, y_pred, a1, a2, b1, b2, b3, W1, W2, W3, learning_rate=0.01):
    m = y.shape[0]

    dz3 = y_pred.copy()
    dz3[range(m), y.astype(int)] -= 1
    dz3 /= m

    dW3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    dz2 = np.dot(dz3, W3.T) * (a2 * (1 - a2))
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * (a1 * (1 - a1))
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    return W1, b1, W2, b2, W3, b3


def train_network(X, y, W1, b1, W2, b2, W3, b3, num_epochs=500, learning_rate=0.1):
    for epoch in range(num_epochs):
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, W3) + b3
        y_pred = softmax(z3)

        loss = cross_entropy_loss(y, y_pred)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        W1, b1, W2, b2, W3, b3 = backward_propagation(X, y, y_pred, a1, a2, b1, b2, b3, W1, W2, W3, learning_rate)

    return W1, b1, W2, b2, W3, b3


hidden_layer_size1 = 20
hidden_layer_size2 = 20

W1 = np.random.randn(images.shape[1], hidden_layer_size1)
b1 = np.zeros((1, hidden_layer_size1))

W2 = np.random.randn(hidden_layer_size1, hidden_layer_size2)
b2 = np.zeros((1, hidden_layer_size2))

W3 = np.random.randn(hidden_layer_size2, one_hot_labels.shape[1])
b3 = np.zeros((1, one_hot_labels.shape[1]))

W1, b1, W2, b2, W3, b3 = train_network(images, one_hot_labels, W1, b1, W2, b2, W3, b3)


def predict_character(image_path, W1, b1, W2, b2, W3, b3):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, -1) / 255.0

    z1 = np.dot(img_array, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W3) + b3
    y_pred = softmax(z3)

    predicted_class = np.argmax(y_pred)
    return predicted_class


def draw_and_predict(image_path, W1, b1, W2, b2, W3, b3, labels):
    predicted_class = predict_character(image_path, W1, b1, W2, b2, W3, b3)
    predicted_character = labels[predicted_class]

    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Character: {predicted_character}")
    plt.show()


image_to_predict = './predict/k.png'
draw_and_predict(image_to_predict, W1, b1, W2, b2, W3, b3, labels_for_prediction)
