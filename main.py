import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# './data/{images}' - Dima`s dataset
# './DmDataset/{images}' Donbas`s dataset
image_dir = 'C:/PyProjects/neural_networks/CharacterRecognition/data/'

alphabet = 'АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ'

images = []
labels = []

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    # in "data" dataset ".png" format
    if img_path.endswith('.png'):
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        images.append(img_array)
        label = filename.split('-')[0]
        label_enc = np.zeros(len(alphabet))
        label_enc[alphabet.index(label)] = 1
        labels.append(label_enc)

images = np.array(images)
ic(images.shape)
labels = np.array(labels)
ic(labels.shape)

images = images.reshape(images.shape[0], -1) / 255.0
ic(images.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(y_pred, y):
    return np.square(y_pred - y)


def backward_propagation(X, y, y_pred, a1, a2, b1, b2, b3, W1, W2, W3, learning_rate=0.01):
    dz3 = (y_pred - y) * (y_pred * (1 - y_pred))

    dW3 = np.dot(dz3, a2.T)
    db3 = dz3

    dz2 = np.dot(W3.T, dz3) * (a1 * (1 - a1))
    dW2 = np.dot(dz2, a1.T)
    db2 = dz2

    dz1 = np.dot(W2.T, dz2) * (a2 * (1 - a2))
    dW1 = np.dot(dz1, X.T)
    db1 = dz1

    return dW1, db1, dW2, db2, dW3, db3


def train_network(X, y, W1, b1, W2, b2, W3, b3, num_epochs=2501, learning_rate=0.1):
    for epoch in range(num_epochs):
        W1_sum, b1_sum, W2_sum, b2_sum, W3_sum, b3_sum, loss_sum = 0, 0, 0, 0, 0, 0, 0
        for y_example, X_example in zip(y, X):
            y_example = y_example.reshape(len(y_example), 1)
            X_example = X_example.reshape(len(X_example), 1)

            z1 = np.dot(W1, X_example) + b1
            a1 = sigmoid(z1)

            z2 = np.dot(W2, a1) + b2
            a2 = sigmoid(z2)

            z3 = np.dot(W3, a2) + b3
            y_pred = sigmoid(z3)

            loss = cost(y_pred, y_example)
            loss_sum += loss


            dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X_example, y_example, y_pred, a1, a2, b1, b2, b3, W1, W2, W3,
                                                          learning_rate)
            W1_sum += dW1
            W2_sum += dW2
            W3_sum += dW3
            b1_sum += db1
            b2_sum += db2
            b3_sum += db3
        if epoch % 100 == 0:
            loss_sum = loss_sum / X.shape[0]
            ic(loss_sum.mean())

        W1 -= (W1_sum / X.shape[0]) * learning_rate
        W2 -= (W2_sum / X.shape[0]) * learning_rate
        W3 -= (W3_sum / X.shape[0]) * learning_rate
        b3 -= (b3_sum / X.shape[0]) * learning_rate
        b2 -= (b2_sum / X.shape[0]) * learning_rate
        b1 -= (b1_sum / X.shape[0]) * learning_rate

    return W1, b1, W2, b2, W3, b3


hidden_layer_size1 = 66
hidden_layer_size2 = 66

W1 = np.random.randn(images.shape[1], hidden_layer_size1).T
b1 = np.zeros((hidden_layer_size1, 1))

W2 = np.random.randn(hidden_layer_size1, hidden_layer_size2).T
b2 = np.zeros((hidden_layer_size2, 1))

W3 = np.random.randn(hidden_layer_size2, 33).T
b3 = np.zeros((33, 1))

W1, b1, W2, b2, W3, b3 = train_network(images, labels, W1, b1, W2, b2, W3, b3)


def predict_character(image_path, W1, b1, W2, b2, W3, b3):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, -1) / 255.0
    img_array = img_array.T

    z1 = np.dot(W1, img_array) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)

    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    prediction = a3.argmax()

    return prediction


def draw_and_predict(image_path, W1, b1, W2, b2, W3, b3):
    predicted_class = predict_character(image_path, W1, b1, W2, b2, W3, b3)
    ic(predicted_class)
    predicted_character = alphabet[predicted_class]
    ic(predicted_character)

    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted Character: {predicted_character}")
    plt.show()


image_to_predict = './predict/t.png'
draw_and_predict(image_to_predict, W1, b1, W2, b2, W3, b3)

