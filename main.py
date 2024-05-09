import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from icecream import ic

# './data/{images}' - Dima`s dataset
# './DmDataset/{images}' Donbas`s dataset
image_dir = 'C:/PyProjects/neural_networks/CharacterRecognition/combined/'

images = []
labels = []

for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    # in './data/{images}' dataset ".png" format
    if img_path.endswith('.png') or img_path.endswith('.jpg'):
        img = Image.open(img_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        images.append(img_array)
        label = filename.split('-')[0]
        labels.append(label)

images = np.array(images)
labels = np.array(labels)
ic(labels.shape)

encoded_labels = np.arange(labels.shape[0])
one_hot_labels = np.eye(labels.shape[0])[encoded_labels]
ic(one_hot_labels.shape)

images = images / 255.0
images = images.reshape(images.shape[0], -1)
ic(images.shape)

X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def cross_entropy_loss(y, y_pred):
    m = y.shape[0]
    log_prob = -np.log(y_pred[range(m), np.argmax(y, axis=1)])
    loss = np.sum(log_prob) / m
    return loss

def backward_propagation(X, y, y_pred, a1, b1, W1, W3, b3, learning_rate=0.01):
    m = y.shape[0]
    dz3 = y_pred - y
    dW3 = np.dot(a1.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)
    dz1 = np.dot(dz3, W3.T) * (a1 * (1 - a1))
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W3, b3

def train_network(X_train, y_train, X_test, y_test, W1, b1, W3, b3, num_epochs=501, learning_rate=0.01):
    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        # Forward propagation for training data
        z1_train = np.dot(X_train, W1) + b1
        a1_train = sigmoid(z1_train)
        z3_train = np.dot(a1_train, W3) + b3
        y_pred_train = softmax(z3_train)

        # Calculate training loss
        train_loss = cross_entropy_loss(y_train, y_pred_train)
        train_losses.append(train_loss)

        # Forward propagation for testing data
        z1_test = np.dot(X_test, W1) + b1
        a1_test = sigmoid(z1_test)
        z3_test = np.dot(a1_test, W3) + b3
        y_pred_test = softmax(z3_test)

        # Calculate testing loss
        test_loss = cross_entropy_loss(y_test, y_pred_test)
        test_losses.append(test_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")

        # Backward propagation and update weights
        W1, b1, W3, b3 = backward_propagation(X_train, y_train, y_pred_train, a1_train, b1, W1, W3, b3, learning_rate)

    return W1, b1, W3, b3, train_losses, test_losses

# Continue with the training using X_train, y_train, X_test, y_test
hidden_layer_size = 20

W1 = np.random.randn(X_train.shape[1], hidden_layer_size)
b1 = np.zeros((1, hidden_layer_size))

W3 = np.random.randn(hidden_layer_size, y_train.shape[1])
b3 = np.zeros((1, y_train.shape[1]))

W1, b1, W3, b3, train_losses, test_losses = train_network(X_train, y_train, X_test, y_test, W1, b1, W3, b3)

def predict_new_image(image_path, W1, b1, W3, b3, labels):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, -1) / 255.0

    z1 = np.dot(img_array, W1) + b1
    a1 = sigmoid(z1)

    z3 = np.dot(a1, W3) + b3
    y_pred = softmax(z3)

    predicted_class = np.argmax(y_pred)
    predicted_character = labels[predicted_class]
    return predicted_character

# Path to the new image you want to predict
new_image_path = './predict/о.png'

predicted_character = predict_new_image(new_image_path, W1, b1, W3, b3, labels)
print(f"Predicted Character for the New Image: {predicted_character}")
