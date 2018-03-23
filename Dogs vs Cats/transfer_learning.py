import os
from glob import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np


def saveTrainImg(dir):
    X = []
    y = []
    f = []
    for class_folder_name in os.listdir(dir):
        class_folder_path = os.path.join(dir, class_folder_name)
        for file in os.listdir(class_folder_path):
            image_path = os.path.join(os.path.join(class_folder_path, file)).replace('\\', '/')
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            print(image_bgr.shape)
            image_bgr = cv2.resize(image_bgr, (128, 128))
            image = image_bgr.reshape(1, 128 * 128 * 3)
            X.append(image[0])
            y.append(class_folder_name)
            f.append('{}'.format(file))
    train = pd.DataFrame(X)
    train['label'] = f
    train['class'] = y
    train.to_csv('image_train.csv')


def saveTestImg(dir):
    X = []
    f = []

    for file in os.listdir(dir):
        image_path = os.path.join(os.path.join(dir, file)).replace('\\', '/')
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_bgr = cv2.resize(image_bgr, (32, 32))
        image = image_bgr.reshape(1, 32 * 32 * 3)
        X.append(image[0])
        f.append('{}'.format(file))
    test = pd.DataFrame(X)
    test['label'] = f
    test.to_csv('image_test.csv')


def plot_images(X, label):
    nb_row = 3
    nb_col = 3
    fig, axs = plt.subplots(nb_row, nb_col, figsize=(6, 6))
    n = 0
    for i in range(nb_row):
        for j in range(nb_col):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(X[n].reshape(32, -1))
            n += 1


################### Main ###################

data_dir = 'data/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')
saveTrainImg(train_dir)
saveTestImg(test_dir)
'''

train = pd.read_csv('image_train.csv')
test = pd.read_csv('image_test.csv')

train.drop(train.columns[0], axis=1, inplace=True)
test.drop(test.columns[0], axis=1, inplace=True)

label_train = train['label'].values
y = train['class'].values
train.drop(['label', 'class'], axis=1, inplace=True)
train = np.float32(train.values)

print(label_train[1])
plt.imshow(train[1].reshape(32, 32, 3))
plt.show()
'''
