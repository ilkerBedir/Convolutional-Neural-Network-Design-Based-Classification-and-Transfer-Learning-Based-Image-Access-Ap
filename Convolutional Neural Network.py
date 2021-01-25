from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation, BatchNormalization, ZeroPadding2D
from keras.models import Sequential, load_model
from keras import utils as np_utils
from keras.optimizers import Adam
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import random
import os
from cv2 import cv2
# resim kategoriler
Categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

# Train ve test dosyalarından okuma ve veri çekme.


def read_image():
    try:
        x = []
        y = []
        for category in Categories:
            path = os.path.join(DATADIR, category)
            print(path)
            os.chdir(path)
            label = Categories.index(category)
            i = 0
            for im in os.listdir(path):
                img = cv2.imread(im, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                x.append(img)
                y.append(label)
                # if i>5000:
                #   break
                i += 1
    except Exception as e:
        print(e)
        pass
    return x, y


# Train ve Test dosyaları alma
DATADIR = os.path.dirname('/content/train/')
x_train, y_train = read_image()
DATADIR = os.path.dirname('/content/test/')
x_test, y_test = read_image()

# Gerekli Train Ve Test Datları karıştırmak için yazılan kod
c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)
d = list(zip(x_test, y_test))
random.shuffle(d)
x_test, y_test = zip(*d)

# Normalizasyon yapımı
x_train = np.array(x_train)  # Numpy arraye çevirme
y_train = np.array(y_train)
x_train = x_train.astype(np.float32)    # Bellek için Float32 ye çevirme
x_train = x_train/255
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.astype(np.float32)
x_test = x_test/255


# bu benim oluşturduğum Convoluttional katmanın burada testleri teker teker denemek için fonksiyonel yazdım
def convolutional_layer(x_train, x_test, y_train, y_test, layer_count, filter_number, filter_size, initializer, activation_function, dropout_number, optimization_algorithm):
    i = 0
    model = Sequential()
    model.add(Conv2D(filter_number, filter_size,
                     activation=activation_function, input_shape=(224, 224, 3)))
    model.add(Dropout(dropout_number))
    i += 1
    while i < layer_count:
        model.add(ZeroPadding2D(1))
        model.add(Conv2D(filter_number, filter_size,
                         activation=activation_function))
        model.add(Dropout(dropout_number))
        i += 1
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimization_algorithm,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    model.fit(x_train, y_train, epochs=10)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# 1.Soru için İstenen Çıktılar Fonksiyonu Buradadır.
convolutional_layer(x_train, x_test, y_train, y_test, 2, 32,
                    3, "glorot_uniform", 'relu', 0.2, 'adam')

convolutional_layer(x_train, x_test, y_train, y_test, 2, 32,
                    5, "glorot_uniform", 'relu', 0.2, 'adam')

convolutional_layer(x_train, x_test, y_train, y_test, 2, 32,
                    3, "glorot_uniform", 'relu', 0.7, 'adam')

convolutional_layer(x_train, x_test, y_train, y_test, 3, 32,
                    3, "glorot_uniform", 'relu', 0.2, 'adam')

convolutional_layer(x_train, x_test, y_train, y_test, 3, 32,
                    3, "glorot_uniform", 'relu', 0.7, 'adam')

convolutional_layer(x_train, x_test, y_train, y_test, 3, 32,
                    5, "glorot_uniform", 'relu', 0.2, 'adam')
