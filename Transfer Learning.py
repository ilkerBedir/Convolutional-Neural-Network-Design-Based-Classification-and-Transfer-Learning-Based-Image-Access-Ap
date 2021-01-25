import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras import utils as np_utils
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.applications.vgg16 import preprocess_input
import tensorflow
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import random
import numpy as np
import os
from cv2 import cv2

Categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

# Verileri Okuma burada memory yetersizliğinden dolayı train ve test için 10000 tane resim kullandım.


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
                if i > 500:
                    break
                i += 1
    except Exception as e:
        print(e)
        pass
    return x, y


# Train ve Test verilerini burada kullandım
DATADIR = os.path.dirname('/content/train/')
x_train, y_train = read_image()
DATADIR = os.path.dirname('/content/test/')
x_test, y_test = read_image()

# Karıştırma Fonksiyonumum
c = list(zip(x_train, y_train))
random.shuffle(c)
x_train, y_train = zip(*c)
d = list(zip(x_test, y_test))
random.shuffle(d)
x_test, y_test = zip(*d)

# Normalizasyon
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = x_train.astype(np.float32)
x_train = x_train/25
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test=x_test.astype(np.float32)
x_test=x_test/255

# VGG16 modelinin oluşması
model=tensorflow.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

# Son katmanı yerine kendi Fully Connected ağımızın eklenmesi ve sınıflandırma
my_fc=Dense(1024, activation='relu', name='my_fc')(model.layers[-3].output)
pred=Dense(10, activation='softmax', name='myprediction')(my_fc)
my_model=Model(model.input, pred)

# Son katmanı Eğitilebilir yapma
for i in range(0, 21):
    my_model.layers[i].trainable=False
#Test verilerini kategoriye çevirme
y_train=np_utils.to_categorical(y_train, 10)
y_test=np_utils.to_categorical(y_test, 10)

#Model eğitimi ve tamamlanması,test edilmesi
my_model.compile(optimizer='adam',
                 loss='categorical_crossentropy', metrics=['accuracy'])
my_model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
y_pred=my_model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


my_fc=Dense(1024, activation='relu', name='my_fc')(model.layers[-3].output)
pred=Dense(10, activation='softmax', name='myprediction')(my_fc)
my_model1=Model(model.input, pred)

#Son 4KATMANI EĞİTİLEBİLİR YAPMA
for i in range(0, 18):
    my_model1.layers[i].trainable=False
my_model1.layers[18].trainable=True
my_model1.layers[19].trainable=True
my_model1.layers[20].trainable=True

y_test=np_utils.to_categorical(y_test, 10)

my_model1.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
my_model1.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
y_pred1=my_model1.predict(x_test)
y_pred1=np.argmax(y_pred1, axis=1)
y_test=np.argmax(y_test, axis=1)
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

#Resnet50-Modelini oluşturulması
model_resnet=tensorflow.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
#Son katmanı değiştirme
my_fc = Dense(1024, activation='relu', name='my_fc')(model_resnet.layers[-2].output)
pred = Dense(10, activation='softmax', name='myprediction')(my_fc)
my_model_resnet = Model(model_resnet.input, pred)

# Son katmanı Eğitilebilir yapma
for layer in model_resnet.layers:
  layer.trainable=False
my_model_resnet.layers[-2].trainable=True
my_model_resnet.layers[-1].trainable=True

y_test = np_utils.to_categorical(y_test,10)
#Modeli eğitmek ve test etmek
my_model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_model_resnet.fit(x_train,y_train,batch_size=64,epochs=10,verbose=1)
y_pred2= my_model_resnet.predict(x_test)
y_pred2 = np.argmax(y_pred2, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

# Son 4 katmanı Eğitilebilir yapma
my_model_resnet.layers[-3].trainable=True
my_model_resnet.layers[-4].trainable=True
my_model_resnet.layers[-5].trainable=True

y_test = np_utils.to_categorical(y_test,10)

#Modeli eğitmek ve test etmek
my_model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
my_model_resnet.fit(x_train,y_train,batch_size=64,epochs=10,verbose=1)
y_pred3= my_model_resnet.predict(x_test)
y_pred3 = np.argmax(y_pred3, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))
