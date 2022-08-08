import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import itertools

# Metadaten für das Programm
data = []
labels = []
height = 30
width = 30
channels = 3
classes = 43
input_size = height * width * channels

# Verkehrsschilder
sign_label = ["20 km/h","30 km/h","50 km/h","60 km/h","70 km/h","80 km/h","80 km/h Aufhebung","100 km/h",
          "120 km/h", "Überholverbot", "LKW-Überholverbot", "Vorfahrt", "Vorfahrtsstraße","Vorfahrt gewähren",
          "Stop","Fahrverbot","Verbot für Lastwagen","Einfahrt verboten","Gefahr","Linkskurve","Rechtskurve","Doppelkurve",
          "Bodenwelle","Schleudergefahr","Verengung","Baustelle","Ampel","Zebrastreifen","Kinder","Fahrradweg","Schneegefahr",
          "Wildwechsel","Unbegrenzte Geschwindigkeit","Rechtsabbiegen", "Linksabbiegen", "Geradeaus fahren", "Geradeaus oder Rechtsabbiegen",
          "Geradeaus oder Linksabbiegen", "Hindernis rechts umfahren", " Hindernis links umfahren", "Kreisverkehr", "Ende des Überholverbotes",
          "Ende des LKW-Überholverbotes"]

# Einlesen der Bild-Datein
for i in range(classes):
    path = r"./Data/Train/{0}/".format(i)
    print(path)
    Class = os.listdir(path)
    for a in Class:
        try:
            image = cv2.imread(path + a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

# Zusammenfassen aller Bilder in einem Array
Cells = np.array(data)
labels = np.array(labels)
print("Gesamtzahl der Bilder:",len(Cells))

# Darstellen der Bias des Datensets
from collections import Counter
c = Counter(labels)
plt.bar(c.keys(), c.values())
plt.show()

# Vermischen der Daten und Labels
s = np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells = Cells[s]
labels = labels[s]

# Beispiele aus dem Dataset
fig=plt.figure(figsize=(10, 10))
columns = 3
rows = 3

for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(Cells[i])
    plt.title(sign_label[labels[i]])
    plt.axis("off")
plt.show()

# Aufteilen der Daten in ein Trainings- und ein Validierungsset
(x_train,x_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
x_train = x_train.astype('float32')/255
x_val = x_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

print("Größe des Trainingsets:    ",len(x_train))
print("Größe des Validierungssets:",len(x_val))

# Die Labels mit einem One-Hot-Encoder kategorisieren
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

# Aufstellen des Netztes und Trainieren des Netztes
import tensorflow
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import datetime, os

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

"""inputs = tensorflow.keras.layers.Input(shape = x_train.shape[1:])

x = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(inputs)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.5)(x)
output = Dense(43, activation='softmax')(x)

model = tensorflow.keras.Model(inputs = inputs, outputs = output)"""

# Kompilieren des Models
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

epochs = 20
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks = tensorboard_callback)

model.save("Traffic_Sign_Net")

img = np.reshape(x_train[0:10], (10, height, width, 3))
file_writer = tf.summary.create_file_writer(logdir)
with file_writer.as_default():
      tf.summary.image("Training data", img, step=0)

y_test=pd.read_csv(r'.\Data\Test.csv')
names =y_test['Path'].to_numpy()
y_test =y_test['ClassId'].values
data=[]

from tqdm import tqdm
for f in tqdm(names):
    image=cv2.imread(r'.\Data\Test/'+f.replace('Test', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

fig=plt.figure(figsize=(10, 10))
columns = 3
rows = 3

for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(data[i])
    plt.title(sign_label[y_test[i]])
    plt.axis("off")
plt.show()

x_test=np.array(data)
x_test = x_test.astype('float32')/255
pred = model.predict(x_test)

y_eva_test = to_categorical(y_test, 43)
score = model.evaluate(x_test, y_eva_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

meta_data=[]
for f in range(42):
    image=cv2.imread(".\Data/Meta/"+str(f)+".png")
    image_from_array = Image.fromarray(image, 'RGB')
    meta_data.append(np.array(image_from_array))

x = 1699  #63, 331, 1699

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (12, 12))

y = [i for i in range(len(sign_label))]

ax1.imshow(data[x])
ax2.bar(y, pred[x])
ax3.imshow(meta_data[np.argmax(pred[x])])

asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
ax2.set_aspect(asp)

ax1.axis("off")
ax3.axis("off")

ax1.title.set_text("Input: "+str(sign_label[y_test[x]])) #("Input")
ax2.title.set_text("Zuversicht: "+ str(max(pred[x])))
ax3.title.set_text("Vorhersage")#("Vorhersage: "+sign_label[np.argmax(pred[x])])
plt.show()