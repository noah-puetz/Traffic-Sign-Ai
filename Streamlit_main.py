import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import random
import plotly

# Streamlit main page configuration
st.set_page_config(page_title="Traffic Sign AI",
                   page_icon="🚗",
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://www.extremelycoolapp.com/help',
                       'Report a bug': "https://www.extremelycoolapp.com/bug",
                       'About': "# This is a header. This is an *extremely* cool app!"
                   })
st.title("Adverserial Attacks")
st.header("🚦 Manipulation einer Verkehrsschilderkennung")
st.markdown(
    """Spätestens seit dem „Big Bang des Deep Learning“ im Jahre 2009 sind Begriffe wie künstliche Intelligenz, 
    maschinelles Lernen und neuronale Netze ständige Begleiter nicht nur in der Informatik, sondern auch in unserem 
    alltäglichen Leben. In dem Jahr konnte Nvidia mit einer neuen Generation von Grafikarten die Geschwindigkeit von 
    Deep Learning Systemen verhundertfachen; Der Startschuss für eine Revolution der neuronalen Netze war gegeben. 
    """)

# Verkehrsschilder
sign_label = ["20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h", "80 km/h", "80 km/h Aufhebung", "100 km/h",
              "120 km/h", "Überholverbot", "LKW-Überholverbot", "Vorfahrt", "Vorfahrtsstraße", "Vorfahrt gewähren",
              "Stop", "Fahrverbot", "Verbot für Lastwagen", "Einfahrt verboten", "Gefahr", "Linkskurve", "Rechtskurve",
              "Doppelkurve", "Bodenwelle", "Schleudergefahr", "Verengung", "Baustelle", "Ampel", "Zebrastreifen",
              "Kinder", "Fahrradweg", "Schneegefahr",
              "Wildwechsel", "Unbegrenzte Geschwindigkeit", "Rechtsabbiegen", "Linksabbiegen", "Geradeaus fahren",
              "Geradeaus oder Rechtsabbiegen",
              "Geradeaus oder Linksabbiegen", "Hindernis rechts umfahren", " Hindernis links umfahren", "Kreisverkehr",
              "Ende des Überholverbotes",
              "Ende des LKW-Überholverbotes"]

st.header("⛔ Beispiele aus dem Datenset")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    image_format = st.number_input("Height and Width of the Images",0,50,value=30,step=1)

@st.cache(show_spinner=False)
def load_data(height, width):
    # Metadaten für das Programm
    data = []
    label = []
    height = int(height)
    width = int(width)
    channels = 3
    classes = 43
    input_size = height * width * channels

    # Einlesen der Bild-Datein
    for i in range(classes):
        path = r"./Data/Train/{0}/".format(i)
        # print(path)
        image_class = os.listdir(path)
        for a in image_class:
            image_path = path + a
            if image_path.endswith(".ppm"):

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image_from_array = Image.fromarray(image, 'RGB')
                size_image = image_from_array.resize((height, width))
                data.append(np.array(size_image))
                label.append(i)

            elif image_path.endswith(".csv"):
                pass
            else:
                print(image_path)

    # Zusammenfassen aller Bilder in einem Array
    images = np.array(data)
    label = np.array(label)

    # Zählen des Bias des Datasets
    from collections import Counter
    c = Counter(label)

    # Durchmischen des Dataset
    s = np.arange(images.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    images = images[s]
    label = label[s]

    return images, label

# Beispiele aus dem Dataset
def show_example(sep = False):
    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 3
    random.seed()
    if sep:
        columns += 1
        for i in range(1, rows*4+1,4):
            randint = random.randint(0, len(images))
            rand_image = cv2.split(images[randint])
            rand_label = sign_label[labels[randint]]
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[randint])
            plt.title(rand_label)
            plt.axis("off")
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(rand_image[0], cmap='Reds_r')
            plt.axis("off")
            fig.add_subplot(rows, columns, i+2)
            plt.imshow(rand_image[1], cmap='Greens_r')
            plt.axis("off")
            fig.add_subplot(rows, columns, i + 3)
            plt.imshow(rand_image[2], cmap='Blues_r')
            plt.axis("off")

    else:
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            randint = random.randint(0,len(images))
            plt.imshow(images[randint])
            plt.title(sign_label[labels[randint]])
            plt.axis("off")


    st.pyplot(fig)

with st.spinner("Loading the Image Data with a format of "+str(image_format)+"x"+str(image_format)+" pixels"):
    images, labels = load_data(image_format,image_format)

with col3:
    st.write("")
    colors_separatly = st.checkbox("Displaying every Color channel separately")

with col2:
    st.write("")
    example_button = st.button("Show random examples of the Dataset")

if example_button:
    show_example(colors_separatly)

st.header("⛔ Bias of the Dataset")
from collections import Counter
import pandas as pd
import plotly.express as px
c = Counter(labels)
label = [sign_label[x] for x in list(c.keys())]
dic = {"Count":c.values(),"Label":label}
df = pd.DataFrame(dic)
df = df.sort_values("Count")
plotly_fig = px.bar(df,x="Label",y="Count",template="simple_white")

st.plotly_chart(plotly_fig,use_container_width=True)

st.header("⛔ Training of the AI")
# Aufteilen der Daten in ein Trainings- und ein Validierungsset
(x_train,x_val)=images[(int)(0.2*len(labels)):],images[:(int)(0.2*len(labels))]
x_train = x_train.astype('float32')/255
x_val = x_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

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

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
st.write(model.summary())
