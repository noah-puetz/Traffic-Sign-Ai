import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random
import keras
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from collections import Counter
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import altair as alt
import sys
print(sys.path)

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

model = keras.models.load_model("Traffic_Sign_Ai/Traffic_Sign_Net")


@st.cache(show_spinner=False)
def load_data(height=30, width=30):
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
        path = r"Traffic_Sign_Ai/Data/Train/{0}/".format(i)
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

    # Durchmischen des Dataset
    s = np.arange(images.shape[0])
    np.random.seed(43)
    np.random.shuffle(s)
    images = images[s]
    label = label[s]

    return images, label


def load_model():
    model = keras.models.load_model("Traffic_Sign_Net")
    return model


@st.cache(show_spinner=False)
def load_test_data(height=30, width=30):
    y_test = pd.read_csv(r'Data/Test.csv', ";")
    names = y_test['Filename'].to_numpy()
    y_test = y_test['ClassId'].values
    data = []
    for name in names:
        image = cv2.imread(r'./Data/Test/' + name.replace('Test', ''))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

    return data, y_test

def load_meta_data(height=30, width=30):
    meta_data = []
    for f in range(43):
        image = cv2.imread("./Data/Meta/" + str(f) + ".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_from_array = Image.fromarray(image, 'RGB')
        meta_data.append(np.array(image_from_array))

    return meta_data

def show_image(image):
    fig = px.imshow(image)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(autosize=True,
                      margin=dict(l=0, r=0, b=0, t=0))
    return fig





# Beispiele aus dem Dataset
def show_example(images, labels, sep=False, ):
    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 1
    random.seed()
    if sep:
        columns += 1
        for i in range(1, rows * 4 + 1, 4):
            randint = random.randint(0, len(images))
            rand_image = cv2.split(images[randint])
            rand_label = sign_label[labels[randint]]
            fig.add_subplot(rows, columns, i)
            plt.imshow(images[randint])
            plt.title(rand_label)
            plt.axis("off")
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(rand_image[0], cmap='Reds_r')
            plt.axis("off")
            fig.add_subplot(rows, columns, i + 2)
            plt.imshow(rand_image[1], cmap='Greens_r')
            plt.axis("off")
            fig.add_subplot(rows, columns, i + 3)
            plt.imshow(rand_image[2], cmap='Blues_r')
            plt.axis("off")

    else:
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            randint = random.randint(0, len(images))
            plt.imshow(images[randint])
            plt.title(sign_label[labels[randint]])
            plt.axis("off")

    return fig

def show_bias(labels):
    c = Counter(labels)
    label = [sign_label[x] for x in list(c.keys())]
    dic = {"Count": c.values(), "Label": label}
    df = pd.DataFrame(dic)
    df = df.sort_values("Count")
    plotly_fig = px.bar(df, x="Label", y="Count", template="simple_white")
    st.plotly_chart(plotly_fig, use_container_width=True)

@st.cache(show_spinner=False)
def show_correlation(test_images, y_test):
    x_test = np.array(test_images)
    x_test = x_test.astype('float32') / 255
    pred = model.predict(x_test)

    predictions = []
    for i in pred:
        predictions.append(i.argmax())

    cm = confusion_matrix(y_test, predictions)
    df_cm = pd.DataFrame(cm, index=[i for i in sign_label],
                         columns=[i for i in sign_label])
    df_perc = pd.DataFrame()
    for i in sign_label:
        row = (df_cm[i] / df_cm[i].sum()) * 100
        df_perc = pd.concat([df_perc, row], axis=1)
    df_perc.round(0)
    fig = px.imshow(df_perc, text_auto=True, color_continuous_scale='blues_r')
    fig.update_coloraxes(showscale=False)
    fig.update_layout(font=dict(size=5),
                      autosize=True,
                      margin=dict(l=0,r=0,b=0,t=10))
    return fig, df_perc

def make_prediction(images, meta_images):
    test_images = np.array(images)
    image = test_images.astype('float32') / 255
    pred = model.predict(image)

    x = random.randint(0, 2000)
    pred_df = pd.DataFrame({"Signs": sign_label, "Predictions": np.round(pred[x], 2)})

    input_fig = px.imshow(test_images[x])
    meta_fig = px.imshow(meta_images[np.argmax(pred[x])])
    pie_fig = px.pie(pred_df, values="Predictions", names="Signs")

    input_fig.update_layout(coloraxis_showscale=False)
    input_fig.update_xaxes(showticklabels=False)
    input_fig.update_yaxes(showticklabels=False)

    meta_fig.update_layout(coloraxis_showscale=False)
    meta_fig.update_xaxes(showticklabels=False)
    meta_fig.update_yaxes(showticklabels=False)

    pie_fig.update_traces(textposition='inside')

    return input_fig, meta_fig, pie_fig

