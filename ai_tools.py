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
from collections import Counter
import plotly.express as px
import tensorflow as tf
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

model = keras.models.load_model("Traffic_Sign_Net")

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
        path = r"Data/Train/{0}/".format(i)
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

@st.cache(show_spinner=False)
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
        image = cv2.imread(r'Data/Test/' + name.replace('Test', ''))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

    return data, y_test

def load_meta_data(height=30, width=30):
    meta_data = []
    for f in range(43):
        image = cv2.imread("Data/Meta/" + str(f) + ".png")
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
def show_example(images, labels, sep=False, rand_seed=0):
    fig = plt.figure(figsize=(10, 10))
    columns = 3
    rows = 1
    random.seed(rand_seed)
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
def show_correlation(test_images=None, test_labels=None, preprocessed=False, predictions=None):
    if not preprocessed:
        test_images = np.array(test_images)
        test_images = test_images.astype('float32') / 255

        #st.write(test_images.shape)
        pred = model.predict(test_images)
        predictions = []
        for i in pred:
            predictions.append(i.argmax())

    else:
        pred = predictions

    cm = confusion_matrix(test_labels, predictions)
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

def make_prediction(image, image_label=None ,pred_size= 11, resize=False, one_hot=False):
    test_images = np.array(image)
    image = test_images.astype('float32') / 255
    image = image.reshape(1, 30, 30, 3)
    image = tf.cast(image, tf.float32)

    pred = model.predict(image)
    if one_hot:
        label = tf.one_hot(image_label, pred.shape[-1])
        one_hot_labels = tf.reshape(label, (1, pred.shape[-1]))

    pred_df = pd.DataFrame({"Signs": sign_label, "Prediction (%)": np.round(pred[0], 5)*100})
    pred_large_df = pred_df.loc[pred_df["Prediction (%)"].nlargest(pred_size).index]

    input_fig = px.imshow(test_images)
    input_fig.update_layout(coloraxis_showscale=False)
    input_fig.update_layout(margin=dict(l=0, b=0, r=0, t=0, pad=0))
    input_fig.update_xaxes(showticklabels=False)
    input_fig.update_yaxes(showticklabels=False)
    input_fig.update_layout(hovermode=False)
    if resize:
        input_fig.update_layout(autosize=False, width=200, height=200)

    if one_hot:
        return input_fig, pred_large_df, one_hot_labels
    else:
        return input_fig, pred_large_df

def model_predict(img_file_buffer):
    height = 30
    width = 30

    y = 0

    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    x = int((cv2_img.shape[1] - cv2_img.shape[0]) / 2)
    cv2_img = cv2_img[y:y + cv2_img.shape[0], x:x + cv2_img.shape[0]]
    cv2_img = cv2.resize(cv2_img, (height, width))
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    input_fig, pred_large_df = make_prediction(cv2_img)

    return input_fig, pred_large_df




