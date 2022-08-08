import numpy as np
import cv2
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import keras
import random
import streamlit as st
from collections import Counter
import plotly.express as px
from sklearn.metrics import confusion_matrix

class traffic_sign_ai():
    height = 30
    width = 30
    channels = 3
    classes = 43
    input_size = height * width * channels

    sign_label = ["20 km/h", "30 km/h", "50 km/h", "60 km/h", "70 km/h", "80 km/h", "80 km/h Aufhebung",
                       "100 km/h",
                       "120 km/h", "Überholverbot", "LKW-Überholverbot", "Vorfahrt", "Vorfahrtsstraße",
                       "Vorfahrt gewähren",
                       "Stop", "Fahrverbot", "Verbot für Lastwagen", "Einfahrt verboten", "Gefahr", "Linkskurve",
                       "Rechtskurve", "Doppelkurve",
                       "Bodenwelle", "Schleudergefahr", "Verengung", "Baustelle", "Ampel", "Zebrastreifen", "Kinder",
                       "Fahrradweg", "Schneegefahr",
                       "Wildwechsel", "Unbegrenzte Geschwindigkeit", "Rechtsabbiegen", "Linksabbiegen",
                       "Geradeaus fahren",
                       "Geradeaus oder Rechtsabbiegen",
                       "Geradeaus oder Linksabbiegen", "Hindernis rechts umfahren", " Hindernis links umfahren",
                       "Kreisverkehr", "Ende des Überholverbotes",
                       "Ende des LKW-Überholverbotes"]

    def __init__(self):
        self.load_data()
        self.load_model()
        self.load_test_data()
        self.load_meta_data()

    def load_data(self, h=height, w=width):
        # Metadaten für das Programm
        data = []
        label = []
        height = int(h)
        width = int(w)
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

        # Durchmischen des Dataset
        s = np.arange(images.shape[0])
        np.random.seed(43)
        np.random.shuffle(s)
        self.train_images = images[s]
        self.train_labels = label[s]

    @st.cache
    def load_model(self):
        self.model = keras.models.load_model("Traffic_Sign_Net")

    @st.cache
    def load_test_data(self, height=30, width=30):
        y_test = pd.read_csv(r'./Data/Test.csv', ";")
        names = y_test['Filename'].to_numpy()
        y_test = y_test['ClassId'].values
        data = []
        for name in names:
            image = cv2.imread(r'./Data/Test/' + name.replace('Test', ''))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))

        self.test_images = data
        self.test_labels = y_test

    @st.cache
    def load_meta_data(self):
        meta_data = []
        for f in range(43):
            image = cv2.imread("./Data/Meta/" + str(f) + ".png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_from_array = Image.fromarray(image, 'RGB')
            meta_data.append(np.array(image_from_array))

        self.meta_images = meta_data

    @st.cache
    def make_prediction(self):
        test_images = np.array(self.test_images)
        image = test_images.astype('float32') / 255
        self.pred = self.model.predict(image)

    @st.cache
    def create_example_fig(self, sep=False):
        fig = plt.figure(figsize=(10, 10))
        columns = 3
        rows = 1
        random.seed()
        if sep:
            columns += 1
            for i in range(1, rows * 4 + 1, 4):
                randint = random.randint(0, len(self.train_images))
                rand_image = cv2.split(self.train_images[randint])
                rand_label = traffic_sign_ai.sign_label[self.train_labels[randint]]
                fig.add_subplot(rows, columns, i)
                plt.imshow(self.train_images[randint])
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
                randint = random.randint(0, len(self.train_images))
                plt.imshow(self.train_images[randint])
                plt.title(traffic_sign_ai.sign_label[self.train_labels[randint]])
                plt.axis("off")

        return fig

    @st.cache
    def create_bias_fig(self):
        c = Counter(self.train_labels)
        label = [traffic_sign_ai.sign_label[x] for x in list(c.keys())]
        dic = {"Count": c.values(), "Label": label}
        df = pd.DataFrame(dic)
        df = df.sort_values("Count")
        bias_fig = px.bar(df, x="Label", y="Count", template="simple_white")
        return bias_fig

    @st.cache
    def create_correlation_fig(self):
        x_test = np.array(self.test_images)
        x_test = x_test.astype('float32') / 255
        pred = self.model.predict(x_test)

        predictions = []
        for i in pred:
            predictions.append(i.argmax())

        cm = confusion_matrix(self.test_labels, predictions)
        df_cm = pd.DataFrame(cm, index=[i for i in traffic_sign_ai.sign_label],
                             columns=[i for i in traffic_sign_ai.sign_label])
        df_perc = pd.DataFrame()
        for i in traffic_sign_ai.sign_label:
            row = (df_cm[i] / df_cm[i].sum()) * 100
            df_perc = pd.concat([df_perc, row], axis=1)
        df_perc.round(0)
        corr_fig = px.imshow(df_perc, text_auto=True, color_continuous_scale='blues_r')
        corr_fig.update_coloraxes(showscale=False)
        corr_fig.update_layout(font=dict(size=5),
                          autosize=True,
                          margin=dict(l=0, r=0, b=0, t=10))
        return corr_fig, df_perc

    @st.cache
    def create_random_prediction_fig(self,x):
        pred_df = pd.DataFrame({"Signs": traffic_sign_ai.sign_label, "Predictions": np.round(self.pred[x], 2)})

        input_fig = px.imshow(self.test_images[x])
        meta_fig = px.imshow(self.meta_images[np.argmax(self.pred[x])])
        pie_fig = px.pie(pred_df, values="Predictions", names="Signs")

        input_fig.update_layout(coloraxis_showscale=False)
        input_fig.update_xaxes(showticklabels=False)
        input_fig.update_yaxes(showticklabels=False)

        meta_fig.update_layout(coloraxis_showscale=False)
        meta_fig.update_xaxes(showticklabels=False)
        meta_fig.update_yaxes(showticklabels=False)

        pie_fig.update_traces(textposition='inside')

        return input_fig, meta_fig, pie_fig