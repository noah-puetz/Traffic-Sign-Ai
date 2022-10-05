import sys
import os
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import random
import cv2
import keras
import tensorflow as tf
import plotly.express as px
from matplotlib import pyplot as plt
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sign_label = ["20 km/h","30 km/h","50 km/h","60 km/h","70 km/h","80 km/h","80 km/h Aufhebung","100 km/h",
          "120 km/h", "Überholverbot", "LKW-Überholverbot", "Vorfahrt", "Vorfahrtsstraße","Vorfahrt gewähren",
          "Stop","Fahrverbot","Verbot für Lastwagen","Einfahrt verboten","Gefahr","Linkskurve","Rechtskurve","Doppelkurve",
          "Bodenwelle","Schleudergefahr","Verengung","Baustelle","Ampel","Zebrastreifen","Kinder","Fahrradweg","Schneegefahr",
          "Wildwechsel","Unbegrenzte Geschwindigkeit","Rechtsabbiegen", "Linksabbiegen", "Geradeaus fahren", "Geradeaus oder Rechtsabbiegen",
          "Geradeaus oder Linksabbiegen", "Hindernis rechts umfahren", " Hindernis links umfahren", "Kreisverkehr", "Ende des Überholverbotes",
          "Ende des LKW-Überholverbotes"]

model = keras.models.load_model("Traffic_Sign_Ai/Traffic_Sign_Net")

@st.cache(show_spinner=False)
def load_test_data(height=30, width=30):
    y_test = pd.read_csv(r'Traffic_Sign_Ai/Data/Test.csv', ";")
    names = y_test['Filename'].to_numpy()
    y_test = y_test['ClassId'].values
    data = []
    for name in names:
        image = cv2.imread(r'Traffic_Sign_Ai/Data/Test/' + name.replace('Test', ''))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

    return data, y_test



def adversarial_pattern(image, label, height=30, width=30, channels=3):
    test_images = np.array(image)
    image = test_images.astype('float32') / 255
    image = image.reshape(1, 30, 30, 3)
    image = tf.cast(image, tf.float32)
    model = keras.models.load_model("Traffic_Sign_Ai/Traffic_Sign_Net")

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)

    signed_grad = tf.sign(gradient)

    return signed_grad

def generate_noise_image(perturbations):
    image = plt.imshow(perturbations[0]).figure #*0.5+0.5).figure
    return image

def adverserial_prediction(image, pred_size= 11, resize=False):

    pred = model.predict(image)
    pred_df = pd.DataFrame({"Signs": sign_label, "Prediction (%)": np.round(pred[0], 5)*100})
    pred_large_df = pred_df.loc[pred_df["Prediction (%)"].nlargest(pred_size).index]

    input_fig = px.imshow(image[0])
    input_fig.update_layout(coloraxis_showscale=False)
    input_fig.update_layout(margin=dict(l=0, b=0, r=0, t=0, pad=0))
    input_fig.update_xaxes(showticklabels=False)
    input_fig.update_yaxes(showticklabels=False)
    input_fig.update_layout(hovermode=False)

    if resize:
        input_fig.update_layout(autosize=False, width=200, height=200)

    return input_fig, pred_large_df


def giant_attack(dataset_images, dataset_labels, iterations=False, max_iterations=0.03, confidence=0.9):
    attacked_dataset = []
    height = 30
    width = 30
    channels = 3

    my_bar = st.progress(0.0)

    for i in range(len(dataset_images)):
        image = dataset_images[i]
        image = image.reshape(1, height, width, channels)

        image_label = dataset_labels[i]
        image_probs = model.predict(image)

        label = tf.one_hot(image_label, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))

        perturbations = adversarial_pattern(image, label)

        if iterations:
            for x in np.arange(0.0, max_iterations, 0.001):

                adversarial = image + perturbations * x
                adversarial = tf.clip_by_value(adversarial, -1, 1)

                original_prediction = sign_label[model.predict(image).argmax()]
                adversarial_prediction = sign_label[model.predict(adversarial).argmax()]

                if original_prediction != adversarial_prediction and \
                        max(model.predict(adversarial)[0]) > confidence:
                    noise = x
                    break
        else:
            adversarial = image + perturbations * max_iterations
            adversarial = tf.clip_by_value(adversarial, -1, 1)

        attacked_dataset.append(np.array(adversarial))
        st.write(j)
        #st.write(len(attacked_dataset), "of", len(dataset_images), "Images", end="\r")

    return attacked_dataset

def load_adverserial_images():
    with open("/Users/noahpuetz/PycharmProjects/Traffic_Sign_AI_Clean/Traffic_Sign_Ai/adversarial_images"
            , "rb") as fp:
        adverserial_images = pickle.load(fp)

    return adverserial_images

def load_adverserial_predictions():
    with open("/Users/noahpuetz/PycharmProjects/Traffic_Sign_AI_Clean/Traffic_Sign_Ai/adversarial_predictions"
            , "rb") as fp:
        adverserial_predictions = pickle.load(fp)

    return adverserial_predictions