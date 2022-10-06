import streamlit as st
from streamlit_plotly_events import plotly_events
from ai_tools import *
import keras

if 'Example_Button' not in st.session_state:
    st.session_state['Example_Button'] = 0

if 'Prediction_Button' not in st.session_state:
    st.session_state['Prediction_Button'] = 0

if 'First_Start' not in st.session_state:
    st.session_state['First_Start'] = True

st.title("🚏 Traffic Sign AI")


images, labels = load_data()
test_images, test_labels = load_test_data()
meta_images = load_meta_data()

"""On this page you can learn more about the "victim" or the "target" of the attack. Since the adversarial attack 
used here is a white box attack, the attacker knows certain information about his target.

Modern image recognition models are almost exclusively programmed with deep neural networks (DNN). The principle by 
which deep learning 
systems are trained and learn can be divided into three main categories; supervised learning, unsupervised learning 
and reinforcement learning. Supervised learning, in contrast to unsupervised or reinforcement learning, 
requires labelled data. Within supervised learning there are two major problems: Regression problems and 
classification problems. As a part of classification problems, image classification deals with almost all visual and 
labelled data. Today's applications are diverse, from diagnostics, where image classification is used to determine 
tumours, to the identification of street signs. 

A data set is needed to train a DNN. The data set, 
which provides the basis for training the attacked DNN is called the "German Traffic Sign Recognition 
Benchmark" or GTSRB. It was first presented as part of a challenge at the International Joint Conference on Neural Networks (IJN). 
on Neural Networks" (IJCNN) 2011 and contains more than 50,000 images and 43 classes. """

st.info("👆 The sources of all information and information that goes beyond what is written here can be found in the "
        "project report (downloadable on the first page).")

st.header("⛔️ Example of the Dataset")
"""
The images are divided into a training folder with 39,209 different images and a test folder with 12,630 different images. 
and a test folder with 12,630 different images. 
The images in the training folder are in turn sorted into further folders according to their class. 
All images in the test folder are intermixed and not marked according to their class.
"""
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    example_button = st.button("Show random examples")
    if example_button:
        st.session_state['Example_Button'] += 1

example_fig = show_example(images, labels, rand_seed=st.session_state['Example_Button'])

st.write(example_fig)

st.header("⛔️ Bias of the Dataset")
"""One challenge presented by the data set is the imbalance of data between the individual categories; in research, 
this is referred to as a "bias". Bias means that categories with more data can be taken more into account during 
training, whereas categories with less data are less considered during training. In the case of the GTSRB, 
the category "50 km/h" contains more than 2,250 images, in contrast to the sign "Attention left turn", which contains 
only 210 different images. This suggests, even before training, that a "50 km/h" sign will be more reliably 
categorised correctly after training is complete. However, since a neural network is supposed to work similarly 
reliably for all categories, this is an undesirable drawback of the data set. Normally, each category is reduced to 
the number of the category with the lowest recordings, but since the number of all training data would then be too 
small, the imbalance is accepted in this case. """
st.info("👆 Move the cursor over the chart to get to know the dataset better!")
show_bias(labels)

st.header("⛔️ The Convolutional Neural Network")

"""For training the network, all input data must be scaled to the input size of the network. The input size of the 
network is 30 pixels in height and 30 pixels in width for each three colour channels.  To complete the preparation, 
the test data is blended and split into a training set (31,368 images) and a validation set (7,841 images). The attacked 
neural network is a "Convolutional Neural Network" (CNN or ConvNet). CNNs were invented by Yann LeCun and 
are mainly used in the machine processing of image and audio data. CNNs are currently one of the most effective ways 
of building an image classification model. This is due to the special way such a network works.  A CNN processes all 
image channels and each pixel always in the context of its neighbouring pixels. The task of a CNN is to 
put the image into a structure that is easier to process, without losing important information that is relevant for 
the prediction. """
img_url="https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png"
st.image(img_url,caption="Image by Wikipedia: https://de.wikipedia.org/wiki/Convolutional_Neural_Network")
with st.expander("See the tensorflow generated summary of the model"):
    model = keras.models.load_model("Traffic_Sign_Net")
    model.summary(print_fn=lambda x: st.text(x))


st.header(" ⛔️ Confusion Matrix")

"""The DNN usually achieves an accuracy of over 97% after training. To get a more precise insight into the training, 
a so-called confusion matrix is used. is used. For better visualisation, no values are shown, only colour tones. On 
both the X- and Y-axis all categories of the data set are listed. The lighter the colour of a box is, the more images 
from the corresponding category were identified as the corresponding other category. It is hoped that the diagonal is 
as bright as possible, which means that all pictures have been correctly assigned. In the case of the trained CNN, 
this bright diagonal is largely given. The outlier is, for example, the class "zebra crossing", which was only 90% 
correctly categorised. The remaining 10% were mainly identified as danger signs. identified. There are two reasons 
for this: Firstly, the signs are relatively similar due to their shape and the black symbol in the middle. symbol in 
the middle. In addition, however, as already mentioned, the "zebra crossing" class had a very low number of training 
data with 210 images. of training data, whereas the class "Dangers" is strongly represented with 1200 pictures. In 
this example, the previously discussed bias becomes noticeable. This leads to the fact that, with similar looking 
input, images are more likely to be categorised as the class that had more test data. This unbalanced training plays 
a role if the artificial intelligence is attacked by manipulations. """

col1, col2 = st.columns([6, 2])
with col1:

    corr, df_perc = show_correlation(test_images, test_labels)
    selected_points = plotly_events(corr)

    if selected_points:
        y = selected_points[0]["pointNumber"][0]
        x = selected_points[0]["pointNumber"][1]
        x_name = selected_points[0]["x"]
        y_name = selected_points[0]["y"]
with col2:
    st.subheader("Confusion:")
    if selected_points:
        st.write(str(round(df_perc[x_name][y_name], 2)) + "% of " + str(selected_points[0]["x"]))
        st.image(meta_images[x])
        st.write("were classified as " + str(selected_points[0]["y"]))
        st.image(meta_images[y])
    else:
        st.info("👆 Click on the Confusion Matrix to gain more information!")

st.header("⛔ Make a Prediction from Test Data set")
"""To get a little hands on experience, you can click through the test data set using the "Make Prediction" Button. 
The percentage next to the picture is to be 
interpreted as the confidence in the prediction of the DNN.

**Important!** The DNN has not seen any of these images during the training.  """

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    pred_button = st.button("Make Prediction")
    if pred_button:
        st.session_state['Prediction_Button'] += 1

    if pred_button:
        image = test_images[st.session_state['Prediction_Button']]
        input_fig, pred_df = make_prediction(image)

if pred_button:
    col1, col2= st.columns([1,1])
    with col1:
        config = {'displayModeBar': False}
        st.plotly_chart(input_fig,use_container_width=True,config=config)
    with col2:
        st.table(pred_df)
else:
    st.info("👆 Press the Button to generate a Prediction from the Test Data set!")

st.header("⛔ Make a Prediction from your Webcam")

"""Last but not least, you now have the possibility to provide your own input data for the DNN. To do this, 
you must enable this webpage in your browser to access the camera (none of the captured images are stored in any 
way). Just take a picture of a street sign you have painted yourself and see what the DNN predicts.

It is very likely that the DNN will not perform too well, as the background and colours may differ greatly from those 
of the dataset. """

st.info("👆 Take a photo with your Webcam of a real or a drawn Street Sign and let the model guess the Sign!")
img_file_buffer = st.camera_input("")
if img_file_buffer is not None:
        image, pred_input_df = model_predict(img_file_buffer)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(image, use_container_width=True)
        with col2:
            st.table(pred_input_df)

