import streamlit as st
import numpy as np
import cv2
import random
import tensorflow as tf
from streamlit_plotly_events import plotly_events

from adverserial_tools import *
from ai_tools import make_prediction, show_correlation, load_meta_data, show_example


if 'Rand_image' not in st.session_state:
    st.session_state['Rand_image'] = 0

if 'Perturbation' not in st.session_state:
    st.session_state['Perturbation'] = 0

if 'Amplifier' not in st.session_state:
    st.session_state['Amplifier'] = 0

if 'Adverserial_image' not in st.session_state:
    st.session_state['Adverserial_image'] = 0

st.title("🥷 Adverserial Attack")
"""
An adversarial attack is a deliberate manipulation of input to an AI by an attacker in 
order to provoke an erroneous output from the AI. It is important to note that the 
manipulation has no effect on human perception, but only on AI processes. 
The attack is successful if the human eye does not notice any difference between a manipulated 
and an original image. In this context, there is often talk of "optical illusions" 
for artificial intelligences.

The Fast Gradient Sign Method (FGSM) was the first demonstrably successful attack on a Deep Neural Net (DNN) in 2014. 
The method was invented in the same year by Ian J. Goodfellow, Jonathon Shlens and Christian Szegedy at Google. It 
can be performed as an untargeted or targeted attack and assumes a white box scenario, as it determines a disturbance 
(also called noise) based on the parameters of the attacked network. The noise is then added to the image with a 
factor (in this case 0.007) and the manipulation is complete (see figure). """

st.image("https://miro.medium.com/max/1400/1*PmCgcjO3sr3CPPaCpy5Fgw.png",
         caption="Famous example of an adverserial attack (Source: "
                 "https://miro.medium.com/max/1400/1*PmCgcjO3sr3CPPaCpy5Fgw.png)")

test_images, test_labels = load_test_data()
meta_images = load_meta_data()

st.title("🥷 Single Attack")
"""
The FGSM can be represented as the following mathematical function:
"""

st.latex(r"""
M = x + \epsilon * Noise
""")

"""
This can be divided into two parts. The first part describes the derivation of the image-specific noise: 
"""

st.latex(r"""
Noise = sign(\nabla J(f,x,y))
""")

"""The FGSM method is based on the same principle as the backpropagation of a neural network. The neural network can 
be understood as a function **f**, the parameters of the network as **x** and the error of the neural network as 
**y**.  If one now sets up a loss function **J(f,x,y)**, you can calculate the corresponding gradient for each parameter 
value of **x** in order to reduce the error of the network **y**. The FGSM method now does the same except that **x** are no 
longer the parameters of the neural network but the pixels of the input image. Accordingly, it can now be calculated 
to what extent the pixels of the image have to be adjusted in order to maximise the loss of the neural network. 

In the second part of the formula, the noise created in the first part is added to the original image with a factor 
**𝛆**, to create the corresponding manipulated image. Without this factor, the noise would paint over the entire 
image and the attack would be very clearly recognisable to humans. This leads back to the complete formula: """

st.latex(r"""
M = x + \epsilon ∗ sign(\nabla J(f,x,y))
""")

"""
This mathematical function can be implemented as a function in Python as follows:
"""

code = """
def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(image) # Setting the Focus on the Image's Pixel
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)
    
    gradient = tape.gradient(loss, image) #Calculating the Gradient in Context of the image
    
    signed_grad = tf.sign(gradient)
    
    return signed_grad
"""

st.code(code)

"""
The return of this function is the noise that can now be added to the original image with a **𝛆**. 
In the following section you can carry out an adverserial attack yourself in a few steps: 
- Choose a random image 
- The adversarial_pattern function creates the corresponding noise
- Choose the value for epsilon and the intensity of the noise accordingly
- See how the prediction of the CNN changes
"""

col1, col2, col3, col4, col5 = st.columns([4, 1, 4, 1, 4])
with col1:

    select_button = st.button("1️⃣ Select random Sign")

    if select_button:
        random.seed()
        st.session_state['Rand_image'] = random.randint(0, 12000) #408

    image = test_images[st.session_state['Rand_image']]          #x_test
    image_label = test_labels[st.session_state['Rand_image']]    #y_test

    rand_fig, pred_rand_df, one_hot = make_prediction(image, image_label=image_label, pred_size=3 ,resize=True, one_hot=True)

    config = {'displayModeBar': False}
    st.plotly_chart(rand_fig, use_container_width=True, config=config)
    st.write("Prediction on the original Image:")
    st.table(pred_rand_df)

with col2:
    st.header("")
    st.header("")
    st.header("")
    st.header("➕️")

with col3:

    st.header("")
    st.subheader("")

    st.session_state['Perturbation'] = adversarial_pattern(image, one_hot)
    st.pyplot(generate_noise_image(st.session_state['Perturbation']))
    st.session_state['Amplifier'] = st.slider("2️⃣ Noise Amplification",
                                              min_value= 0.0,
                                              max_value=0.5,
                                              step=0.01)

with col4:
    st.header("")
    st.header("")
    st.header("")
    st.header("🟰️")

with col5:
    image_adver = np.array(image)
    image_adver = image_adver.astype('float32') / 255
    image_adver = image_adver.reshape(1, 30, 30, 3)
    image_adver = tf.cast(image_adver, tf.float32)

    st.session_state['Adverserial_image'] = image_adver + st.session_state['Perturbation'] * st.session_state['Amplifier']

    st.header("")
    st.subheader("")

    adver_fig, pred_adver_df = adverserial_prediction(st.session_state['Adverserial_image'],pred_size=3 , resize=True)
    st.plotly_chart(adver_fig, use_container_width=True, config=config)
    st.write("Prediction on the manipulated Image:")
    st.table(pred_adver_df)

st.info("👆 Not all attacks are always successful, for some images the noise may have to become too strong to have an "
        "effect on the neural network.")

st.header("🥷 Overall Attack")

"""To assess the overall effectiveness of the attack, each of the 12,630 images in the test dataset is attacked 
simultaneously to conclude this work. 𝛆 is set to a fixed value of 0.03 for the subsequent attack. The attack thus 
takes about five minutes on a very powerful machine. 

"""

adverserial_predictions = load_adverserial_predictions()
col1, col2, col3 = st.columns([1,1,1])
with col2:
    checkbox = st.checkbox("Check this Box to see the result for the whole dataset")

if checkbox:
    col1, col2 = st.columns([6, 2])
    with col1:
        corr, df_perc = show_correlation(test_labels=test_labels,
                                         preprocessed=True,
                                         predictions=adverserial_predictions)

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
            st.info("Click on the Confusion Matrix to gain more information")

"""The attack carried out in the context of this work was successful and revealed a lot about the CNN. By means of an 
adversarial attack, uncertainties within artificial intelligences can be revealed and insights gained into the still 
incomprehensible way in which neural networks work. Above all, errors in the training data used became clear, 
which should be avoided in a new training. In addition, an adversarial training method could become part of the 
training. In such a method, manipulated data is generated in addition to the normal training data, which also flows 
into the training. The problem with this method of defence is that in most cases it is only successful against the 
method of attack that was used to create the manipulated training data set; when a new method is used, the artificial 
intelligences are back to square one. OpenAI, one of the leading research companies in the field of artificial 
intelligence, described the problem in 2017 as follows: "Every strategy we have tested so far fails because it is not 
adaptive: it may block one kind of attack, but it leaves another vulnerability open to an attacker who knows about 
the defence being used." """

"""Finally, you now have the possibility to download the codes of the CNN and the Adverserial Attack as Jupiter 
notebook from my github. I hope you were able to learn a bit and I am happy about any feedback. """
#
# col1, col2 = st.columns([1,1])
# with col1:
#     st.download_button(label="📥 Download the Traffic Sign AI Notebook",
#                        data="Traffic_Sign_AI_Clean.ipynb",
#                        file_name='Traffic_Sign_AI.ipynb', )
# with col2:
#     st.download_button(label="📥 Download the Adversarial Attack Notebook",
#                        data="Adversarial_Attacks.ipynb",
#                        file_name='Adversarial_Attacks.ipynb', )
col1, col2, col3 = st.columns([2,1,2])
with col2:
    but = st.button("You have made it to the end!")
    if but:
        st.balloons()
