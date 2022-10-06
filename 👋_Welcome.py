from ai_tools import *
from PIL import Image

# Streamlit main page configuration
st.set_page_config(page_title="Adverserial Attack",
                   page_icon="🚦",
                   layout="centered",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'About': "This Webapp was made by Noah Pütz"
                   })

st.title("Manipulation of a Traffic Sign Recognition AI 🤖")
st.subheader("Project by Noah Pütz")
col1, col2, col3 = st.columns([10, 1, 9])
with col1:
    """**Welcome!** 👋"""
with col2:
    url = "https://www.linkedin.com/in/noah-puetz/"
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/640px"
        "-LinkedIn_logo_initials.png",
        width=20)
with col3:
    st.write("Check me out on [**LinkedIn**](%s)" % url)

col1, col2, = st.columns([10, 10])
with col1:
    '''
    This webpage demonstrates how to perform an Adversarial Attack using a street sign recognition AI as an example. 
    First, the page "Traffic Sign AI" explains the actual AI, i.e. the goal of the attack. Then, on the page "Adverserial 
    Attack", the execution of the attack is described. 
    '''
    '''The code and a report were developed in the course of a project at the University of Applied Sciences Cologne 
    in 2020. In 2022, I took up the project again to publish it as a streamlit web app. You can download the original 
    report here (by now it is only available in german). '''
    with open("PDFs/Individuelle_Projektarbeit_Adversarial_Attacks_Noah_Pütz.pdf", "rb") as pdf_file:
        PDF = pdf_file.read()

    st.download_button(label="📥 Download the Project Report (german)",
                       data=PDF,
                       file_name='PDFs/Individuelle_Projektarbeit_Adversarial_Attacks_Noah_Pütz.pdf', )
    """If you would like to know more about me, you can either visit my LinkedIn, download my CV or contact me via 
    mail:
    """
    """
    📧 noah.c.puetz@gmail.com 
    """
    with open("PDFs/CV_Noah_Puetz.pdf", "rb") as pdf_file:
        PDF = pdf_file.read()

    st.download_button(label="📥 Download my CV (english)",
                       data=PDF,
                       file_name='PDFs/Individuelle_Projektarbeit_Adversarial_Attacks_Noah_Pütz.pdf', )

    """
    I appreciate any feedback and have fun on the website! 😊
    """
    st.info("👆 Currently, both a light and dark version of the app are available. However, I would recommend the light version! ")
with col2:
    st.image("Data/Noah_Puetz_vertical.jpeg")
