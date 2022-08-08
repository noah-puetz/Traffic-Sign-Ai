import streamlit as st
import ai_tools
from streamlit_option_menu import option_menu
from streamlit_plotly_events import plotly_events

# Streamlit main page configuration
st.set_page_config(page_title="Traffic Sign AI",
                   page_icon="🚦",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={
                       'Get Help': 'https://www.extremelycoolapp.com/help',
                       'Report a bug': "https://www.extremelycoolapp.com/bug",
                       'About': "# This is a header. This is an *extremely* cool app!"
                   })

st.title("Adverserial Attacks")
st.header("🚦 Manipulation of a Traffic Sign Recognition AI")

'''An adversarial attack is a deliberate manipulation of input to an AI by an attacker in 
order to provoke an erroneous output from the AI. It is important to note that the 
manipulation has no effect on human perception, but only on AI processes. 
In most cases, the human eye does not notice any difference between a manipulated 
and an original image. In this context, there is often talk of "optical illusions" 
for artificial intelligences.'''




selected = option_menu(
    menu_title = None,
    options=["Classification AI", "Adverserial Attack"],
    icons=["stoplights","stoplights-fill"],
    orientation = "horizontal"
)

images, labels = ai_tools.load_data()
test_images, test_labels = ai_tools.load_test_data()
meta_images = ai_tools.load_meta_data()

if selected == "Classification AI":
    with st.expander("⛔ Beispiele aus dem Datenset"):
        example_button = st.button("Show random examples of the Dataset")
        ai_tools.show_example(images, labels)

    with st.expander("⛔ Bias of the Dataset"):
        ai_tools.show_bias(labels)

    with st.expander("⛔ Confusion Matrix"):

        col1, col2 = st.columns([6,2])
        with col1:
            corr,df_perc = ai_tools.show_correlation(test_images, test_labels)
            selected_points = plotly_events(corr)
            y = selected_points[0]["pointNumber"][0]
            x = selected_points[0]["pointNumber"][1]
            x_name = selected_points[0]["x"]
            y_name = selected_points[0]["y"]
        with col2:
            st.subheader("Confusion:")
            st.write("Between: "+str(selected_points[0]["x"]))
            st.image(meta_images[x])
            st.write("And: "+str(selected_points[0]["y"]))
            st.image(meta_images[y])
            st.write("Classified: "+str(round(df_perc[x_name][y_name],2)))

    with st.expander("⛔ Make a Prediction"):
        pass

if selected == "Adverserial Attack":
    pass

