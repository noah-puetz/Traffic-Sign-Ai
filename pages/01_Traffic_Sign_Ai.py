import streamlit as st
from Traffic_Sign_Ai import ai_tools
from streamlit_plotly_events import plotly_events

images, labels = ai_tools.load_data()
test_images, test_labels = ai_tools.load_test_data()
meta_images = ai_tools.load_meta_data()

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

