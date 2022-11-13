import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title("fetch_lfw_people")
st.markdown("Labeled Faces in the Wild (LFW) people dataset (classification)")

st.header("Faces Features")
col1, col2 = st.columns(2)
with col1:
        st.text("Estimater characteristics")
        sepal_l = st.slider('Estimater lenght ', 1.0, 8.0, 0.5)
        sepal_w = st.slider('Estimater width ', 2.0, 4.4, 0.5)
with col2:
        st.text("Accuracy characteristics")
        petal_l = st.slider('Accuracy lenght', 1.0, 7.0, 0.5)
        petal_w = st.slider('Accuracy width', 0.1, 2.5, 0.5)


if st.button("Predict type of People"):
        result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
        st.text(result[0])