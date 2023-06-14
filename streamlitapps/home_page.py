import streamlit as st
from health_query import set_background, sidebar_bg


def app():
    set_background('denoised_aipictureResized-PhotoRoom (2).png')
    sidebar_bg('AiDoc.jpg')
    st.title('Home')


    st.info('Welcome to our AI Health Assistant application. This interactive tool leverages advanced Artificial Intelligence technologies including Machine Learning (ML), to provide users with insightful healthcare information.')
    st.info('Whether you have questions about symptoms, medications, or healthcare procedures, our AI Health Assistant is here to help.')
    