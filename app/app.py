from pathlib import Path
from PIL import Image

import pandas as pd
import streamlit as st

path = Path(__file__).absolute()

st.set_page_config(
    page_title="UNGA 2023 Speech Analysis App",
    # page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

df_speech_url = pd.read_csv(path / "UN Speeches.csv")

countries = df_speech_url["country"].unique()
country_option = st.sidebar.selectbox("", sorted(list(countries)))

col1, col2 = st.columns(2)
with col1:
    image = Image.open(path / "wordclouds" / f"{country_option}_words.png")
    st.image(image, caption=f"Wordcloud {country_option}")
with col2:
    st.video(df_speech_url[df_speech_url["country"] == country_option]["url"].values[0])
