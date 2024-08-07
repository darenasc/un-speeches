from pathlib import Path
from PIL import Image
from random import randrange

import geopandas as gpd
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st

from unga_speeches import speech_analysis as sa

# To download data from nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords")
nltk.download("wordnet")
# end download data from nltk

APP_DIR = Path(__file__).absolute().parent
DATA_DIR = APP_DIR.parent / "data"

### App ###

st.set_page_config(
    page_title="UNGA78 Speech Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)


df_speech_url = pd.read_csv(DATA_DIR / "UN Speeches.csv")
geo_data = gpd.read_file(DATA_DIR / "ne_110m_admin_0_countries.zip")

if "random_initial_country" not in st.session_state:
    st.session_state.random_initial_country = randrange(len(df_speech_url))
    st.session_state.disabled = False

countries = df_speech_url["country"].unique()
country_option = st.sidebar.selectbox(
    "", sorted(list(countries)), index=st.session_state.random_initial_country
)
number_of_words = st.sidebar.slider("How many words?", 10, 200, 42)


try:
    st.sidebar.text(
        f'{geo_data[geo_data["ADMIN"]==country_option]["CONTINENT"].values[0]}'
    )
    st.sidebar.text(
        f'(Economy) {geo_data[geo_data["ADMIN"]==country_option]["ECONOMY"].values[0].split(". ")[-1]}'
    )
    st.sidebar.text(
        f'(Income group) {geo_data[geo_data["ADMIN"]==country_option]["INCOME_GRP"].values[0].split(". ")[-1]}'
    )
    st.sidebar.text(
        f'Population: {geo_data[geo_data["ADMIN"]==country_option]["POP_EST"].apply(int).values[0]:,} (Est. {geo_data[geo_data["ADMIN"]==country_option]["POP_YEAR"].values[0]})'
    )
    st.sidebar.text(
        f'GDP: USD${geo_data[geo_data["ADMIN"]==country_option]["GDP_MD"].apply(int).values[0]:,}M ({geo_data[geo_data["ADMIN"]==country_option]["GDP_YEAR"].apply(int).values[0]})'
    )

except:
    pass

col1, col2 = st.columns(2)
with col1:
    try:
        sa.get_wordcloud(df_speech_url, country_option)
        image = Image.open(DATA_DIR / "wordclouds" / f"{country_option}_words.png")
        st.image(image, caption=f"Wordcloud {country_option}", use_column_width=True)
    except:
        pass

with col2:
    st.video(sa.get_video_url(df_speech_url, country_option))

col3, col4 = st.columns(2)
with col3:
    try:
        st.pyplot(sa.plot_freq_words(country_option, top_n_words=25))
    except:
        st.write("No transcript available :'(")
        pass

with col4:
    try:
        plot = sa.plot_top_n_words(country_option, top_n_words=25)
        fig = plt.gcf()
        ax = plt.gca()
        st.pyplot(fig)
    except:
        st.write("No transcript available :'(")
        pass
