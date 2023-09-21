import json
import string
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

# To download data from nltk
import nltk
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
stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def clean(doc: str) -> str:
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_corpus_from_file(country: str, path: Path = DATA_DIR / "2023") -> str:
    with open(path / f"{country}.json") as f:
        json_data = json.load(f)
    corpus = [x["text"] for x in json_data]
    large_corpus = " ".join([x for x in corpus])
    return large_corpus


def get_wordcloud(country_option: str):
    country_mask = np.array(Image.open(DATA_DIR / "masks" / f"{country_option}.jpg"))
    wc = WordCloud(
        background_color="white",
        # max_words=2000,
        max_words=number_of_words,
        mask=country_mask,
        stopwords=stopwords,
        width=600,
        height=400,
        contour_width=3,
        contour_color="steelblue",
    )
    corpus = get_corpus_from_file(country_option)
    corpus = clean(corpus)
    wc.generate(corpus)
    wc.to_file(DATA_DIR / "wordclouds" / f"{country_option}_words.png")


st.set_page_config(
    page_title="UNGA78 Speech Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)

df_speech_url = pd.read_csv(DATA_DIR / "UN Speeches.csv")

countries = df_speech_url["country"].unique()
country_option = st.sidebar.selectbox("", sorted(list(countries)))


col1, col2 = st.columns(2)
with col1:
    number_of_words = st.slider("How namy words?", 10, 200, 40)
    get_wordcloud(country_option)

    image = Image.open(DATA_DIR / "wordclouds" / f"{country_option}_words.png")
    st.image(image, caption=f"Wordcloud {country_option}")


with col2:
    st.video(df_speech_url[df_speech_url["country"] == country_option]["url"].values[0])
