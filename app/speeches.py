import json
import string
from pathlib import Path
from PIL import Image

import geopandas as gpd
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

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
stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def get_video_url(country: str) -> str:
    """Generate a url with the exact time when the leader start their speech.

    Args:
        country (str): Speaker's country.

    Returns:
        str: A youtube url.
    """
    if isinstance(
        df_speech_url[df_speech_url["country"] == country]["start"].values[0], str
    ):
        h, m, s = (
            df_speech_url[df_speech_url["country"] == country]["start"]
            .values[0]
            .split(":")
        )
        seconds = int(h) * 3600 + int(m) * 60 + int(s)
        url = f'{df_speech_url[df_speech_url["country"] == country]["url"].values[0]}&t={seconds}s'
        return url
    else:
        url = df_speech_url[df_speech_url["country"] == country]["url"].values[0]
        return url


def clean(doc: str) -> str:
    """Cleans the passed string. Apply lowercase, remove stopwords and punctuation and lemmatize the string.

    Args:
        doc (str): A string to be cleaned.

    Returns:
        str: A cleaned string.
    """
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def get_corpus_from_file(
    country: str, start: int = 0, end: int = 3600, path: Path = DATA_DIR / "2023"
) -> str:
    """Returns a string with the transcript of a video.

    Args:
        country (str): Country of origin.
        start (int, optional): Second when the speaker starts speaking. Defaults to 0.
        end (int, optional): Second when the speaking finish speaking. Defaults to 3600.
        path (Path, optional): Path where the transcripts are stored. Defaults to DATA_DIR/"2023".

    Returns:
        str: A string with the transcript of a video.
    """
    with open(path / f"{country}.json") as f:
        json_data = json.load(f)
    corpus = [x["text"] for x in json_data if x["start"] > start and x["start"] < end]
    large_corpus = " ".join([x for x in corpus])
    return large_corpus


def filter_words(corpus: str):
    words_filter = [
        "excellency",
        "assembly",
        "president",
        "mr",
        "weve",
        "also",
        "uh",
    ]
    filtered_corpus = " ".join(
        [x for x in corpus.split(" ") if x not in words_filter and len(x) > 1]
    )
    return filtered_corpus


def get_wordcloud(country_option: str):
    country_mask = np.array(Image.open(DATA_DIR / "masks" / f"{country_option}.jpg"))
    wc = WordCloud(
        background_color="white",
        max_words=number_of_words,
        mask=country_mask,
        stopwords=stopwords,
        contour_width=3,
        contour_color="steelblue",
    )

    if isinstance(
        df_speech_url[df_speech_url["country"] == country_option]["start"], str
    ) and isinstance(
        df_speech_url[df_speech_url["country"] == country_option]["end"], str
    ):
        h_start, m_start, s_start = (
            df_speech_url[df_speech_url["country"] == country_option]["start"]
            .values[0]
            .split(":")
        )
        start = int(h_start) * 60 * 60 + int(m_start) * 60 + int(s_start)
        h_end, m_end, s_end = (
            df_speech_url[df_speech_url["country"] == country_option]["end"]
            .values[0]
            .split(":")
        )
        end = int(h_end) * 60 * 60 + int(m_end) * 60 + int(s_end)
        corpus = get_corpus_from_file(country_option, start=start, end=end)
    elif isinstance(
        df_speech_url[df_speech_url["country"] == country_option]["start"], str
    ):
        h, m, s = (
            df_speech_url[df_speech_url["country"] == country_option]["start"]
            .values[0]
            .split(":")
        )
        start = int(h) * 60 * 60 + int(m) * 60 + int(s)
        corpus = get_corpus_from_file(country_option, start=start)
    else:
        corpus = get_corpus_from_file(country_option)

    corpus = get_corpus_from_file(country_option)
    corpus = clean(corpus)
    corpus = filter_words(corpus)
    wc.generate(corpus)
    wc.to_file(DATA_DIR / "wordclouds" / f"{country_option}_words.png")


def dispersion_plot(text, words, ignore_case=False, title="Lexical Dispersion Plot"):
    """
    Generate a lexical dispersion plot.

    :param text: The source text
    :type text: list(str) or iter(str)
    :param words: The target words
    :type words: list of str
    :param ignore_case: flag to set if case should be ignored when searching text
    :type ignore_case: bool
    :return: a matplotlib Axes object that may still be modified before plotting
    :rtype: Axes
    """
    # Coping nltk.draw.dispersion_plot directly because it has a bug that hasn't been merged into main
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "The plot function requires matplotlib to be installed. "
            "See https://matplotlib.org/"
        ) from e

    word2y = {
        word.casefold() if ignore_case else word: y
        for y, word in enumerate(reversed(words))
    }
    xs, ys = [], []
    for x, token in enumerate(text):
        token = token.casefold() if ignore_case else token
        y = word2y.get(token)
        if y is not None:
            xs.append(x)
            ys.append(y)

    words = words[::-1]  # this fix the order of words in the labels
    _, ax = plt.subplots()
    ax.plot(xs, ys, "|")
    ax.set_yticks(list(range(len(words))), words, color="C0")
    ax.set_ylim(-1, len(words))
    ax.set_title(title)
    ax.set_xlabel("Word Offset")
    return ax.figure


def plot_freq_words(country_option: str, top_n_words: int = 20):
    corpus = get_corpus_from_file(country_option)
    corpus = clean(corpus)
    corpus = filter_words(corpus)

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(corpus.lower())
    text = nltk.Text(tokens)

    words = [w.lower() for w in tokens]

    filtered_words = [
        word for word in words if len(word) > 1 and word not in ["na", "la", "uh"]
    ]
    fdist = nltk.FreqDist(filtered_words)
    st.pyplot(
        dispersion_plot(text, [str(w) for w, f in fdist.most_common(top_n_words)])
    )

    return


def plot_top_n_words(country_option: str, top_n_words: int = 20):
    corpus = get_corpus_from_file(country_option)
    corpus = clean(corpus)
    corpus = filter_words(corpus)

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(corpus.lower())

    words = [w.lower() for w in tokens]

    filtered_words = [
        word for word in words if len(word) > 1 and word not in ["na", "la", "uh"]
    ]
    fdist = nltk.FreqDist(filtered_words)

    df = pd.DataFrame(columns=("word", "freq"))
    i = 0
    for word, frequency in fdist.most_common(top_n_words):
        df.loc[i] = (word, frequency)
        i += 1

    title = f"Top {top_n_words} words in the speech"
    st.pyplot(df.plot.barh(x="word", y="freq", title=title).invert_yaxis())

    return


### App ###

st.set_page_config(
    page_title="UNGA78 Speech Analysis App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.set_option("deprecation.showPyplotGlobalUse", False)


df_speech_url = pd.read_csv(DATA_DIR / "UN Speeches.csv")
geo_data = gpd.read_file(DATA_DIR / "ne_110m_admin_0_countries.zip")

countries = df_speech_url["country"].unique()
country_option = st.sidebar.selectbox("", sorted(list(countries)))
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
        get_wordcloud(country_option)
        image = Image.open(DATA_DIR / "wordclouds" / f"{country_option}_words.png")
        st.image(image, caption=f"Wordcloud {country_option}", use_column_width=True)
    except:
        pass

with col2:
    st.video(get_video_url(country_option))

col3, col4 = st.columns(2)
with col3:
    try:
        plot_freq_words(country_option, top_n_words=25)
    except:
        st.write("No transcript available :'(")
        pass

with col4:
    try:
        plot_top_n_words(country_option, top_n_words=25)
    except:
        st.write("No transcript available :'(")
        pass
