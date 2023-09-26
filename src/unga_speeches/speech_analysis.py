import json
import ssl
import string
from pathlib import Path
from PIL import Image

import geopandas as gpd
import nltk
import numpy as np
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from wordcloud import WordCloud, STOPWORDS

from unga_speeches.config import DATA_DIR, URL_SPEECHES

# To download data from nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("stopwords")
nltk.download("wordnet")
# end download data from nltk

stop = set(stopwords.words("english"))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stopwords = set(STOPWORDS)


def get_data(
    url: str = URL_SPEECHES,
    save: bool = True,
    path: Path = DATA_DIR / "UN Speeches.csv",
) -> pd.DataFrame:
    """Return a dataframe with columns ['url', 'country', 'start', 'end'].

    The function download the spreadsheet, store it locally and then returns the stored copy of the spreadsheet.

    Args:
        url (str): URL of the google spreadsheet with the data.
        save (bool, optional): _description_. Defaults to True.
        path (Path, optional): _description_. Defaults to DATA_DIR/"UN Speeches.csv".

    Returns:
        pd.DataFrame: Dataframe with data of the speeches.
    """

    def save_spreadsheet(path: Path):
        with open(path, "wb") as f:
            f.write(response.content)

    response = requests.get(url)

    if save:
        save_spreadsheet(path)

    df_speech_url = pd.read_csv(path)
    return df_speech_url


def get_transcript(video_id: str) -> dict:
    """Collect the transcript from YouTube.

    Args:
        video_id (str): `id` of the YouTube video.

    Returns:
        dict: The video's transcript.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(e)
        return None, None


def save_json(data: dict, country: str, output_path: Path = DATA_DIR / "2023"):
    """Save the transcript as JSON file.

    Args:
        data (dict): `dict` with the transcript.
        country (str): Country of the speech.
        output_path (Path, optional): . Defaults to DATA_DIR/"2023".
    """
    with open(output_path / f"{country}.json", "w") as outfile:
        json.dump(data, outfile)


def download_speech_transcriptions(
    df_speeches: pd.DataFrame, overwrite: bool = False, path: Path = DATA_DIR / "2023"
):
    """Downloads the transcript of all the urls in the dataframe.

    Args:
        df_speech_url (pd.DataFrame): Dataframe with urls to speeches.
        overwrite (bool, optional): Download again if the file already exists. Defaults to False.
    """
    pbar = tqdm(df_speeches.iterrows(), total=len(df_speeches))
    for _, r in pbar:
        pbar.set_description(r["country"])
        if (path / f"{r['country']}.json").is_file() and not overwrite:
            continue
        transcript = get_transcript(r["url"].split("?v=")[-1])
        if transcript:
            save_json(transcript, r["country"])


def generate_masks(df_speeches: pd.DataFrame, masks_path: Path = DATA_DIR / "masks"):
    """Generate masks of the countries with transcripts.

    Countries shape https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip

    Args:
        df_speeches (pd.DataFrame): Dataframe with countries with speeches.
        output_path (Path, optional): Path where to save the .jpg with the country mask. Defaults to DATA_DIR/"masks".
    """
    natural_data_admin_countries = gpd.read_file(
        DATA_DIR / "ne_110m_admin_0_countries.zip"
    )
    wab = gpd.read_file(DATA_DIR / "world-administrative-boundaries.zip")

    # Generate masks
    pbar = tqdm(df_speeches.iterrows(), total=len(df_speeches))
    not_found = []
    for i, r in pbar:
        pbar.set_description(r["country"])
        if (
            len(
                natural_data_admin_countries[
                    natural_data_admin_countries["SOVEREIGNT"] == r["country"]
                ]
            )
            > 0
        ):
            ax = natural_data_admin_countries[
                natural_data_admin_countries["SOVEREIGNT"] == r["country"]
            ].plot()
            ax.axis("off")
            ax.figure.savefig(masks_path / f"{r['country']}.jpg")
        elif len(wab[wab["name"] == r["country"]]) > 0:
            ax = wab[wab["name"] == r["country"]].plot()
            ax.axis("off")
            ax.figure.savefig(masks_path / f"{r['country']}.jpg")
        else:
            not_found.append(r["country"])

    for country_not_found in not_found:
        print(f"{country_not_found} not found.")


def generate_mask(
    country_option: str,
    masks_path: Path = DATA_DIR / "masks",
):
    natural_data_admin_countries = gpd.read_file(
        DATA_DIR / "ne_110m_admin_0_countries.zip"
    )
    wab = gpd.read_file(DATA_DIR / "world-administrative-boundaries.zip")

    if (
        len(
            natural_data_admin_countries[
                natural_data_admin_countries["SOVEREIGNT"] == country_option
            ]
        )
        > 0
    ):
        ax = natural_data_admin_countries[
            natural_data_admin_countries["SOVEREIGNT"] == country_option
        ].plot()
        ax.axis("off")
        ax.figure.savefig(masks_path / f"{country_option}.jpg")
    elif len(wab[wab["name"] == country_option]) > 0:
        ax = wab[wab["name"] == country_option].plot()
        ax.axis("off")
        ax.figure.savefig(masks_path / f"{country_option}.jpg")
    else:
        pass


def get_mask(
    country_option: str,
    masks_path: Path = DATA_DIR / "masks",
):
    generate_mask(country_option)
    country_mask = np.array(Image.open(masks_path / f"{country_option}.jpg"))
    return country_mask


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


def filter_words(corpus: str):
    words_filter = [
        "excellency",
        "assembly",
        "president",
        "mr",
        "weve",
        "also",
        "uh",
        "000",
    ]
    filtered_corpus = " ".join(
        [x for x in corpus.split(" ") if x not in words_filter and len(x) > 1]
    )
    return filtered_corpus


def get_wordcloud(
    df_speeches: pd.DataFrame,
    country_option: str,
    number_of_words: int = 42,
    masks_path: Path = DATA_DIR / "masks",
):
    country_mask = get_mask(country_option=country_option, masks_path=masks_path)
    wc = WordCloud(
        background_color="white",
        max_words=number_of_words,
        mask=country_mask,
        stopwords=stopwords,
        contour_width=3,
        contour_color="steelblue",
    )

    if isinstance(
        df_speeches[df_speeches["country"] == country_option]["start"], str
    ) and isinstance(df_speeches[df_speeches["country"] == country_option]["end"], str):
        h_start, m_start, s_start = (
            df_speeches[df_speeches["country"] == country_option]["start"]
            .values[0]
            .split(":")
        )
        start = int(h_start) * 60 * 60 + int(m_start) * 60 + int(s_start)
        h_end, m_end, s_end = (
            df_speeches[df_speeches["country"] == country_option]["end"]
            .values[0]
            .split(":")
        )
        end = int(h_end) * 60 * 60 + int(m_end) * 60 + int(s_end)
        corpus = get_corpus_from_file(country_option, start=start, end=end)
    elif isinstance(
        df_speeches[df_speeches["country"] == country_option]["start"], str
    ):
        h, m, s = (
            df_speeches[df_speeches["country"] == country_option]["start"]
            .values[0]
            .split(":")
        )
        start = int(h) * 60 * 60 + int(m) * 60 + int(s)
        corpus = get_corpus_from_file(country_option, start=start)
    else:
        corpus = get_corpus_from_file(country_option)

    corpus = clean(corpus)
    corpus = filter_words(corpus)
    wc.generate(corpus)
    wc.to_file(DATA_DIR / "wordclouds" / f"{country_option}_words.png")


def get_video_url(df_speeches: pd.DataFrame, country: str) -> str:
    """Generate a url with the exact time when the leader start their speech.

    Args:
        country (str): Speaker's country.

    Returns:
        str: A youtube url.
    """
    if isinstance(
        df_speeches[df_speeches["country"] == country]["start"].values[0], str
    ):
        h, m, s = (
            df_speeches[df_speeches["country"] == country]["start"].values[0].split(":")
        )
        seconds = int(h) * 3600 + int(m) * 60 + int(s)
        url = f'{df_speeches[df_speeches["country"] == country]["url"].values[0]}&t={seconds}s'
        return url
    else:
        url = df_speeches[df_speeches["country"] == country]["url"].values[0]
        return url


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
    disp_plot = dispersion_plot(
        text, [str(w) for w, f in fdist.most_common(top_n_words)]
    )
    return disp_plot


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
    return df.plot.barh(x="word", y="freq", title=title).invert_yaxis()
