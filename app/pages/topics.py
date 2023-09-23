from pathlib import Path

from streamlit import components

APP_DIR = Path(__file__).absolute().parent.parent
DATA_DIR = APP_DIR.parent / "data"

with open(APP_DIR / "pages" / "lda.html", "r") as f:
    html_string = f.read()
components.v1.html(html_string, width=1300, height=800, scrolling=True)
