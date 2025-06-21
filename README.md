# Dutch European Parliament Adopted Texts

This scraper collects Dutch-language adopted texts from the European Parliament website. Starting from the first available Table of Contents page at:

```
https://www.europarl.europa.eu/doceo/document/TA-5-1999-07-21-TOC_NL.html
```

The scraper follows the "Volgende" links to navigate chronologically. For each page the `-TOC` part of the URL is removed to obtain the actual adopted text, for example:

```
https://www.europarl.europa.eu/doceo/document/TA-5-1999-07-21_NL.html
```

Some older pages occasionally link to a non-existent URL with an incorrect parliamentary
term number (e.g. `TA-0-2002-11-29-TOC_NL.html`). The scraper automatically corrects the
term based on the year so that scraping can continue.

Scraped texts are uploaded to the Hugging Face dataset repository [`vGassen/Dutch-European-Parliament-Adopted-Texts`](https://huggingface.co/datasets/vGassen/Dutch-European-Parliament-Adopted-Texts).

## Usage

Run the scraper locally using Python 3:

```bash
pip install -r requirements.txt
python scraper.py
```

Set the environment variables `HF_USERNAME` and `HF_TOKEN` if you wish to automatically push the data to Hugging Face Hub.
