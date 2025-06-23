import os
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from datasets import Dataset
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Start from the first available adopted texts page and follow "Volgende" links
START_TOC_URL = "https://www.europarl.europa.eu/doceo/document/TA-5-1999-07-21-TOC_NL.html"
HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HUGGINGFACE_USERNAME")
HF_DATASET_NAME = "Dutch-European-Parliament-Adopted-Texts"
HF_REPO_ID = f"{HF_USERNAME}/{HF_DATASET_NAME}"


def fix_term_number(url: str) -> str:
    """Correct the parliamentary term number in a TOC URL based on its year."""
    m = re.search(r"TA-(\d)-(\d{4})", url)
    if not m:
        return url
    term = int(m.group(1))
    year = int(m.group(2))
    if 1999 <= year <= 2004 and term != 5:
        return url.replace(f"TA-{term}-", "TA-5-")
    if 2004 <= year <= 2009 and term != 6:
        return url.replace(f"TA-{term}-", "TA-6-")
    if 2009 <= year <= 2014 and term != 7:
        return url.replace(f"TA-{term}-", "TA-7-")
    if 2014 <= year <= 2019 and term != 8:
        return url.replace(f"TA-{term}-", "TA-8-")
    if year >= 2019 and term != 9:
        return url.replace(f"TA-{term}-", "TA-9-")
    return url


def collect_text_urls(start_url: str):
    urls = []
    visited = set()
    current = start_url

    session = requests.Session()
    while current and current not in visited:
        visited.add(current)
        resp = session.get(current, timeout=20)
        if resp.status_code == 404:
            fixed = fix_term_number(current)
            if fixed != current:
                resp = session.get(fixed, timeout=20)
                if resp.status_code == 404:
                    break
                current = fixed
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        text_url = current.replace("-TOC", "")
        urls.append(text_url)
        next_link = soup.find("a", title="Volgende")
        if not next_link:
            next_link = soup.find("a", string=re.compile("Volgende", re.I))
        if not next_link or not next_link.get("href"):
            break
        current = urljoin(current, next_link["href"])
    return urls


def clean_text(text: str) -> str:
    """Apply common cleanup rules."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(The sitting (?:was suspended|opened|closed|ended) at.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Voting time ended at.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:debat|stemming|vraag|interventie)\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Het woord wordt gevoerd door:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\(|\[)\s*(?:(?:[a-zA-Z]{2,3})\s*(?:|\s|))?\s*(?:artikel|rule|punt|item)\s*\d+(?:,\s*lid\s*\d+)?\s*(?:\s+\w+)?\s*(\)|\])", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(COM|A)\d+-\d+(/\d+)?\]", "", text)
    text = re.sub(r"\(?(?:http|https):\/\/[^\s]+?\)", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(COD\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(INI\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(RSP\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(IMM\)\]", "", text)
    text = re.sub(r"\[\s*\d{4}/\d{4}\(NLE\)\]", "", text)
    text = re.sub(r"\[\s*\d{5}/\d{4}\s*-\s*C\d+-\d+/\d+\s*-\s*\d{4}/\d{4}\(NLE\)\]", "", text)
    text = re.sub(r"\(\u201cStemmingsuitslagen\u201d, punt \d+\)", "", text)
    text = re.sub(r"\(de Voorzitter(?: maakt na de toespraak van.*?| weigert in te gaan op.*?| stemt toe| herinnert eraan dat de gedragsregels moeten worden nageleefd| neemt er akte van|)\)", "", text)
    text = re.sub(r"\(zie bijlage.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*De vergadering wordt om.*?geschorst\.\)", "", text)
    text = re.sub(r"\(\s*De vergadering wordt om.*?hervat\.\)", "", text)
    text = re.sub(r"Volgens de \u201ccatch the eye\u201d-procedure wordt het woord gevoerd door.*?\.", "", text)
    text = re.sub(r"Het woord wordt gevoerd door .*?\.", "", text)
    text = re.sub(r"De vergadering wordt om \d{1,2}\.\d{2} uur gesloten.", "", text)
    text = re.sub(r"De vergadering wordt om \d{1,2}\.\d{2} uur geopend.", "", text)
    text = re.sub(r"Het debat wordt gesloten.", "", text)
    text = re.sub(r"Stemming:.*?\.", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_dutch_text_from_html(html_content: str) -> str | None:
    """Parse HTML adopted text and return cleaned Dutch text."""
    soup = BeautifulSoup(html_content, "lxml")
    paragraphs = [
        p.get_text(" ", strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
    ]
    final_text = clean_text("\n".join(paragraphs))
    if final_text and len(final_text) > 50:
        return final_text
    return None


def fetch_text(url: str, session: requests.Session) -> str | None:
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    # Some older pages incorrectly declare their charset as ISO-8859-1 while
    # the HTML is actually UTF-8 encoded. Force UTF-8 decoding to avoid
    # garbled characters appearing in the scraped text.
    resp.encoding = resp.apparent_encoding or "utf-8"
    return extract_dutch_text_from_html(resp.text)


def scrape() -> list:
    toc_urls = collect_text_urls(START_TOC_URL)
    data = []
    with requests.Session() as session:
        for url in tqdm(toc_urls, desc="Scraping adopted texts"):
            try:
                text = fetch_text(url, session)
                if text:
                    data.append({"URL": url, "text": text, "source": "European Parliament Adopted Texts"})
            except Exception as e:
                print(f"Failed to scrape {url}: {e}")
    return data


def push_dataset(records):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not provided")
        return
    login(token=token)
    ds = Dataset.from_list(records)
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    ds.push_to_hub(HF_REPO_ID, private=False)


def main():
    records = scrape()
    if records:
        push_dataset(records)
    else:
        print("No data scraped")


if __name__ == "__main__":
    main()
