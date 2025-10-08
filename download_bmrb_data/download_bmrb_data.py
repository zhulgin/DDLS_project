
import os, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/"
OUTDIR = "bmrb_nmrstar"
os.makedirs(OUTDIR, exist_ok=True)

def get_links(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

for folder in get_links(BASE):
    if not folder.endswith("/"): continue
    sublinks = get_links(folder)
    for file in sublinks:
        if file.endswith(".str") or file.endswith(".nmrstar"):
            local_dir = os.path.join(OUTDIR, os.path.basename(folder.strip("/")))
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(file))
            if not os.path.exists(local_path):
                print("Downloading:", file)
                r = requests.get(file, timeout=60)
                with open(local_path, "wb") as f: f.write(r.content)

