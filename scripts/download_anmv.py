"""
Scrape and download full RCP (Résumé des Caractéristiques du Produit) from ANMV.
Source: www.ircp.anmv.anses.fr

The index pages list drugs with fiche.aspx links (metadata only).
The actual clinical content (posology, dosage, contraindications) is at rcp.aspx.
"""

import httpx
import time
import re
from pathlib import Path
from html.parser import HTMLParser
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR    = Path(__file__).parent.parent / "data" / "raw_pdfs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL      = "https://www.ircp.anmv.anses.fr"
INDEX_URL     = BASE_URL + "/index.aspx?letter={letter}"
TARGET_SPECIES = {"Porcins", "Volailles"}
LETTERS        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_PER_RUN    = 300


class IndexParser(HTMLParser):
    """Extract drug names from ANMV index pages."""

    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self._in_td   = False
        self._td_idx  = 0
        self._current: dict = {}
        self._in_tr   = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._in_tr  = True
            self._td_idx = 0
            self._current = {}
        if tag == "td" and self._in_tr:
            self._in_td = True
        if tag == "a" and self._in_td and self._td_idx == 2:
            href = attrs.get("href", "")
            if "NomMedicament=" in href:
                drug_name = href.split("NomMedicament=")[-1]
                self._current["name"] = drug_name
                # always use rcp.aspx for full clinical content
                self._current["rcp_url"] = f"{BASE_URL}/rcp.aspx?NomMedicament={drug_name}"

    def handle_endtag(self, tag):
        if tag == "td" and self._in_tr:
            self._in_td = False
            self._td_idx += 1
        if tag == "tr" and self._in_tr:
            self._in_tr = False
            if self._current.get("name") and self._current.get("species"):
                self.rows.append(self._current.copy())

    def handle_data(self, data):
        data = data.strip()
        if not data or not self._in_td:
            return
        if self._td_idx == 9:
            self._current["species"] = data


def matches_target(species_text: str) -> bool:
    return any(s in species_text for s in TARGET_SPECIES)


def fetch_index_page(letter: str) -> list[dict]:
    url = INDEX_URL.format(letter=letter)
    try:
        r = httpx.get(url, timeout=30, follow_redirects=True)
        r.raise_for_status()
        parser = IndexParser()
        parser.feed(r.text)
        return [row for row in parser.rows if matches_target(row.get("species", ""))]
    except Exception as e:
        print(f"[ERROR] index {letter}: {e}")
        return []


def download_rcp(name: str, rcp_url: str) -> bool:
    safe_name = re.sub(r'[^\w\-,]', '_', name)[:80]
    dest = OUTPUT_DIR / f"{safe_name}.html"

    if dest.exists():
        print(f"[SKIP] {safe_name}")
        return False

    try:
        r = httpx.get(rcp_url, timeout=30, follow_redirects=True)
        r.raise_for_status()
        # sanity check — real RCP pages are > 10KB
        if len(r.content) < 5000:
            print(f"[WARN] {safe_name} — too small ({len(r.content)}B), skipping")
            return False
        dest.write_text(r.text, encoding="utf-8")
        print(f"[OK]   {safe_name} ({len(r.content)//1024}KB)")
        return True
    except Exception as e:
        print(f"[ERROR] {safe_name}: {e}")
        return False


def main():
    downloaded = 0
    for letter in LETTERS:
        if downloaded >= MAX_PER_RUN:
            break
        print(f"\n── Letter {letter}")
        matches = fetch_index_page(letter)
        print(f"   {len(matches)} notices matching target species")
        for row in matches:
            if downloaded >= MAX_PER_RUN:
                break
            if download_rcp(row["name"], row["rcp_url"]):
                downloaded += 1
            time.sleep(0.4)

    print(f"\nDone. {downloaded} RCP files saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
