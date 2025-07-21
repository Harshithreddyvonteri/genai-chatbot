# scraper.py
"""
Simple site scraper for GitLab Handbook & Direction pages.
Usage:
    python scraper.py --out data/raw --max-pages 400 --max-depth 3
"""

import argparse
import os
import time
import json
import hashlib
from urllib.parse import urljoin, urldefrag, urlparse
import requests
from bs4 import BeautifulSoup

ALLOWED_DOMAINS = {
    "handbook.gitlab.com",
    "about.gitlab.com",
}

SEED_URLS = [
    "https://handbook.gitlab.com/",
    "https://about.gitlab.com/direction/",
]

HEADERS = {
    "User-Agent": "GenAI-Chatbot-Dev/0.1 (+learning project)",
}


def canonicalize_url(url: str) -> str:
    # Remove URL fragments (#section)
    url, _ = urldefrag(url)
    # Strip trailing slash except root
    if url.endswith("/") and len(url) > len("https://x/"):
        url = url[:-1]
    return url


def is_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        # Strip www.
        if host.startswith("www."):
            host = host[4:]
        return host in ALLOWED_DOMAINS
    except Exception:
        return False


def get_links(base_url: str, soup: BeautifulSoup) -> set:
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        full = canonicalize_url(full)
        if is_allowed(full):
            links.add(full)
    return links


def extract_text(soup: BeautifulSoup) -> tuple[str, list[str]]:
    # Remove unwanted elements
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    # Grab headings for metadata
    headings = []
    for h_tag in soup.find_all(["h1", "h2"]):
        t = h_tag.get_text(strip=True)
        if t:
            headings.append(t)

    # Get text
    text = soup.get_text(separator="\n", strip=True)
    # Normalize
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    clean = "\n".join(lines)
    return clean, headings


def fetch(url: str, timeout: int = 20) -> tuple[str, BeautifulSoup] | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            print(f"[WARN] {resp.status_code} {url}")
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        return resp.text, soup
    except Exception as e:
        print(f"[ERR] fetch failed {url}: {e}")
        return None


def save_page(out_dir: str, url: str, html: str, text: str, headings: list[str], title: str | None):
    # Hash for filename
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    html_path = os.path.join(out_dir, f"{h}.html")
    json_path = os.path.join(out_dir, f"{h}.json")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    rec = {
        "url": url,
        "title": title,
        "headings": headings,
        "text": text,
        "html_file": os.path.relpath(html_path, out_dir)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    return json_path


def crawl(out_dir: str, max_pages: int = 400, max_depth: int = 3, delay: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)
    queue = [(u, 0) for u in SEED_URLS]
    seen = set()
    saved = 0

    while queue and saved < max_pages:
        url, depth = queue.pop(0)
        url = canonicalize_url(url)
        if url in seen:
            continue
        seen.add(url)

        print(f"[{saved+1}] Fetching (depth={depth}): {url}")
        fetched = fetch(url)
        if not fetched:
            continue
        html, soup = fetched

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Text + metadata
        text, headings = extract_text(soup)
        if not text.strip():
            print(f"[SKIP] No text found: {url}")
            continue

        save_page(out_dir, url, html, text, headings, title)
        saved += 1

        # Enqueue child links
        if depth < max_depth:
            for link in get_links(url, soup):
                if link not in seen:
                    queue.append((link, depth + 1))

        time.sleep(delay)

    print(f"\nDone. Saved {saved} pages to {out_dir}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="Output directory for scraped pages.")
    ap.add_argument("--max-pages", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--delay", type=float, default=0.5, help="Seconds between requests.")
    args = ap.parse_args()

    crawl(
        out_dir=args.out,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
