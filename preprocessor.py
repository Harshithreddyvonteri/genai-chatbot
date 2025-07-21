# preprocessor.py
"""
Clean scraped GitLab pages: remove navigation/footer junk, filter short docs,
deduplicate, and produce a clean corpus ready for chunking & embedding.

Example usage:
    python preprocessor.py --in-dir data/raw_v2 --out data/processed/processed_v2.jsonl
    python preprocessor.py --in-dir data/raw_v3 --min-words 200 --report-csv stats_clean_v3.csv
"""

import argparse
import os
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

from bs4 import BeautifulSoup  # Already installed w/ scraper
# We will only re-parse HTML if available in raw JSON records.


# ---------------------------------------------------------
# Patterns of global nav / boilerplate we want to remove.
# (Case-insensitive contains() match.)
# You can extend this list as you inspect data.
# ---------------------------------------------------------
COMMON_NOISE_PATTERNS = [
    "the most comprehensive ai-powered devsecops platform",
    "platform explore our platform",
    "meet gitlab duo",
    "10 reasons why enterprises choose gitlab",
    "the gitlab handbook",
    "gitlab values",
    "contribute to this page",
    "edit this page",
    "you are here:",
    "on this page",
    "quick links",
    "common links",
    "contact us",
]


# Some pages include massive repeating link lists separated by pipes or spaces.
PIPE_LINK_RE = re.compile(r"\s*\|\s*")  # detect pipe-separated menus


def load_raw_records(in_dir: Path) -> List[Dict[str, Any]]:
    """Load all *.json files written by scraper."""
    recs = []
    for jf in sorted(in_dir.glob("*.json")):
        try:
            with jf.open("r", encoding="utf-8") as f:
                rec = json.load(f)
                rec["_json_path"] = jf  # keep pointer to source file
                recs.append(rec)
        except Exception as e:
            print(f"[WARN] Failed to read {jf}: {e}")
    return recs


def try_extract_main_from_html(html_path: Path) -> Optional[str]:
    """
    Attempt to re-parse raw HTML and extract the main article content.
    Returns cleaned visible text or None if any failure.
    """
    try:
        with html_path.open("r", encoding="utf-8") as f:
            html = f.read()
    except Exception:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Priority: <main>
    main_el = soup.find("main")
    if main_el:
        return extract_visible_text(main_el)

    # <article>
    art_el = soup.find("article")
    if art_el:
        return extract_visible_text(art_el)

    # id/class heuristics
    candidates = soup.select(
        "[id*='content'], [id*='markdown'], [class*='content'], [class*='markdown'], [class*='prose'], [role='main']"
    )
    if candidates:
        # choose the longest text candidate
        best = max(candidates, key=lambda el: len(el.get_text(strip=True)))
        return extract_visible_text(best)

    # fallback: whole doc
    return extract_visible_text(soup)


def extract_visible_text(node) -> str:
    """
    Minimal tag stripping; returns visible text from a soup node.
    """
    for t in node(["script", "style", "noscript", "svg"]):
        t.decompose()
    txt = node.get_text(separator="\n", strip=True)
    # collapse whitespace lines
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return "\n".join(lines)


def remove_noise_lines(lines: List[str]) -> List[str]:
    """
    Drop lines that look like navigation, boilerplate, or repeated marketing banners.
    """
    clean = []
    for ln in lines:
        low = ln.lower()

        # Drop if matches known noise patterns
        if any(pat in low for pat in COMMON_NOISE_PATTERNS):
            continue

        # Drop pipe-separated nav menu lines like: "Mission | Vision | Team | Contact"
        if PIPE_LINK_RE.search(ln) and len(ln) < 200:
            continue

        # Drop super-short all-caps "MENU" style items
        if len(ln.split()) <= 3 and ln.isupper():
            continue

        # Drop lines that are mostly links (heuristic: starts with http or contains >3 'http')
        if ln.count("http") > 3:
            continue

        clean.append(ln)
    return clean


def clean_text(raw_text: str) -> str:
    """
    Take raw multi-line text, strip junk, return cleaned body text.
    """
    if not raw_text:
        return ""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    lines = remove_noise_lines(lines)
    # Collapse duplicates if same line repeated
    deduped = []
    prev = None
    for ln in lines:
        if ln == prev:
            continue
        deduped.append(ln)
        prev = ln
    return "\n".join(deduped).strip()


def preprocess_records(
    recs: List[Dict[str, Any]],
    base_dir: Path,
    min_words: int = 200,
    dedupe: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    """
    Clean all records and return filtered list + raw/clean stats.

    Returns:
        cleaned_recs: list of cleaned records
        agg_raw: dict with raw totals
        agg_clean: dict with clean totals
    """
    cleaned_recs = []
    raw_word_total = 0
    clean_word_total = 0

    dropped_short = 0
    dropped_empty = 0
    dropped_dup = 0

    seen_hashes = {}

    for rec in recs:
        raw_text = rec.get("text", "") or ""
        raw_words = len(raw_text.split())
        raw_word_total += raw_words

        # Try to re-parse HTML for better main extraction if available
        html_rel = rec.get("html_file")
        cleaned_text = None
        if html_rel:
            html_path = (base_dir / html_rel).resolve()
            cleaned_text = try_extract_main_from_html(html_path)
        # fallback to raw
        if not cleaned_text:
            cleaned_text = raw_text

        cleaned_text = clean_text(cleaned_text)
        clean_words = len(cleaned_text.split())
        clean_word_total += clean_words

        if not cleaned_text:
            dropped_empty += 1
            continue

        if clean_words < min_words:
            dropped_short += 1
            continue

        hash_key = hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest()
        if dedupe and hash_key in seen_hashes:
            dropped_dup += 1
            continue
        seen_hashes[hash_key] = True

        cleaned_recs.append({
            "doc_id": hash_key[:16],
            "url": rec.get("url"),
            "title": rec.get("title"),
            "text": cleaned_text,
            "num_words": clean_words,
        })

    agg_raw = {
        "num_docs": len(recs),
        "total_words": raw_word_total,
    }
    agg_clean = {
        "num_docs": len(cleaned_recs),
        "total_words": clean_word_total,
        "dropped_short": dropped_short,
        "dropped_empty": dropped_empty,
        "dropped_dup": dropped_dup,
    }
    return cleaned_recs, agg_raw, agg_clean


def write_jsonl(out_path: Path, recs: List[Dict[str, Any]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")


def write_report_csv(csv_path: Path, recs: List[Dict[str, Any]]):
    import csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "url", "title", "num_words"])
        for r in recs:
            w.writerow([r["doc_id"], r.get("url", ""), r.get("title", ""), r.get("num_words", 0)])


def approx_tokens(words: int) -> int:
    # Rough English heuristic: ~0.75 words per token (inverse of ~1.33 tokens per word)
    # But to stay conservative, we’ll approximate char/4 using avg 5 chars/word → ~1.25 token/word.
    return int(words * 1.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Input folder from scraper (e.g., data/raw_v2).")
    ap.add_argument("--out", default=None, help="Output JSONL path (default auto in data/processed).")
    ap.add_argument("--min-words", type=int, default=200, help="Drop docs below this word count after cleaning.")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable duplicate removal.")
    ap.add_argument("--report-csv", default=None, help="Optional CSV summary path.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).resolve()
    if not in_dir.exists():
        print(f"[ERR] Input dir not found: {in_dir}")
        return

    # determine base_dir for html files (scraper wrote html + json in same folder)
    base_dir = in_dir

    if args.out is None:
        out_name = in_dir.name.replace("raw", "processed") + ".jsonl"
        out_path = Path("data/processed") / out_name
    else:
        out_path = Path(args.out).resolve()

    recs = load_raw_records(in_dir)
    cleaned_recs, agg_raw, agg_clean = preprocess_records(
        recs,
        base_dir=base_dir,
        min_words=args.min_words,
        dedupe=not args.no_dedupe,
    )

    write_jsonl(out_path, cleaned_recs)

    if args.report_csv:
        write_report_csv(Path(args.report_csv), cleaned_recs)

    # Reporting
    print("=" * 72)
    print(f"Preprocess Summary: {in_dir}")
    print("=" * 72)
    print(f"Input docs:          {agg_raw['num_docs']}")
    print(f"Output docs:         {agg_clean['num_docs']}")
    print(f"Dropped short:       {agg_clean['dropped_short']}")
    print(f"Dropped empty:       {agg_clean['dropped_empty']}")
    print(f"Dropped duplicates:  {agg_clean['dropped_dup']}")
    print(f"Raw words total:     {agg_raw['total_words']:,}")
    print(f"Clean words total:   {agg_clean['total_words']:,}")
    if agg_raw['total_words']:
        shrink = 100.0 * (1 - (agg_clean['total_words'] / agg_raw['total_words']))
        print(f"Shrink:              {shrink:.1f}%")
    raw_tok = approx_tokens(agg_raw['total_words'])
    clean_tok = approx_tokens(agg_clean['total_words'])
    print(f"Est tokens raw:      {raw_tok:,}")
    print(f"Est tokens clean:    {clean_tok:,}")
    print(f"Output corpus:       {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
