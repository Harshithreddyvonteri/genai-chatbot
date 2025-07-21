"""
Quick dataset quality + size inspection for scraped GitLab pages.

Usage:
    python analyze_dataset.py path to data raw_v1
    python analyze_dataset.py data/raw_v2 --samples 5
    python analyze_dataset.py data/raw_v3 --csv stats_v3.csv

What it does:
- Counts JSON records in the folder
- Aggregates total characters, words, and rough token estimate
- Reports avg/min/max text length
- Counts duplicate texts (hash match)
- Tallies most common H1/H2 headings
- Shows sample records (URL, title, first ~200 chars)

No external dependencies beyond the Python stdlib.
"""

import argparse
import json
import os
import statistics
import hashlib
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any


def approx_tokens_from_chars(char_count: int) -> int:
    """
    Very rough heuristic: ~4 chars per token (OpenAI-ish English average).
    """
    return max(1, char_count // 4)


def load_records(folder: Path) -> List[Dict[str, Any]]:
    """
    Load all *.json files in folder into memory (list of dicts).
    """
    recs = []
    for jf in folder.glob("*.json"):
        try:
            with jf.open("r", encoding="utf-8") as f:
                rec = json.load(f)
                recs.append(rec)
        except Exception as e:
            print(f"[WARN] Failed to read {jf}: {e}")
    return recs


def analyze_records(recs: List[Dict[str, Any]]) -> dict:
    """
    Compute dataset stats.
    """
    if not recs:
        return {
            "num_records": 0,
            "total_chars": 0,
            "total_words": 0,
            "approx_tokens": 0,
            "avg_words": 0,
            "min_words": 0,
            "max_words": 0,
            "num_duplicates": 0,
            "top_headings": [],
        }

    word_counts = []
    char_counts = []
    headings_counter = Counter()
    text_hashes = Counter()

    for r in recs:
        txt = r.get("text", "") or ""
        # normalize whitespace a touch
        norm = " ".join(txt.split())
        words = norm.split()
        word_counts.append(len(words))
        char_counts.append(len(norm))
        # headings
        hs = r.get("headings") or []
        headings_counter.update(hs)
        # duplicates (hash)
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        text_hashes[h] += 1

    total_words = sum(word_counts)
    total_chars = sum(char_counts)
    approx_tokens = approx_tokens_from_chars(total_chars)

    dup_count = sum(c for c in text_hashes.values() if c > 1)

    stats = {
        "num_records": len(recs),
        "total_chars": total_chars,
        "total_words": total_words,
        "approx_tokens": approx_tokens,
        "avg_words": round(statistics.mean(word_counts), 2),
        "median_words": statistics.median(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "num_duplicates": dup_count,
        "top_headings": headings_counter.most_common(15),
    }
    return stats


def print_stats(folder: Path, stats: dict):
    print("=" * 72)
    print(f"DATASET: {folder}")
    print("=" * 72)
    print(f"Records:         {stats['num_records']}")
    print(f"Total words:     {stats['total_words']:,}")
    print(f"Total chars:     {stats['total_chars']:,}")
    print(f"Approx tokens:   {stats['approx_tokens']:,}")
    print(f"Avg words/doc:   {stats['avg_words']}")
    print(f"Median words:    {stats['median_words']}")
    print(f"Min words/doc:   {stats['min_words']}")
    print(f"Max words/doc:   {stats['max_words']}")
    print(f"Duplicate texts: {stats['num_duplicates']}")
    print("\nTop headings (H1/H2 frequency):")
    for h, c in stats["top_headings"]:
        print(f"  {c:4d}  {h}")
    print()


def show_samples(recs: List[Dict[str, Any]], n: int = 5):
    """
    Print a few random-ish samples (take first n sorted by URL for reproducibility).
    """
    if not recs:
        print("No records to sample.\n")
        return
    # sort by URL for deterministic display
    sample = sorted(recs, key=lambda r: r.get("url", ""))[:n]
    print("-" * 72)
    print(f"SAMPLES (showing {len(sample)} of {len(recs)} records)")
    print("-" * 72)
    for r in sample:
        url = r.get("url", "")
        title = r.get("title", "")
        txt = (r.get("text") or "").strip().replace("\n", " ")
        snippet = txt[:200] + ("..." if len(txt) > 200 else "")
        print(f"URL   : {url}")
        print(f"Title : {title}")
        print(f"Text  : {snippet}")
        print("-" * 72)
    print()


def write_csv(folder: Path, recs: List[Dict[str, Any]], csv_path: Path):
    """
    Optional: write a flat CSV with url,title,words,chars.
    Useful if you want to inspect in Excel.
    """
    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "title", "words", "chars"])
        for r in recs:
            txt = r.get("text", "") or ""
            norm = " ".join(txt.split())
            words = len(norm.split())
            chars = len(norm)
            w.writerow([r.get("url", ""), (r.get("title") or "").strip(), words, chars])
    print(f"[INFO] CSV written: {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="Path to dataset folder (with .json files).")
    ap.add_argument("--samples", type=int, default=5, help="Number of sample records to show.")
    ap.add_argument("--csv", type=str, default=None, help="Optional path to write summary CSV.")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists():
        print(f"[ERR] folder not found: {folder}")
        return

    recs = load_records(folder)
    stats = analyze_records(recs)
    print_stats(folder, stats)
    show_samples(recs, args.samples)

    if args.csv:
        write_csv(folder, recs, Path(args.csv).expanduser().resolve())


if __name__ == "__main__":
    main()
