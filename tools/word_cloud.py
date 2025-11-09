#!/usr/bin/env python3
"""
word_cloud.py – word-frequency map and word cloud visualization with NLTK stop-words removed.
"""

import argparse
import collections
import json
import string
import sys
from pathlib import Path

import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure the data files are present (run once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def load_stopwords(lang: str) -> set[str]:
    """Return a lower-cased set of stop-words for *lang*."""
    return {w.lower() for w in stopwords.words(lang)}


def tokenise(text: str) -> list[str]:
    """Simple tokeniser that respects Unicode letters."""
    return word_tokenize(text)


def count_words(
    sources: list[Path],
    *,
    stop_set: set[str] | None = None,
    case_sensitive: bool = False,
) -> collections.Counter:
    counter = collections.Counter()
    for src in sources:
        f = sys.stdin if src == Path("-") else src.open(encoding="utf-8")
        with f:
            for line in f:
                if not case_sensitive:
                    line = line.lower()
                for word in tokenise(line):
                    # Remove punctuation from word
                    word = word.strip(string.punctuation)
                    if not word:
                        continue
                    if stop_set and word.lower() in stop_set:
                        continue
                    if word.isdigit():
                        continue
                    # Skip if word is only punctuation
                    if all(c in string.punctuation for c in word):
                        continue
                    counter[word] += 1
    return counter


def generate_wordcloud(
    freq: collections.Counter,
    output_path: Path | None = None,
    width: int = 1600,
    height: int = 800,
) -> None:
    """Generate and display/save a word cloud from frequency data."""
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        relative_scaling=0.5,
        min_font_size=10,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Word cloud saved to: {output_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a word-frequency map and word cloud visualization, dropping NLTK stop-words."
    )
    parser.add_argument("files", nargs="*", default=["-"], help="File(s) or '-' for STDIN.")
    parser.add_argument("-l", "--lang", default="english", help="Stop-word language (NLTK).")
    parser.add_argument("-c", "--case-sensitive", action="store_true", help="Keep original case.")
    parser.add_argument("-j", "--json", action="store_true", help="Emit JSON instead of plain text.")
    parser.add_argument("-x", "--extra-stop", help="File with extra stop-words (one per line).")
    parser.add_argument("-w", "--wordcloud", help="Generate word cloud and save to this path (e.g., wordcloud.png).")
    parser.add_argument("--width", type=int, default=1600, help="Word cloud width in pixels (default: 1600).")
    parser.add_argument("--height", type=int, default=800, help="Word cloud height in pixels (default: 800).")
    args = parser.parse_args()

    stop_set = load_stopwords(args.lang)

    if args.extra_stop:
        extra_stop_set = {
            ln.strip().lower()
            for ln in Path(args.extra_stop).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }
        stop_set |= extra_stop_set

    paths = [Path(p) if p != "-" else Path("-") for p in args.files]
    freq = count_words(paths, stop_set=stop_set, case_sensitive=args.case_sensitive)

    if args.wordcloud:
        generate_wordcloud(freq, Path(args.wordcloud), width=args.width, height=args.height)
    elif args.json:
        json.dump(dict(freq.most_common()), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for word, cnt in freq.most_common():
            print(f"{cnt}\t{word}")


if __name__ == "__main__":
    main()

