"""Shared helpers and constants of :mod:`vqa` (split from the original module)."""

from __future__ import annotations

import re
import string

_ARTICLES = {"a", "an", "the"}


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hedve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id": "I'd",
    "Idve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itdve": "it'd've",
    "itll": "it'll",
    "its": "it's",
    "lets": "let's",
    "maam": "ma'am",
    "mayve": "may've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "shant": "shan't",
    "shed": "she'd",
    "shedve": "she'd've",
    "shell": "she'll",
    "shes": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theyd": "they'd",
    "theydve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed": "we'd",
    "wedve": "we'd've",
    "well": "we'll",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whodve": "who'd've",
    "wholl": "who'll",
    "whomll": "whom'll",
    "whos": "who's",
    "whove": "who've",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "yall": "y'all",
    "youd": "you'd",
    "youdve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}


def _normalise_vqa(text: str) -> str:
    """Apply VQA v2 answer normalisation.

    Steps (matching official VQA eval script):
    1. Lower-case
    2. Remove period at end if not a number
    3. Replace commas in numbers with nothing (1,000 → 1000)
    4. Remove articles (a, an, the) at word boundaries
    5. Strip punctuation except apostrophes
    6. Strip leading/trailing whitespace
    """
    text = text.lower().strip()
    # Remove trailing period if not part of a number
    if text.endswith(".") and not re.match(r".*\d\.$", text):
        text = text[:-1]
    # Normalise number commas
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation (keep apostrophes in words)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_simple(text: str) -> str:
    """Light normalisation: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())
