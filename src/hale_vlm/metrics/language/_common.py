"""Shared helpers and constants of :mod:`language` (split from the original module)."""

from __future__ import annotations

import string
from collections import Counter


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Count all n-grams in a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute the length of the Longest Common Subsequence between two token lists.

    Uses the classic O(|a|·|b|) DP algorithm.  For long texts consider limiting
    input length before calling.
    """
    m, n = len(a), len(b)
    # Use two rolling rows to keep memory at O(min(m,n))
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]
