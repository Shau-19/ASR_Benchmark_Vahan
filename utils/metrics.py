import re
from difflib import SequenceMatcher


# ── Text normalisation ────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Word Error Rate (WER) ─────────────────────────────────────

def wer(reference: str, hypothesis: str) -> float:
    """
    Standard WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = ref word count.
    Returns a float in [0, ∞).  0.0 = perfect.
    """
    ref  = normalise(reference).split()
    hyp  = normalise(hypothesis).split()

    # Dynamic programming edit distance
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                    dp[i][j-1],    # insertion
                                    dp[i-1][j-1])  # substitution

    return dp[r][h] / max(r, 1)


# ── Character Error Rate (CER) ────────────────────────────────

def cer(reference: str, hypothesis: str) -> float:
    """
    Same as WER but at character level.
    Useful for catching near-miss transcriptions of locality names.
    """
    ref = normalise(reference)
    hyp = normalise(hypothesis)

    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[r][h] / max(r, 1)


# ── Entity Match Rate (EMR) ───────────────────────────────────

def entity_match(locality: str, hypothesis: str, threshold: float = 0.75) -> dict:
    """
    Checks if the locality name was correctly captured in the hypothesis.

    Three levels:
      - exact   : normalised locality string appears verbatim in hypothesis
      - fuzzy   : SequenceMatcher ratio >= threshold (catches 'Marathhalli' etc.)
      - missed  : neither

    Returns a dict with keys: exact (bool), fuzzy (bool), ratio (float)
    """
    loc_norm = normalise(locality)
    hyp_norm = normalise(hypothesis)

    exact = loc_norm in hyp_norm

    # fuzzy match against every sliding window of same word-length in hypothesis
    loc_words = loc_norm.split()
    hyp_words = hyp_norm.split()
    n = len(loc_words)

    best_ratio = 0.0
    for i in range(len(hyp_words) - n + 1):
        window = " ".join(hyp_words[i:i+n])
        ratio  = SequenceMatcher(None, loc_norm, window).ratio()
        best_ratio = max(best_ratio, ratio)

    fuzzy = best_ratio >= threshold

    return {
        "exact" : exact,
        "fuzzy" : fuzzy,       # True even if not exact
        "ratio" : round(best_ratio, 3),
    }


# ── Aggregate helper ──────────────────────────────────────────

def aggregate_metrics(results: list[dict]) -> dict:
    """
    Given a list of per-sample result dicts, compute aggregate stats.
    Each dict must have: wer, cer, entity_exact, entity_fuzzy
    """
    n = len(results)
    if n == 0:
        return {}

    return {
        "avg_wer"         : round(sum(r["wer"] for r in results) / n, 3),
        "avg_cer"         : round(sum(r["cer"] for r in results) / n, 3),
        "entity_exact_pct": round(sum(r["entity_exact"] for r in results) / n * 100, 1),
        "entity_fuzzy_pct": round(sum(r["entity_fuzzy"] for r in results) / n * 100, 1),
        "n_samples"       : n,
    }
