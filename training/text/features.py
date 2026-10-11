"""Turn a message into numbers the text model can use.

Two featurisers live here:

* `featurise_text` — the one that ships. A handful of engineered signals:
  how often the message applies pressure, threatens, asks for secrets, and so
  on (see lexicon.py), plus a few surface measures.
* `TextVectoriser` — a TF-IDF bag of words, kept as the measured alternative.
  It scores higher on the test split and far worse on ordinary modern
  messages, which is why it does not ship. See the model card.

The shipped featuriser is reimplemented in TypeScript for the browser
(extension/src/lib/text-features.ts) and the two must agree exactly, so the
tokeniser is deliberately plain: ASCII-only rules, one regular expression that
means the same thing in Python and JavaScript, no stemming, no locale calls.
"""

import math
import re
from collections import Counter

import numpy as np
from scipy import sparse

from training.text.lexicon import LEXICON, MAX_PHRASE_TOKENS

MAX_CHARS = 5000

# Placeholders: what matters is that a message contains a link, an amount or an
# address — not which one. This also keeps personal data out of the vocabulary.
URL_TOKEN, EMAIL_TOKEN, NUMBER_TOKEN, CURRENCY_TOKEN = "_url_", "_email_", "_num_", "_cur_"

_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
# Order matters: earlier alternatives win. Only ASCII classes, so Python and JavaScript agree.
_TOKEN = re.compile(
    r"(?P<url>(?:https?://|www\.)[^ \t\r\n]+)"
    r"|(?P<email>[a-z0-9._%+\-]+@[a-z0-9.\-]+)"
    r"|(?P<num>[0-9]+(?:[.,][0-9]+)*)"
    r"|(?P<word>[a-z]+(?:'[a-z]+)?)"
    r"|(?P<cur>[$£€₹])"
)
_PLACEHOLDER = {"url": URL_TOKEN, "email": EMAIL_TOKEN, "num": NUMBER_TOKEN, "cur": CURRENCY_TOKEN}

# Words that are artefacts of where the training emails were collected (the
# corpus owner's name and domain), not signs of phishing. Never in the vocabulary.
CORPUS_ARTEFACTS = frozenset({"jose", "monkey", "nazario"})


def tokenise(text: str) -> list[str]:
    """Lowercased words (2+ letters) and placeholders, in order."""
    lowered = text[:MAX_CHARS].translate(_ASCII_LOWER)
    tokens = []
    for match in _TOKEN.finditer(lowered):
        kind = match.lastgroup
        if kind == "word":
            if len(match.group()) >= 2:
                tokens.append(match.group())
        else:
            tokens.append(_PLACEHOLDER[kind])
    return tokens


_PHRASES = {
    category: frozenset(tuple(phrase.split()) for phrase in phrases)
    for category, phrases in LEXICON.items()
}

# Categories that are ways of manipulating the reader (first_person is not one).
MANIPULATION = tuple(category for category in LEXICON if category != "first_person")

TEXT_FEATURE_NAMES = (
    *LEXICON,           # one count per lexicon category, in lexicon order
    "signal_kinds",     # how many different kinds of manipulation appear together
    "number_count",
    "currency_count",
    "token_count",
    "uppercase_ratio",
    "exclamation_count",
)


def lexicon_matches(tokens: list[str]) -> list[tuple[str, int, int]]:
    """Every lexicon phrase found: (category, index of first token, number of tokens).

    At each position the longest matching phrase of a category wins, so
    "final notice" counts once, not once for a shorter phrase inside it.
    """
    found = []
    for category, phrases in _PHRASES.items():
        for start in range(len(tokens)):
            for length in range(min(MAX_PHRASE_TOKENS, len(tokens) - start), 0, -1):
                if tuple(tokens[start:start + length]) in phrases:
                    found.append((category, start, length))
                    break
    return found


def extract_text_features(text: str) -> dict[str, float]:
    clipped = text[:MAX_CHARS]
    tokens = tokenise(clipped)
    counts = Counter(category for category, _, _ in lexicon_matches(tokens))
    letters = sum(1 for ch in clipped if "a" <= ch <= "z" or "A" <= ch <= "Z")
    capitals = sum(1 for ch in clipped if "A" <= ch <= "Z")
    raw = {
        **{category: counts[category] for category in LEXICON},
        # Links and addresses are deliberately not features: the legitimate training
        # email is mailing-list traffic full of footer links, so the model would learn
        # that a link is reassuring. Links are judged by the URL model instead.
        "signal_kinds": sum(1 for category in MANIPULATION if counts[category]),
        "number_count": tokens.count(NUMBER_TOKEN),
        "currency_count": tokens.count(CURRENCY_TOKEN),
        "token_count": len(tokens),
        "exclamation_count": clipped.count("!"),
    }
    # Counts are right-skewed (one email can repeat a phrase twenty times): log(1 + x).
    features = {
        name: float(value) if name == "signal_kinds" else math.log1p(value)
        for name, value in raw.items()
    }
    features["uppercase_ratio"] = capitals / letters if letters else 0.0
    return {name: float(features[name]) for name in TEXT_FEATURE_NAMES}


def featurise_text(texts: list[str]) -> np.ndarray:
    """Feature matrix of shape (len(texts), len(TEXT_FEATURE_NAMES))."""
    rows = [list(extract_text_features(text).values()) for text in texts]
    return np.array(rows, dtype=np.float64).reshape(len(texts), len(TEXT_FEATURE_NAMES))


def terms(text: str) -> list[str]:
    """Single tokens followed by adjacent pairs: `verify`, `your`, `verify your`."""
    tokens = tokenise(text)
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class TextVectoriser:
    """TF-IDF with a vocabulary chosen from training data.

    weight(term) = count(term) * idf(term), and each message's vector is scaled
    to unit length so long emails and short texts are comparable.
    idf(term) = ln((1 + n_docs) / (1 + docs_containing_term)) + 1.
    """

    def __init__(self, vocabulary: list[str], idf: list[float]):
        self.vocabulary = vocabulary
        self.idf = np.asarray(idf, dtype=np.float64)
        self._index = {term: i for i, term in enumerate(vocabulary)}

    @classmethod
    def fit(cls, texts: list[str], max_terms: int, min_documents: int) -> "TextVectoriser":
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(terms(text)))
        eligible = [
            (term, df) for term, df in document_frequency.items()
            if df >= min_documents and not (set(term.split()) & CORPUS_ARTEFACTS)
        ]
        # Most widespread terms first; ties broken alphabetically so the choice is stable.
        eligible.sort(key=lambda pair: (-pair[1], pair[0]))
        chosen = sorted(eligible[:max_terms])
        n = len(texts)
        return cls(
            [term for term, _ in chosen],
            [math.log((1 + n) / (1 + df)) + 1 for _, df in chosen],
        )

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        rows, columns, values = [], [], []
        for row, text in enumerate(texts):
            counts = Counter(self._index[t] for t in terms(text) if t in self._index)
            if not counts:
                continue
            weights = {column: count * self.idf[column] for column, count in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values()))
            for column in sorted(weights):
                rows.append(row)
                columns.append(column)
                values.append(weights[column] / norm)
        return sparse.csr_matrix(
            (values, (rows, columns)), shape=(len(texts), len(self.vocabulary)), dtype=np.float64
        )
