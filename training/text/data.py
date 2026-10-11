"""Load the message datasets and build leakage-safe splits for the text model.

Three sources, two kinds of message:

* SMS — the UCI SMS Spam Collection (spam vs legitimate texts).
* Phishing email — Jose Nazario's phishing corpus, one mailbox per year (CC-BY-4.0).
* Legitimate email — the SpamAssassin public corpus ("easy ham").

Leakage guards:

* Near-identical messages (the same template with a different amount or
  reference number) form one group, and only one message per group is kept,
  so a template cannot appear in two splits.
* Phishing email is split by the year of its mailbox: the test split is the
  newest year. The other sources carry no usable dates and are split by a
  stable hash of the message.
"""

import email
import email.policy
import hashlib
import mailbox
import re
import tarfile
import zipfile
from html.parser import HTMLParser
from typing import Any

import pandas as pd

from training.data.loaders import BENIGN, PHISHING, DataError, _pinned_file
from training.data.split import _hash_fraction
from training.text.features import MAX_CHARS

SPLITS = ("train", "val", "test")


class _TextExtractor(HTMLParser):
    """Visible text of an HTML email body: tags dropped, scripts and styles skipped."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style", "head"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
    except Exception:  # noqa: BLE001 - malformed markup in hostile email is expected
        return ""
    return " ".join(extractor.parts)


def email_text(raw: bytes) -> str:
    """Subject and readable body of an email, as a person would see it."""
    message = email.message_from_bytes(raw, policy=email.policy.default)
    plain, html = [], []
    for part in message.walk():
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        kind = part.get_content_type()
        if kind not in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except Exception:  # noqa: BLE001 - unknown charsets and broken encodings are common
            payload = part.get_payload(decode=True) or b""
            content = payload.decode("utf-8", "replace")
        (plain if kind == "text/plain" else html).append(str(content))
    body = "\n".join(plain) if any(p.strip() for p in plain) else html_to_text("\n".join(html))
    try:
        subject = str(message["subject"] or "")
    except Exception:  # noqa: BLE001 - undecodable headers
        subject = ""
    return re.sub(r"\s+", " ", f"{subject}\n{body}").strip()[:MAX_CHARS]


def load_sms(cfg: dict[str, Any]) -> pd.DataFrame:
    path, entry = _pinned_file(cfg, "sms_spam")
    with zipfile.ZipFile(path) as z, z.open("SMSSpamCollection") as f:
        lines = f.read().decode("utf-8").splitlines()
    if len(lines) != entry["rows"]:
        raise DataError(f"sms_spam: {len(lines)} rows on disk, manifest says {entry['rows']}")
    rows = [line.split("\t", 1) for line in lines]
    unexpected = {label for label, _ in rows} - {"ham", "spam"}
    if unexpected:
        raise DataError(f"sms_spam: unexpected labels {sorted(unexpected)}")
    return pd.DataFrame({
        "text": [text for _, text in rows],
        "label": [PHISHING if label == "spam" else BENIGN for label, _ in rows],
        "kind": "sms",
        "year": pd.NA,
    })


def load_phishing_email(cfg: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for year in cfg["data"]["nazario"]["years"]:
        path, entry = _pinned_file(cfg, f"nazario_{year}")
        box = mailbox.mbox(str(path), create=False)
        texts = [email_text(box.get_bytes(key)) for key in box.keys()]
        box.close()
        if len(texts) != entry["rows"]:
            raise DataError(f"nazario_{year}: {len(texts)} messages, manifest says {entry['rows']}")
        frames.append(
            pd.DataFrame({"text": texts, "label": PHISHING, "kind": "email", "year": year})
        )
    return pd.concat(frames, ignore_index=True)


def load_legitimate_email(cfg: dict[str, Any]) -> pd.DataFrame:
    texts = []
    for archive in cfg["data"]["spamassassin"]["archives"]:
        path, entry = _pinned_file(cfg, f"spamassassin_{archive}")
        count = 0
        with tarfile.open(path, "r:bz2") as tar:
            for member in tar:
                if member.isfile() and "cmds" not in member.name:
                    texts.append(email_text(tar.extractfile(member).read()))
                    count += 1
        if count != entry["rows"]:
            raise DataError(
                f"spamassassin_{archive}: {count} messages, manifest says {entry['rows']}"
            )
    return pd.DataFrame({"text": texts, "label": BENIGN, "kind": "email", "year": pd.NA})


def template_key(text: str) -> str:
    """Identifier shared by messages that differ only in digits, case or spacing."""
    template = re.sub(r"\s+", " ", re.sub(r"[0-9]+", "0", text.lower())).strip()
    return hashlib.sha256(template.encode()).hexdigest()[:16]


def build_text_dataset(messages: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return columns text, label, kind, group, split with one row per message template."""
    split_cfg = cfg["text_model"]["split"]
    frame = messages[messages["text"].str.strip().str.len() > 0].copy()
    frame["group"] = frame["text"].map(template_key)

    # A template seen with both labels has no usable ground truth.
    labels_per_group = frame.groupby("group")["label"].nunique()
    frame = frame[~frame["group"].isin(labels_per_group[labels_per_group > 1].index)]
    # Keep the earliest copy of each template, so the newest split holds only new wording.
    frame["_order"] = frame["year"].fillna(0).astype(int)
    frame = frame.sort_values(["_order", "text"], kind="stable").drop_duplicates("group")

    train_frac, val_frac, _ = split_cfg["fractions"]

    def assign(row: pd.Series) -> str:
        if not pd.isna(row["year"]):
            if row["year"] in split_cfg["test_years"]:
                return "test"
            return "val" if row["year"] in split_cfg["val_years"] else "train"
        fraction = _hash_fraction(row["group"], cfg["seed"])
        if fraction < train_frac:
            return "train"
        return "val" if fraction < train_frac + val_frac else "test"

    frame["split"] = frame.apply(assign, axis=1)
    frame = frame.sort_values("group").reset_index(drop=True)
    if frame.groupby("group")["split"].nunique().max() > 1:
        raise DataError("a message template appears in more than one split")
    return frame[["text", "label", "kind", "group", "split"]]


def load_messages(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.concat(
        [load_sms(cfg), load_phishing_email(cfg), load_legitimate_email(cfg)], ignore_index=True
    )
