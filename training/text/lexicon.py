"""Signals the text model looks for in a message.

A bag-of-words model trained on the available corpora learns their quirks:
the legitimate email is from 2002 mailing lists, so "your account" or a price
looks like a scam simply because old mailing lists never said it. These
signals are about *how* a message tries to make you act — pressure, threats,
requests for secrets, prizes — which carries over to messages the training
data never contained.

Each phrase is a sequence of tokens as produced by `tokenise` (lowercase
words and the placeholders `_url_`, `_num_`, `_cur_`, `_email_`). The lists
are exported with the model so the browser uses exactly the same ones.
"""

LEXICON: dict[str, tuple[str, ...]] = {
    # Pressure to act quickly.
    "urgency": (
        "urgent", "urgently", "immediately", "asap", "hurry", "expires", "expire", "expiring",
        "deadline", "final notice", "last chance", "final warning", "act now", "right away",
        "within _num_ hours", "within _num_ hour", "within _num_ days", "before midnight",
        "ends tonight", "ends today", "as soon as possible", "without delay", "now to",
    ),
    # Something bad has happened or will happen to your account.
    "threat": (
        "suspended", "suspend", "suspension", "locked", "blocked", "disabled", "deactivated",
        "deactivate", "terminated", "restricted", "unauthorized", "unauthorised",
        "unusual activity",
        "suspicious activity", "declined", "will be closed", "be closed", "will be charged",
        "avoid losing", "avoid suspension", "avoid interruption", "legal action", "almost full",
        "has expired", "on hold", "unpaid",
    ),
    # Asking you to hand over or re-enter a secret.
    "credential_request": (
        "verify your", "verify now", "confirm your", "validate your", "update your",
        "enter your", "submit your", "provide your", "reply with", "card details", "card number",
        "bank details", "account number", "your pin", "cvv",
        "billing information", "billing details", "login details",
        "security details", "keep the same password",
    ),
    # Telling you to follow a link or contact a number.
    "call_to_action": (
        "click here", "click the", "click below", "click on", "link below", "the link",
        "call now", "call _num_", "calling _num_", "text _num_", "txt _num_", "to _num_",
        "log in here", "sign in here", "tap here", "open the attachment", "claim now",
        "claim your", "to claim",
    ),
    # Too good to be true.
    "reward": (
        "won", "winner", "winnings", "prize", "reward", "congratulations", "congrats",
        "selected", "guaranteed", "lottery", "jackpot", "bonus", "free entry", "free prize",
        "cash award", "cash prize", "cash reward", "gift card", "eligible for",
        "tax refund", "weekly competition",
    ),
    # Addressed to nobody in particular.
    "generic_greeting": (
        "dear customer", "dear user", "dear client", "dear member", "dear account",
        "dear valued", "valued customer", "dear sir", "dear madam", "dear friend",
        "dear email", "dear beneficiary",
    ),
    # Written by a person about themselves: typical of real personal messages.
    "first_person": (
        # A bare "i" is absent on purpose: the tokeniser drops one-letter words.
        "i'm", "i'll", "i've", "i'd", "me", "my", "mine", "we", "we're", "we'll", "our", "us",
    ),
}

MAX_PHRASE_TOKENS = max(len(phrase.split()) for phrases in LEXICON.values() for phrase in phrases)
