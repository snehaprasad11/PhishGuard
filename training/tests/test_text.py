import math

import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import TensorProto, helper

from training.quantise import quantise_and_measure
from training.text.data import build_text_dataset, email_text, html_to_text, template_key
from training.text.features import (
    TEXT_FEATURE_NAMES,
    TextVectoriser,
    extract_text_features,
    featurise_text,
    lexicon_matches,
    terms,
    tokenise,
)
from training.text.lexicon import LEXICON
from training.text.pipeline import evaluate_text, fit_text_model, text_threshold

CFG = {
    "seed": 0,
    "threshold": {"target_fpr": 0.05, "confidence": 0.9},
    "text_model": {
        "version": "text-test", "C": 1.0,
        "split": {"val_years": [2024], "test_years": [2025], "fractions": [0.6, 0.2, 0.2]},
    },
}


def test_tokenise_words_and_placeholders():
    text = "URGENT: pay £2.99 NOW at http://evil.top/pay?id=1 or email Help@Bank.com, ref 4821. I'm ok"
    assert tokenise(text) == [
        "urgent", "pay", "_cur_", "_num_", "now", "at", "_url_", "or", "email", "_email_",
        "ref", "_num_", "i'm", "ok",
    ]


def test_tokenise_drops_single_letters_and_keeps_only_ascii_words():
    assert tokenise("a I x héllo wörld naïve") == ["llo", "rld", "na", "ve"]
    assert tokenise("") == []


def test_tokenise_clips_very_long_messages():
    assert len(tokenise("word " * 5000)) == 1000  # 5000 characters / 5 per "word "


def test_terms_are_tokens_then_adjacent_pairs():
    assert terms("Verify your account") == [
        "verify", "your", "account", "verify your", "your account",
    ]


def test_lexicon_matches_prefer_the_longest_phrase():
    tokens = tokenise("This is your final notice. Respond within 24 hours.")
    found = [(category, " ".join(tokens[i:i + n])) for category, i, n in lexicon_matches(tokens)]
    assert ("urgency", "final notice") in found
    assert ("urgency", "within _num_ hours") in found
    assert len([f for f in found if f[0] == "urgency"]) == 2


def test_every_lexicon_phrase_is_reachable_by_the_tokeniser():
    """A phrase the tokeniser can never produce would silently never match."""
    placeholders = {"_url_", "_email_", "_num_", "_cur_"}
    for category, phrases in LEXICON.items():
        for phrase in phrases:
            for token in phrase.split():
                # Either a placeholder the tokeniser emits, or a word it passes through unchanged.
                assert token in placeholders or tokenise(token) == [token], (category, phrase)
        assert len(set(phrases)) == len(phrases), category


def test_hand_checked_text_features():
    f = extract_text_features("URGENT! Dear customer, verify your account now: http://x.top Call 0800 123")
    # tokens: urgent dear customer verify your account now _url_ call _num_ _num_  (11)
    assert f["urgency"] == math.log1p(1)             # "urgent"
    assert f["generic_greeting"] == math.log1p(1)    # "dear customer"
    assert f["credential_request"] == math.log1p(1)  # "verify your"
    assert f["call_to_action"] == math.log1p(1)      # "call _num_"
    assert f["threat"] == f["reward"] == f["first_person"] == 0.0
    assert f["signal_kinds"] == 4.0
    assert f["number_count"] == math.log1p(2)
    assert f["token_count"] == math.log1p(11)
    assert f["exclamation_count"] == math.log1p(1)
    # letters: URGENT(6) Dear(4) customer(8) verify(6) your(4) account(7) now(3) http(4) x(1) top(3) Call(4)
    assert f["uppercase_ratio"] == pytest.approx(8 / 50)
    assert list(f) == list(TEXT_FEATURE_NAMES)


def test_a_personal_message_has_no_manipulation_signals():
    f = extract_text_features("Can you pick up some milk on your way home? We are out of bread.")
    assert f["signal_kinds"] == 0.0
    assert f["first_person"] == math.log1p(1)


def test_featurise_text_shape_and_empty_input():
    assert featurise_text(["hello there", ""]).shape == (2, len(TEXT_FEATURE_NAMES))
    assert featurise_text([]).shape == (0, len(TEXT_FEATURE_NAMES))
    assert np.isfinite(featurise_text(["", "!!!", "12345"])).all()


def test_tfidf_matches_a_hand_calculation():
    texts = ["win cash now", "win prize", "hello friend"]
    vec = TextVectoriser.fit(texts, max_terms=100, min_documents=1)
    assert vec.vocabulary == sorted(vec.vocabulary)
    row = vec.transform(["win win cash"]).toarray()[0]
    idf = {term: vec.idf[i] for i, term in enumerate(vec.vocabulary)}
    assert idf["win"] == pytest.approx(math.log(4 / 3) + 1)
    assert idf["cash"] == pytest.approx(math.log(4 / 2) + 1)
    # terms present: win x2, cash x1, "win win" (not in vocabulary), "win cash" x1
    weights = {"win": 2 * idf["win"], "cash": idf["cash"], "win cash": idf["win cash"]}
    norm = math.sqrt(sum(w * w for w in weights.values()))
    for term, weight in weights.items():
        assert row[vec.vocabulary.index(term)] == pytest.approx(weight / norm)
    assert np.linalg.norm(row) == pytest.approx(1.0)
    assert vec.transform(["zzz qqq"]).nnz == 0  # nothing known: all zeros, no division by zero


def test_tfidf_vocabulary_excludes_corpus_artefacts_and_rare_terms():
    texts = ["dear jose please verify", "jose verify now", "verify once"]
    vec = TextVectoriser.fit(texts, max_terms=100, min_documents=2)
    assert "verify" in vec.vocabulary
    assert not any("jose" in term for term in vec.vocabulary)
    assert "once" not in vec.vocabulary


def test_html_and_email_text_extraction():
    html = "<html><head><style>p{color:red}</style></head><body><p>Verify <b>now</b></p>" \
           "<script>alert(1)</script></body></html>"
    assert html_to_text(html).split() == ["Verify", "now"]
    raw = (b"Subject: Account notice\r\nContent-Type: text/html; charset=utf-8\r\n\r\n" + html.encode())
    assert email_text(raw) == "Account notice Verify now"
    plain = b"Subject: Hi\r\nContent-Type: text/plain\r\n\r\nSee you   at\r\n5pm"
    assert email_text(plain) == "Hi See you at 5pm"
    assert email_text(b"\xff\xfe garbage with no headers") != None  # noqa: E711 - must not raise


def test_template_key_groups_messages_that_differ_only_in_numbers():
    a = template_key("You won £1000! Call 09061234567 now")
    assert a == template_key("you won £500!  call 09069999999 NOW")
    assert a != template_key("You won a holiday! Call 09061234567 now")


YEAR_WORD = {2022: "alpha", 2023: "bravo", 2024: "charlie", 2025: "delta"}


def _tag(i: int) -> str:
    """A distinct made-up word per i: digits alone would not make two templates differ."""
    return "q" + chr(97 + i % 26) + chr(97 + i // 26)


def _messages() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append((f"hey are we still on for lunch my friend {_tag(i)}", 0, "sms", pd.NA))
        rows.append((f"urgent verify your account now or it will be suspended {_tag(i)}", 1, "sms", pd.NA))
    for year, word in YEAR_WORD.items():
        for i in range(25):
            rows.append((f"dear customer confirm your pin click here {word} {_tag(i)}", 1, "email", year))
            rows.append((f"here are my notes from our meeting {word} {_tag(i)}", 0, "email", pd.NA))
    rows.append(("same template 111", 0, "sms", pd.NA))
    rows.append(("same template 222", 0, "sms", pd.NA))       # duplicate template: dropped
    rows.append(("both labels 5", 0, "sms", pd.NA))
    rows.append(("both labels 6", 1, "sms", pd.NA))           # conflicting labels: both dropped
    rows.append(("   ", 0, "sms", pd.NA))                     # empty: dropped
    return pd.DataFrame(rows, columns=["text", "label", "kind", "year"])


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return build_text_dataset(_messages(), CFG)


def test_text_dataset_has_one_row_per_template_and_no_group_in_two_splits(dataset):
    assert dataset["group"].is_unique
    assert set(dataset["split"]) == {"train", "val", "test"}
    assert dataset["text"].str.startswith("same template").sum() == 1
    assert not dataset["text"].str.startswith("both labels").any()
    assert (dataset["text"].str.strip() != "").all()


def test_phishing_email_is_split_by_year(dataset):
    phishing_email = dataset[(dataset["kind"] == "email") & (dataset["label"] == 1)]
    word = phishing_email["text"].str.extract(r"click here (\w+)")[0]
    assert set(word[phishing_email["split"] == "test"]) == {"delta"}
    assert set(word[phishing_email["split"] == "val"]) == {"charlie"}
    assert set(word[phishing_email["split"] == "train"]) == {"alpha", "bravo"}


def test_text_dataset_is_deterministic():
    pd.testing.assert_frame_equal(build_text_dataset(_messages(), CFG), build_text_dataset(_messages(), CFG))


def test_text_model_trains_evaluates_and_separates_the_classes(dataset):
    model = fit_text_model(dataset, CFG)
    threshold = text_threshold(dataset, model, CFG)
    curated = {"development": [("legit", "see you at lunch my friend"),
                               ("scam", "urgent dear customer verify your account click here")]}
    report = evaluate_text(dataset, model, threshold, CFG, curated)
    assert report["metrics"]["pr_auc"] > 0.95
    assert report["curated"]["development"]["scam_caught"] == 1
    assert report["curated"]["development"]["legitimate_flagged"] == []
    assert set(report["coefficients"]) == set(TEXT_FEATURE_NAMES)
    assert report["rows"]["test"]["email_scam"] == 25


def test_quantisation_reports_not_applicable_for_a_linear_classifier(dataset):
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    model = fit_text_model(dataset, CFG)
    n = len(TEXT_FEATURE_NAMES)
    onnx_bytes = to_onnx(model, initial_types=[("features", FloatTensorType([None, n]))],
                         options={id(model[-1]): {"zipmap": False}}).SerializeToString()
    quantised, report = quantise_and_measure(onnx_bytes, np.zeros((2, n)), "features")
    assert quantised is None
    assert report["shipped"] == "float32" and "not applicable" in report["reason"]


def test_quantisation_shrinks_a_graph_it_can_convert():
    """A MatMul-based classifier with a large weight matrix: the case quantisation exists for."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(64, 512)).astype(np.float32)
    project = rng.normal(size=(512, 2)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "w"], ["hidden"]),
            helper.make_node("Relu", ["hidden"], ["act"]),
            helper.make_node("MatMul", ["act", "p"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["probabilities"]),
            helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0),
        ],
        "mlp",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, 64])],
        [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
         helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor("w", TensorProto.FLOAT, weights.shape, weights.flatten()),
         helper.make_tensor("p", TensorProto.FLOAT, project.shape, project.flatten())],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    quantised, report = quantise_and_measure(
        model.SerializeToString(), rng.normal(size=(50, 64)) * 0.05, "features"
    )
    assert quantised is not None
    assert report["size_saving"] > 0.5
    assert report["quantised_kb"] < report["original_kb"]
    assert "max_probability_shift" in report
