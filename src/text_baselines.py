"""Registered text and length-only baselines."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .nested_cv import run_nested_cv


def _classifier(c, seed):
    return LogisticRegression(
        C=c,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
    )


def run_text_baseline(frame, column, config):
    cfg = config["text_baseline"]
    seed = config["project"]["seed"]
    candidates = [(float(c),) for c in cfg["C_grid"]]

    def build(candidate):
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=tuple(cfg["ngram_range"]),
                        min_df=cfg["min_df"],
                        max_features=cfg["max_features"],
                        sublinear_tf=cfg["sublinear_tf"],
                    ),
                ),
                ("classifier", _classifier(candidate[0], seed)),
            ]
        )

    return run_nested_cv(
        frame,
        candidates,
        build,
        lambda x: x[column].fillna("").tolist(),
        config["domains"],
    )


def add_length_features(frame):
    out = frame.copy()
    out["current_word_count"] = out.current_user_message.str.split().str.len()
    out["full_word_count"] = out.full_text_plain.str.split().str.len()
    required = {"current_token_count", "full_token_count"}
    if not required.issubset(out.columns):
        raise ValueError("Length control requires saved tokenizer token counts.")
    return out


def run_length_control(frame, config):
    cols = [
        "current_word_count",
        "full_word_count",
        "current_token_count",
        "full_token_count",
        "turn_index",
    ]
    frame = add_length_features(frame)
    seed = config["project"]["seed"]
    candidates = [(float(c),) for c in config["probe"]["C_grid"]]

    def build(candidate):
        return Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("classifier", _classifier(candidate[0], seed)),
            ]
        )

    return run_nested_cv(
        frame,
        candidates,
        build,
        lambda x: x[cols].to_numpy(dtype=float),
        config["domains"],
    )
