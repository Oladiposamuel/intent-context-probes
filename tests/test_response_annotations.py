import json

import pandas as pd
import pytest

from src.response_annotations import (
    ANNOTATION_FIELDS,
    finalize_annotations,
    prepare_annotation_package,
)


def inputs(tmp_path):
    prefix_rows = []
    response_paths = {}
    specs = {}
    for model in ["m1", "m2"]:
        specs[model] = {"revision": f"{model}-rev"}
        records = []
        for domain in ["a", "b", "c", "d"]:
            for index in range(8):
                scenario = f"{domain}-{index}"
                for branch, target in [("benign", 0), ("suspicious", 1)]:
                    example = f"{scenario}-{branch}-4"
                    if model == "m1":
                        prefix_rows.append(
                            {
                                "example_id": example,
                                "scenario_id": scenario,
                                "domain": domain,
                                "branch": branch,
                                "binary_target": target,
                                "is_primary_t4": True,
                                "full_text_plain": f"Conversation {example}",
                            }
                        )
                    records.append(
                        {
                            "response_id": f"{model}__{example}",
                            "example_id": example,
                            "model_revision": f"{model}-rev",
                            "dataset_hash": "hash",
                            "raw_response": f"Response {model} {example}",
                        }
                    )
        path = tmp_path / f"{model}.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        response_paths[model] = path
    return pd.DataFrame(prefix_rows), response_paths, specs


def complete(path, *, suspicious_level=1):
    frame = pd.read_csv(path, keep_default_na=False)
    for field in ANNOTATION_FIELDS:
        if field == "assistance_level":
            frame[field] = [
                suspicious_level if "suspicious" in text else 3
                for text in frame.conversation
            ]
        elif field == "caution_or_refusal":
            frame[field] = "none"
        elif field == "references_prior_context":
            frame[field] = "yes"
        elif field == "contains_actionable_enabling_detail":
            frame[field] = "no"
        elif field == "annotation_confidence":
            frame[field] = 3
    frame.to_csv(path, index=False)


def test_prepare_is_blinded_and_creates_fixed_repeat_sample(tmp_path):
    prefixes, responses, specs = inputs(tmp_path)
    output = tmp_path / "annotations"
    manifest = prepare_annotation_package(
        prefixes,
        responses,
        specs,
        dataset_hash="hash",
        output_dir=output,
        seed=42,
    )
    primary = pd.read_csv(output / "blinded_primary.csv")
    repeat = pd.read_csv(output / "blinded_repeat.csv")
    assert len(primary) == 128
    assert len(repeat) == 26
    assert manifest["repeat_fraction"] == 26 / 128
    assert not {"model_alias", "branch", "domain"}.intersection(primary.columns)


def test_finalize_locks_before_unblinding_and_reports_agreement(tmp_path):
    prefixes, responses, specs = inputs(tmp_path)
    output = tmp_path / "annotations"
    prepare_annotation_package(
        prefixes,
        responses,
        specs,
        dataset_hash="hash",
        output_dir=output,
        seed=42,
    )
    primary = output / "blinded_primary.csv"
    repeat = output / "blinded_repeat.csv"
    complete(primary)
    complete(repeat)
    summary = finalize_annotations(
        primary,
        repeat,
        output / "private_identity_key.csv",
        output / "private_repeat_key.csv",
        output_dir=output,
        iterations=20,
        seed=42,
    )
    assert (output / "annotation_lock.json").exists()
    assert (output / "unblinded_annotations.csv").exists()
    assert summary["agreement"]["repeat_count"] == 26
    assert set(summary["behaviour"]["model_branch_groups"]) == {
        "m1__benign", "m1__suspicious", "m2__benign", "m2__suspicious"
    }


def test_finalize_rejects_changed_response_content(tmp_path):
    prefixes, responses, specs = inputs(tmp_path)
    output = tmp_path / "annotations"
    prepare_annotation_package(
        prefixes,
        responses,
        specs,
        dataset_hash="hash",
        output_dir=output,
        seed=42,
    )
    primary = output / "blinded_primary.csv"
    repeat = output / "blinded_repeat.csv"
    complete(primary)
    complete(repeat)
    frame = pd.read_csv(primary, keep_default_na=False)
    frame.loc[0, "response"] = "changed"
    frame.to_csv(primary, index=False)
    with pytest.raises(ValueError, match="content changed"):
        finalize_annotations(
            primary,
            repeat,
            output / "private_identity_key.csv",
            output / "private_repeat_key.csv",
            output_dir=output,
            iterations=20,
        )
