"""Leakage-safe nested leave-one-domain-out classification."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def run_nested_cv(frame, candidates, build_model, feature_for, domains):
    rows=[]; selections=[]
    eligible=frame[frame.turn_index.isin([3,4])].reset_index(drop=True)
    for test_domain in domains:
        train=eligible[eligible.domain != test_domain]; test=eligible[eligible.domain == test_domain]
        scored=[]
        for candidate in candidates:
            aucs=[]
            for valid_domain in [d for d in domains if d != test_domain]:
                fit=train[train.domain != valid_domain]; valid=train[train.domain == valid_domain]
                model=build_model(candidate)
                with warnings.catch_warnings(record=True):
                    model.fit(feature_for(fit), fit.binary_target.astype(int))
                aucs.append(roc_auc_score(valid.binary_target.astype(int), model.predict_proba(feature_for(valid))[:,1]))
            scored.append((float(np.mean(aucs)), candidate))
        best=max(scored, key=lambda item:(item[0], tuple(-float(x) for x in item[1])))
        model=build_model(best[1]); model.fit(feature_for(train), train.binary_target.astype(int))
        scores=model.predict_proba(feature_for(test))[:,1]
        for (_, row), score in zip(test.iterrows(), scores):
            rows.append({"example_id":row.example_id,"scenario_id":row.scenario_id,"domain":row.domain,"turn_index":int(row.turn_index),"binary_target":int(row.binary_target),"score":float(score),"test_domain":test_domain})
        selections.append({"test_domain":test_domain,"inner_mean_auc":best[0],"candidate":list(best[1])})
    return pd.DataFrame(rows), selections

def primary_metrics(predictions):
    t4=predictions[predictions.turn_index == 4]
    per=t4.groupby("domain").apply(lambda x: roc_auc_score(x.binary_target,x.score), include_groups=False)
    return {"macro_domain_auroc":float(per.mean()),"overall_auroc":float(roc_auc_score(t4.binary_target,t4.score)),"n":len(t4),"per_domain":per.to_dict()}
