#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import pandas as pd
from src.config import load_config
from src.nested_cv import primary_metrics
from src.text_baselines import run_length_control, run_text_baseline

config=load_config(ROOT/"configs/experiment.yaml"); frame=pd.read_parquet(ROOT/config["project"]["processed_path"]); out=ROOT/"results"; out.mkdir(exist_ok=True)
methods={"tfidf_full":run_text_baseline(frame,"full_text_plain",config),"tfidf_current":run_text_baseline(frame,"current_user_message",config),"length":run_length_control(frame,config)}
summary={}
for name,(pred,selection) in methods.items(): pred.assign(method=name).to_csv(out/f"{name}_predictions.csv",index=False); summary[name]={"metrics":primary_metrics(pred),"selections":selection}
(out/"baseline_metrics.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
