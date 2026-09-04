from __future__ import annotations
import sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.nested_cv import primary_metrics, run_nested_cv

class NestedCVTests(unittest.TestCase):
 def test_every_row_is_out_of_domain(self):
  rows=[]
  for d in ["a","b","c","d"]:
   for turn in [3,4]:
    for y in [0,1]:
     for i in range(2): rows.append({"example_id":f"{d}{turn}{y}{i}","scenario_id":f"{d}{i}","domain":d,"turn_index":turn,"binary_target":y,"value":y+i*.01})
  frame=pd.DataFrame(rows)
  pred,_=run_nested_cv(frame,[(1.0,)],lambda c:LogisticRegression(C=c[0]),lambda x:x[["value"]].to_numpy(),["a","b","c","d"])
  self.assertEqual(len(pred),len(frame)); self.assertTrue((pred.domain==pred.test_domain).all()); self.assertEqual(primary_metrics(pred)["n"],16)
