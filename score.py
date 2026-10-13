"""Assign an already engineered feature table using the saved historical model."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.modeling import predict

def main():
    p=argparse.ArgumentParser()
    p.add_argument("features",type=Path)
    p.add_argument("--model",type=Path,default=Path("models/segmentation.joblib"))
    p.add_argument("--output",type=Path,default=Path("outputs/scored_customers.csv"))
    args=p.parse_args()
    bundle=joblib.load(args.model)
    frame=pd.read_csv(args.features,dtype={"customer_id":str})
    missing=set(bundle["feature_names"])-set(frame.columns)
    if missing:raise ValueError(f"Missing features: {sorted(missing)}")
    frame["cluster"]=predict(bundle,frame)
    frame["assignment_method"]=bundle["assignment_note"]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output,index=False)
    print(f"Saved {len(frame):,} assignments to {args.output}. {bundle['assignment_note']}")

if __name__=="__main__":main()
