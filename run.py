#!/usr/bin/env python3
"""Run: python run.py --output outputs"""
from __future__ import annotations
import argparse
import json
import platform
import sqlite3
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from src.data import download,ingest
from src.features import extract_features,add_personas,holdout,summarize_holdout,cohort_retention
from src.modeling import compare_models
from src.reporting import write_reports

def main():
    parser=argparse.ArgumentParser(description="RetailLens | UCI Online Retail II customer segmentation")
    parser.add_argument("--input",type=Path,help="Existing original UCI .xlsx (otherwise download automatically)")
    parser.add_argument("--cutoff",help="Exclusive feature cutoff, YYYY-MM-DD; default last source date minus 90 days")
    parser.add_argument("--holdout-days",type=int,default=90)
    parser.add_argument("--output",type=Path,default=Path("outputs"))
    parser.add_argument("--rebuild",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.holdout_days < 1:
        parser.error("--holdout-days must be positive")
    started=time.time();root=Path(__file__).resolve().parent
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source=args.input.resolve() if args.input else download(root)
    print("[1/5] Data ingestion and audit",flush=True)
    audit=ingest(source,root/"data/processed/retail.sqlite",root/"data/processed/audit.json",args.rebuild)
    con=sqlite3.connect(root/"data/processed/retail.sqlite")
    # Source ends midday, so exclude the final calendar date from full-day validation.
    end=pd.Timestamp(audit["date_range"][1]).normalize()
    cutoff=pd.Timestamp(args.cutoff).normalize() if args.cutoff else end-pd.Timedelta(days=args.holdout_days)
    if not pd.Timestamp(audit["date_range"][0]) < cutoff < end:
        parser.error("Cutoff must lie strictly within observed data dates")
    holdout_end=min(end,cutoff+pd.Timedelta(days=args.holdout_days))
    print(f"[2/5] SQL features before {cutoff.date()}; holdout until {holdout_end.date()} exclusive",flush=True)
    df=extract_features(con,root/"sql/customer_features.sql",str(cutoff))
    df,thresholds=add_personas(df)
    total=con.execute("SELECT SUM(value) FROM lines WHERE reason='purchase' AND invoice_date < ?",[str(cutoff)]).fetchone()[0]
    if not np.isclose(df.monetary.sum(),total,atol=.01):
        raise AssertionError("Customer revenue does not reconcile to purchase ledger")
    print(f"[3/5] Compare models on {len(df):,} historical customers",flush=True)
    comparison,labels,bundle,metrics,scaled,scores=compare_models(df,args.seed)
    df["cluster"]=labels
    for i in range(scores.shape[1]):df[f"PC{i+1}"]=scores[:,i]
    print("[4/5] Profile segments and evaluate untouched future period",flush=True)
    outcomes=holdout(con,df,cutoff,holdout_end)
    profiles=df.groupby("cluster").agg(customers=("customer_id","size"),median_recency=("recency","median"),
        median_frequency=("frequency","median"),median_monetary=("monetary","median"),
        gross_value=("monetary","sum"),avg_return_value_ratio=("return_value_ratio","mean"),
        median_products=("unique_products","median")).reset_index()
    profiles["customer_share"]=profiles.customers/len(df)
    profiles["gross_value_share"]=profiles.gross_value/df.monetary.sum()
    retention=cohort_retention(con,holdout_end)
    monthly=pd.read_sql_query("""SELECT substr(invoice_date,1,7) AS month,SUM(value) AS gross_value,
       COUNT(DISTINCT invoice) AS orders,COUNT(DISTINCT customer_id) AS buyers
       FROM lines WHERE reason='purchase' AND invoice_date < ? GROUP BY 1 ORDER BY 1""",con,params=[str(end)])
    monthly["partial_month"]=monthly.month.eq(end.strftime("%Y-%m"))
    split_counts=dict(con.execute("""SELECT CASE WHEN invoice_date < ? THEN 'feature_rows'
        WHEN invoice_date < ? THEN 'holdout_rows' ELSE 'outside_analysis_rows' END,COUNT(*)
        FROM lines WHERE reason='purchase' GROUP BY 1""",[str(cutoff),str(holdout_end)]))
    metrics.update(dict(cutoff_exclusive=str(cutoff.date()),holdout_end_exclusive=str(holdout_end.date()),
       holdout_days=int((holdout_end-cutoff).days),feature_customers=len(df),feature_purchase_lines=int(df.purchase_lines.sum()),
       feature_orders=int(df.frequency.sum()),feature_gross_value=float(df.monetary.sum()),
       persona_thresholds=thresholds,purchase_row_split=split_counts,
       overall_holdout_purchase_rate=float(outcomes.purchased_in_holdout.mean()),
       raw_rows=audit["raw_rows"],runtime_seconds=round(time.time()-started,1),
       versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__),
       status="Executed on the complete original UCI workbook; no simulated dataset used"))
    artifacts={"customer_segments":df,"model_comparison":comparison,"cluster_profiles":profiles,
       "holdout_customer_outcomes":outcomes,"holdout_by_cluster":summarize_holdout(outcomes,"cluster"),
       "holdout_by_persona":summarize_holdout(outcomes,"rfm_persona"),"monthly_performance":monthly,
       "cluster_persona_counts":pd.crosstab(df.cluster,df.rfm_persona).reset_index()}
    for name,frame in artifacts.items():frame.to_csv(out/f"{name}.csv",index=False,float_format="%.6f")
    retention.to_csv(out/"cohort_retention.csv")
    pd.DataFrame(bundle["pca"].components_,columns=bundle["feature_names"],
                 index=[f"PC{i+1}" for i in range(bundle["pca"].n_components_)]).to_csv(out/"pca_loadings.csv")
    for name,data in [("metrics",metrics),("data_audit",audit)]:
        (out/f"{name}.json").write_text(json.dumps(data,indent=2,allow_nan=False))
    bundle.update(cutoff=metrics["cutoff_exclusive"],persona_thresholds=thresholds,versions=metrics["versions"])
    (root/"models").mkdir(exist_ok=True)
    joblib.dump(bundle,root/"models/segmentation.joblib")
    print("[5/5] Generate charts, executive report and evidence-based resume bullets",flush=True)
    write_reports(out,df,comparison,profiles,outcomes,retention,metrics,audit,scaled)
    con.close()
    print(json.dumps({k:metrics[k] for k in ["selected_candidate","silhouette","pca_components","stability_mean_ari","feature_customers","runtime_seconds"]},indent=2),flush=True)
    print(f"Complete: {out}",flush=True)

if __name__=="__main__":main()
