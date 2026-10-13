from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["recency", "frequency", "monetary", "unique_products", "avg_basket_units",
            "customer_age_days", "avg_gap_days", "return_value_ratio", "recent_order_share", "single_order"]
LOG_FEATURES = FEATURES[:7]

def extract_features(con: sqlite3.Connection, sql_path: Path, cutoff: str):
    cutoff = pd.Timestamp(cutoff).normalize()
    frame = pd.read_sql_query(sql_path.read_text(), con, params={
        "cutoff":str(cutoff), "recent_start":str(cutoff-pd.Timedelta(days=90))})
    if frame.empty:
        raise ValueError("No eligible purchases before cutoff.")
    if frame.customer_id.duplicated().any():
        raise AssertionError("Joins duplicated customer rows.")
    if not (pd.to_datetime(frame.last_purchase) < cutoff).all():
        raise AssertionError("Future leakage detected.")
    return frame

def add_personas(df):
    """Explicit marketing heuristics, distinct from learned cluster labels."""
    out = df.copy()
    high = out.monetary.quantile(.75)
    loyal = max(3., out.frequency.quantile(.75))
    out["rfm_persona"] = np.select([
        (out.recency >= 90) & (out.monetary >= high),
        out.recency >= 90,
        (out.recency <= 30) & (out.monetary >= high) & (out.frequency >= loyal),
        (out.recency <= 60) & (out.frequency >= loyal),
        (out.frequency == 1) & (out.customer_age_days <= 60),
        (out.recency <= 30) & (out.monetary >= high)],
        ["At-risk high-value", "Dormant / at-risk", "High-value loyal", "Loyal repeat",
         "New / first purchase", "Recent high-value"], default="Developing / occasional")
    thresholds = {"high_value_75th_percentile_gbp":float(high),
                  "loyal_frequency_threshold":float(loyal),"inactive_days":90,
                  "recent_days":30,"loyal_recency_days":60}
    return out, thresholds

def holdout(con, customers, cutoff, end):
    future = pd.read_sql_query("""SELECT customer_id,COUNT(DISTINCT invoice) AS future_orders,
        SUM(value) AS future_gross_value FROM lines WHERE reason='purchase'
        AND invoice_date >= ? AND invoice_date < ? GROUP BY customer_id""", con,
        params=[str(pd.Timestamp(cutoff)), str(pd.Timestamp(end))])
    result = customers[["customer_id","cluster","rfm_persona"]].merge(future,on="customer_id",how="left",validate="one_to_one")
    result[["future_orders","future_gross_value"]] = result[["future_orders","future_gross_value"]].fillna(0)
    result["purchased_in_holdout"] = (result.future_orders > 0).astype(int)
    return result

def wilson(successes, n):
    p, z = successes/n, 1.96
    den = 1+z*z/n
    mid = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def summarize_holdout(outcomes, by):
    s = outcomes.groupby(by).agg(customers=("customer_id","size"),buyers=("purchased_in_holdout","sum"),
        purchase_rate=("purchased_in_holdout","mean"),avg_future_gross_value=("future_gross_value","mean"),
        total_future_gross_value=("future_gross_value","sum"))
    s["purchase_rate_ci_low"], s["purchase_rate_ci_high"] = zip(*[wilson(r.buyers,r.customers) for _,r in s.iterrows()])
    return s.reset_index()

def cohort_retention(con, end):
    # Right-censor incomplete future months; also omit the incomplete final calendar month.
    full_month_end = str(pd.Timestamp(end).replace(day=1))
    activity = pd.read_sql_query("""SELECT DISTINCT customer_id,substr(invoice_date,1,7) AS month
        FROM lines WHERE reason='purchase' AND invoice_date < ?""",con,params=[full_month_end])
    activity["month"] = pd.PeriodIndex(activity.month,freq="M")
    first = activity.groupby("customer_id").month.min().rename("cohort")
    activity = activity.join(first,on="customer_id",validate="many_to_one")
    activity["age_month"] = [m.ordinal-c.ordinal for m,c in zip(activity.month,activity.cohort)]
    last = pd.Period(full_month_end,freq="M")-1
    table = activity.groupby(["cohort","age_month"]).customer_id.nunique().unstack()
    # Explicit zero-activity periods must exist even when no cohort repurchases.
    table = table.reindex(columns=range(last.ordinal-first.min().ordinal+1))
    table = table.div(table[0],axis=0)
    for c in table.index:
        for age in table.columns:
            if c+int(age) <= last and pd.isna(table.loc[c,age]):
                table.loc[c,age]=0.
    table.index = table.index.astype(str)
    return table
