"""Generate reproducible figures and reports from measured outputs only."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from .features import FEATURES,summarize_holdout

COLORS=["#147D92","#E19A38","#7358A6","#D26059","#419D78","#4C75A3","#BE678E","#8D8B49"]
NAVY="#182C42"

def md_table(df):
    cols=list(df.columns)
    def fmt(v):
        if isinstance(v,(float,np.floating)):
            return "—" if pd.isna(v) else f"{v:,.3f}"
        return str(v)
    return "| "+" | ".join(cols)+" |\n|"+"|".join(["---"]*len(cols))+"|\n"+"\n".join("| "+" | ".join(fmt(v) for v in row)+" |" for row in df.itertuples(index=False,name=None))

def save(fig,path):
    fig.savefig(path,dpi=170,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close(fig)

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.titleweight":"bold","axes.titlecolor":NAVY,
        "axes.labelcolor":NAVY,"axes.edgecolor":"#D4DCE1","figure.facecolor":"#FAFBFC",
        "axes.facecolor":"#FAFBFC","grid.color":"#E5EAF0","axes.grid":False})

def write_reports(out,df,comparison,profiles,outcomes,retention,m,audit,scaled):
    style();plots=out/"figures";plots.mkdir(exist_ok=True)
    # Main page: numeric evidence, not a fitted marketing narrative.
    fig=plt.figure(figsize=(14,10),layout="constrained")
    grid=fig.add_gridspec(3,2,height_ratios=[.32,1,1])
    head=fig.add_subplot(grid[0,:]);head.axis("off")
    head.text(0,1,"RETAILLENS  /  CUSTOMER INTELLIGENCE",fontsize=22,color=NAVY,weight="bold",va="top")
    head.text(0,.47,f"{audit['raw_rows']:,} source records   •   {len(df):,} historical customers   •   {m['k']} clusters   •   {m['selected_candidate']}",fontsize=12,color=NAVY)
    head.text(0,.12,f"Features before {m['cutoff_exclusive']}  |  {m['holdout_days']}-day future observation  |  PCA retains {m['pca_explained_variance']:.1%} variance",color="#607083")
    ax=fig.add_subplot(grid[1,0])
    for c in sorted(df.cluster.unique()):
        d=df[df.cluster==c];ax.scatter(d.PC1,d.PC2,s=7,alpha=.35,color=COLORS[c],label=f"Cluster {c}",rasterized=True)
    ax.set(title="Customer landscape in two principal components",xlabel=f"PC1 ({m['pca_variance_ratios'][0]:.1%})",ylabel=f"PC2 ({m['pca_variance_ratios'][1]:.1%})")
    ax.legend(markerscale=2,frameon=False,fontsize=8)
    ax=fig.add_subplot(grid[1,1]);x=np.arange(len(profiles))
    ax.bar(x-.19,profiles.customer_share,width=.36,label="Customer share",color=COLORS[0])
    ax.bar(x+.19,profiles.gross_value_share,width=.36,label="Historical gross value share",color=COLORS[1])
    ax.set_xticks(x,[f"Cluster {c}" for c in profiles.cluster]);ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.0%}"))
    ax.set_title("Where customer value is concentrated");ax.legend(frameon=False,fontsize=8)
    ax=fig.add_subplot(grid[2,0]);h=summarize_holdout(outcomes,"cluster")
    ax.bar(h.cluster.astype(str),h.purchase_rate,color=[COLORS[c] for c in h.cluster],alpha=.9)
    ax.errorbar(np.arange(len(h)),h.purchase_rate,yerr=np.vstack([h.purchase_rate-h.purchase_rate_ci_low,h.purchase_rate_ci_high-h.purchase_rate]),fmt="none",ecolor=NAVY,capsize=4)
    ax.set(title=f"Purchase rate in the following {m['holdout_days']} days",xlabel="Cluster",ylabel="Customers with ≥1 future purchase",ylim=(0,1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.0%}"))
    ax.text(.01,.95,"Error bars: 95% Wilson intervals",transform=ax.transAxes,va="top",fontsize=8)
    ax=fig.add_subplot(grid[2,1])
    cols=["recency","frequency","monetary"]
    z=pd.DataFrame(scaled,columns=FEATURES).assign(cluster=df.cluster.to_numpy()).groupby("cluster")[cols].mean()
    im=ax.imshow(z.to_numpy(),aspect="auto",cmap="BrBG",vmin=-2,vmax=2)
    ax.set_xticks(range(3),["Recency\nHigher = less recent","Frequency","Monetary"])
    ax.set_yticks(range(len(z)),[f"Cluster {c}" for c in z.index]);ax.set_title("RFM profile: mean transformed z-score")
    for r in range(len(z)):
        for c in range(3):ax.text(c,r,f"{z.iloc[r,c]:.2f}",ha="center",va="center",color=NAVY)
    fig.colorbar(im,ax=ax,fraction=.04)
    fig.supxlabel("Source: UCI Online Retail II • Gross merchandise purchases after cleaning • Descriptive segments; no causal uplift claim",fontsize=9,color="#607083")
    save(fig,plots/"executive_summary.png")

    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout="constrained")
    for i,(name,g) in enumerate(comparison.groupby("algorithm")):
        g=g.sort_values("k");axs[0].plot(g.k,g.silhouette,"o-",label=name,color=COLORS[i])
    axs[0].set(title="Separation on identical PCA features",xlabel="Number of clusters",ylabel="Sampled silhouette");axs[0].legend(frameon=False,fontsize=8)
    g=comparison[comparison.algorithm=="KMeans"].sort_values("k")
    axs[1].plot(g.k,g.inertia,"o-",color=COLORS[0]);axs[1].set(title="K-Means elbow diagnostic",xlabel="Number of clusters",ylabel="Within-cluster squared distances")
    g=comparison[comparison.algorithm=="GMM"].sort_values("k")
    axs[2].plot(g.k,g.bic,"o-",color=COLORS[2]);axs[2].set(title="GMM information criterion",xlabel="Number of components",ylabel="BIC (lower is better within GMM)")
    fig.supxlabel("Selection: highest eligible silhouette; k ≥ 3; every group ≥ 2%; largest group ≤ 80%; converged fits only",fontsize=9)
    save(fig,plots/"model_selection.png")

    fig,ax=plt.subplots(figsize=(12,8),layout="constrained")
    view=retention.iloc[:,:13]
    masked=np.ma.masked_invalid(view.to_numpy())
    cmap=plt.get_cmap("YlGnBu").copy();cmap.set_bad("#E9EDF1")
    im=ax.imshow(masked,aspect="auto",cmap=cmap,vmin=0,vmax=.6)
    ax.set_xticks(range(len(view.columns)),view.columns);ax.set_yticks(range(len(view)),view.index)
    ax.set(title="Monthly cohort retention | first 12 months after first observed purchase",xlabel="Months since first observed purchase",ylabel="First-observed purchase cohort")
    for r in range(len(view)):
        for c in range(len(view.columns)):
            v=view.iloc[r,c]
            if pd.notna(v):ax.text(c,r,f"{v:.0%}",ha="center",va="center",fontsize=7,color="white" if v>.35 else NAVY)
    fig.colorbar(im,ax=ax,fraction=.03,format=FuncFormatter(lambda x,p:f"{x:.0%}"))
    fig.supxlabel("Grey cells are unobserved future periods. Final incomplete month omitted. Month 0 = 100%. Existing customers may predate this dataset.",fontsize=9)
    save(fig,plots/"cohort_retention.png")

    fig,axs=plt.subplots(1,3,figsize=(14,4.6),layout="constrained")
    for ax,col,title in zip(axs,["recency","frequency","monetary"],["Days since last purchase","Distinct purchase invoices","Historical gross spend (£)"]):
        ax.hist(df[col],bins=np.geomspace(max(1,df[col].min()),df[col].max()+1,45) if col!="recency" else 45,color=COLORS[0],alpha=.85)
        if col!="recency":ax.set_xscale("log")
        ax.set(title=title,xlabel=col,ylabel="Customers")
    save(fig,plots/"rfm_distributions.png")

    fig,ax=plt.subplots(figsize=(10,8),layout="constrained")
    corr=pd.DataFrame(scaled,columns=FEATURES).corr()
    im=ax.imshow(corr,cmap="RdBu_r",vmin=-1,vmax=1)
    ax.set_xticks(range(len(FEATURES)),FEATURES,rotation=65,ha="right",fontsize=8)
    ax.set_yticks(range(len(FEATURES)),FEATURES,fontsize=8)
    ax.set_title("Feature correlation after fitted preprocessing")
    fig.colorbar(im,ax=ax,fraction=.04);save(fig,plots/"feature_correlation.png")

    future=summarize_holdout(outcomes,"cluster")
    personas=summarize_holdout(outcomes,"rfm_persona")
    best=profiles.sort_values("gross_value_share",ascending=False).iloc[0]
    report=f"""# RetailLens — Customer Segmentation & Retention Intelligence

## Executive result

Executed on **{audit['raw_rows']:,} real source transaction records**, representing item-level rows, not distinct orders. After documented cleaning and the historical cutoff, the model covers **{len(df):,} customers**, **{m['feature_orders']:,} purchase invoices** and **£{m['feature_gross_value']:,.2f} gross merchandise value**.

The selected model is **{m['selected_candidate']}**, with sampled silhouette **{m['silhouette']:.3f}** and mean 80%-subsample adjusted Rand index **{m['stability_mean_ari']:.3f}**. PCA reduces 10 modeled features to **{m['pca_components']} components**, retaining **{m['pca_explained_variance']:.1%}** variance. These are internal diagnostics, not accuracy or proof of business value.

Cluster {int(best.cluster)} represents **{best.customer_share:.1%} of historical customers** and **{best.gross_value_share:.1%} of historical gross value**. The untreated holdout purchase rates below indicate differences in observed future behavior. They do not measure an intervention effect.

![Executive summary](figures/executive_summary.png)

## Problem and intended use

A retailer needs to prioritize retention outreach, recognize valuable repeat buyers, and tailor campaigns without manually inspecting a million purchase rows. This project builds interpretable customer groups and exports actionable customer lists. Campaign strategies are proposals that require randomized testing.

## Data provenance and audit

Source: Daqing Chen (2012), Online Retail II, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5CG6D. Dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Original source preserved in the project download archive; transformed outputs are derived works. Source SHA-256: `{audit['source_sha256']}`.

Source dates: {audit['date_range'][0]} to {audit['date_range'][1]}. Raw rows: {audit['raw_rows']:,}. Exact normalized duplicates removed: {audit['exact_duplicates_removed']:,}. Deduplicated rows: {audit['deduplicated_rows']:,}.

{md_table(pd.DataFrame(list(audit['row_disposition'].items()),columns=['Disposition after deduplication','Rows']))}

Audit categories are mutually exclusive in priority order: invalid record → missing customer → non-merchandise code → nonpositive price/zero quantity → return → purchase. Missing descriptions remain usable if the product code is valid. Returns include C-prefixed invoices or negative quantities. Returns never count as purchase frequency. Gross value excludes returns; observed net value subtracts return magnitudes but is not reconciled accounting revenue. Refunds cannot always be matched to their original invoice. The product-code rule can omit legitimate unusual codes. Identical lines might be genuine repeat entries; exact-row deduplication is an explicit assumption.

## Time boundaries and leakage controls

- Historical features: timestamps strictly before **{m['cutoff_exclusive']} 00:00**.
- Holdout: **{m['cutoff_exclusive']}** inclusive to **{m['holdout_end_exclusive']}** exclusive ({m['holdout_days']} days).
- The final source date is incomplete and excluded from validation.
- Customers first appearing during the holdout are excluded from historical-customer evaluation.
- Imputation, winsorization, scaling, PCA, clustering and persona thresholds use historical features only.
- Holdout outcomes are joined only after cluster selection; they never select the model.
- Monthly cohorts are descriptive over observed full months and are not modeling inputs.

## Features and transformations

SQL aggregates lines into orders, uses LAG for time between purchases and ROW_NUMBER for sequence/country selection, then joins customer, product and return aggregates. Frequency is invoice count, not row count. Monetary value is sum(quantity × unit price) over retained positive purchases, in GBP. Recency is calendar days from last purchase to the exclusive cutoff.

The 10 model inputs are: {', '.join(FEATURES)}. Additional descriptive fields include average order value, active months, purchase span, return value, observed net value, country and first-observed cohort. Country and identifiers are not clustering inputs.

Average gap is missing for one-order customers. The historical median fills it; the single-order indicator retains that information. Seven nonnegative magnitude features are capped at historical 1st/99th percentiles and log1p transformed. Refund ratio is log1p transformed; flags and shares are not percentile-capped. All inputs are standardized. Raw values remain unchanged in the customer export. Correlated RFM/behavior features can affect distance weighting; the correlation plot and an RFM-only partition comparison are supplied. PCA preserves variance, not guaranteed business relevance.

## Model comparison and selection

All models use the same customers, the same PCA representation and the same fixed random sample of {m['silhouette_sample_size']:,} customers for silhouette. All customers are used for fitting, Calinski–Harabasz and Davies–Bouldin. Full-covariance GMM uses three initializations; K-Means uses twenty. Ward hierarchical clustering is computed on all historical customers.

{m['selection_rule']}. The campaign use case requires at least three groups; k=2 is still reported. This business constraint is not a mathematical optimum. Elbow is a diagnostic for K-Means; AIC/BIC are GMM-only diagnostics. Selection does not claim a uniquely correct number of clusters.

{md_table(comparison[['candidate','silhouette','davies_bouldin','smallest_cluster_share','eligible']])}

![Model comparison](figures/model_selection.png)

Subsampling stability: refit the selected family and k on five independent 80% customer samples in the fixed fitted feature space; compare with original assignments on overlapping customers using ARI. Values: {', '.join(f'{v:.3f}' for v in m['stability_80pct_subsample_ari'])}. This checks cluster sensitivity conditional on preprocessing; it does not refit the entire preprocessing pipeline and is not a temporal stability estimate. Different subsamples use different random seeds. RFM-only K-Means versus selected partition ARI: **{m['rfm_only_reference_ari']:.3f}**; this is agreement, not superiority.

## Interpretable cluster profiles

{md_table(profiles)}

## Untouched future-period diagnostics

{md_table(future)}

Purchase rate = customers with at least one retained positive purchase in the holdout / all historical customers in that group. Intervals are 95% Wilson intervals under a binomial approximation and are descriptive; they do not address model-selection uncertainty. Future gross value excludes returns. Overall historical-customer purchase rate: **{m['overall_holdout_purchase_rate']:.1%}**. Seasonality and wholesale behavior may drive differences.

## RFM personas and proposed actions

Personas are explicit rules applied alongside the learned clusters, not names invented for cluster IDs. Rule order: inactive ≥90 days and historical spend ≥75th percentile → at-risk high-value; other inactive ≥90 days → dormant/at-risk; recent ≤30 days with high spend and frequency ≥loyal threshold → high-value loyal; recent ≤60 days with that frequency → loyal repeat; one purchase and observed customer age ≤60 days → new/first purchase; recent ≤30 days and high spend → recent high-value; otherwise developing/occasional. High-spend threshold: £{m['persona_thresholds']['high_value_75th_percentile_gbp']:,.2f}; loyal frequency: {m['persona_thresholds']['loyal_frequency_threshold']:.1f} invoices. 'At risk' is a heuristic inactivity label, not a validated churn prediction.

{md_table(personas)}

| Persona | Proposed action | Experiment outcome | Guardrail |
|---|---|---|---|
| High-value loyal / loyal repeat | Early access or replenishment reminders | Incremental repeat purchase rate | Opt-outs, discount cost |
| At-risk high-value | Personalized win-back message | Incremental contribution margin | Contact complaints, margin |
| Dormant / at-risk | Low-cost reactivation test | Incremental reactivation | Suppression of unresponsive customers |
| New / first purchase | Onboarding and second-purchase prompts | 60-day second-purchase rate | Returns and excessive incentives |
| Developing / occasional | Relevant cross-sell recommendations | Incremental basket value | Returns and opt-outs |

Randomize eligible customers within persona into treatment/control, predefine one primary outcome and observation period, and compare incremental results. Margin requires cost data absent here. No retention, revenue, conversion or ROI uplift has been measured.

## Cohorts, reproducibility and limitations

![Cohort retention](figures/cohort_retention.png)

Cohort month means first observed purchase, not verified acquisition. Month 0 is 100%; later cells divide active cohort customers by original cohort size. Grey/blank future cells are right-censored. The incomplete final calendar month is excluded. Data covers one historical UK retailer with wholesale customers; it does not represent current general e-commerce customers. Anonymous purchases are excluded from customer modeling, which can introduce selection bias. No demographics, product hierarchy, marketing exposure, profit margin or churn labels are available. Outliers and merchant-code assumptions should be reviewed for a live retailer.

The persisted model includes fitted preprocessing, PCA, metadata and assignment logic. K-Means/GMM support native predict; a selected hierarchical model uses a clearly labeled nearest-centroid approximation for new customers. Agreement between those prototypes and the fitted clustering is {m['centroid_training_agreement']:.1%}; this is only relevant for the hierarchical approximation. Do not treat prototype assignment as exact Ward prediction. Retrain and reprofile after material drift; numeric cluster IDs are not stable across refits. Versions: {m['versions']}. Seed: {m['seed']}.

## Source documentation

- UCI dataset and license: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Clustering: https://scikit-learn.org/stable/modules/clustering.html
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
"""
    (out/"REPORT.md").write_text(report)
    bullets=f"""# RetailLens — RFM Customer Segmentation & Retention Analytics

- Engineered SQL-based RFM and behavioral features from {audit['raw_rows']/1e6:.2f}M retail transaction records using joins, aggregations and window functions, modeling {len(df):,} historical customers after auditable cleaning.
- Benchmarked K-Means, Ward hierarchical clustering and Gaussian mixture models across 2–8 clusters; selected a {m['k']}-segment {m['algorithm']} solution with {m['silhouette']:.3f} silhouette and {m['stability_mean_ari']:.3f} mean subsampling ARI.
- Reduced 10 features to {m['pca_components']} principal components ({m['pca_explained_variance']:.1%} variance retained), profiled RFM personas and evaluated segment purchase behavior on an untouched {m['holdout_days']}-day holdout.

Use these only after you have reviewed the code and can explain the methodology. These results measure separation and stability, not classification accuracy or business uplift. '1.07M records' means raw line items, not unique orders. The selected model describes customers observed before {m['cutoff_exclusive']}; it does not use the future period for feature fitting.
"""
    (out/"RESUME_BULLETS.md").write_text(bullets)
