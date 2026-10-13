"""Build the walkthrough and execute its Python cells; record real stdout outputs."""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
cells=[]
namespace={"__name__":"__main__"}
execution=0

def markdown(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip()+"\n"})

def code(source):
    global execution
    source=textwrap.dedent(source).strip()+"\n"
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(source,"<walkthrough>","exec"),namespace)
    execution+=1
    outputs=[]
    if stream.getvalue():outputs=[{"output_type":"stream","name":"stdout","text":stream.getvalue()}]
    cells.append({"cell_type":"code","execution_count":execution,"metadata":{},"source":source,"outputs":outputs})

def figure(name,caption):
    data=base64.b64encode((ROOT/"outputs/figures"/name).read_bytes()).decode()
    cells.append({"cell_type":"markdown","metadata":{},"source":f"![{caption}](attachment:{name})\n\n{caption}",
                  "attachments":{name:{"image/png":data}}})

markdown("""
# RetailLens | Customer segmentation & retention intelligence
**SQL · RFM · behavioral features · K-Means · Ward clustering · GMM · PCA**

This walkthrough includes real outputs from the full UCI Online Retail II dataset. Start with the source, follow the SQL and model comparisons, then interpret the untouched future-period results. Raw transaction records are invoice **line items**, not distinct orders.

**Local:** extract the archive, install `requirements.txt` into Python 3.12, and select that environment as the notebook kernel. **Colab:** the next cell prompts for the delivered project ZIP when needed and uses an isolated Python environment for the pipeline. Colab setup requires internet and has not been run in this environment; local cells have been executed successfully.
""")
code('''
from pathlib import Path
import os, sys, subprocess, json, sqlite3, zipfile
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

candidates = [Path.cwd(), Path.cwd()/"customer_segmentation", Path("/content/customer_segmentation")]
project = next((p for p in candidates if (p/"run.py").exists()), None)
if project is None and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()  # Select RetailLens_Customer_Segmentation.zip.
    archives = [Path(name) for name in uploaded if name.endswith(".zip")]
    if len(archives) != 1:
        raise ValueError("Upload exactly one project ZIP")
    destination = Path("/content").resolve()
    with zipfile.ZipFile(archives[0]) as archive:
        for member in archive.namelist():
            if not (destination/member).resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive path")
        archive.extractall(destination)
    project = destination/"customer_segmentation"
if project is None or not (project/"run.py").exists():
    raise FileNotFoundError("Extract the full project ZIP and open this notebook inside customer_segmentation")
project = project.resolve()
os.chdir(project)
RUN_PY = sys.executable
if IN_COLAB:
    env = project/".colab_env"
    if not (env/"bin/python").exists():
        subprocess.run([sys.executable,"-m","venv",str(env)],check=True)
    RUN_PY = str(env/"bin/python")
    subprocess.run([RUN_PY,"-m","pip","install","-q","-r","requirements.txt"],check=True)
print("Project located. Pipeline Python is configured.")
''')
markdown("""
## 1. Run the audited pipeline
The first run streams both original Excel sheets into SQLite. Later runs validate the source checksum and reuse that database. The algorithm operates on customer aggregates. The model is frozen before future outcomes are attached.
""")
code('''
result = subprocess.run([RUN_PY,"-u","run.py"],check=True,capture_output=True,text=True)
print(result.stdout)
''')
markdown("""
## 2. Validate counting, refunds and time boundaries
The fixtures have hand-calculated answers: invoices versus items, purchase/refund arithmetic, SQL order gaps, exact duplicates, exclusive cutoff, unseen future customers, missing gaps and censored cohorts. They are test fixtures, not substituted project data.
""")
code('''
test_result = subprocess.run([RUN_PY,"-m","unittest","discover","-s","tests","-v"],capture_output=True,text=True)
print(test_result.stdout + test_result.stderr)
test_result.check_returncode()
''')
markdown("## 3. Inspect the full-data audit")
code('''
import pandas as pd
audit = json.loads(Path("outputs/data_audit.json").read_text())
metrics = json.loads(Path("outputs/metrics.json").read_text())
print("Raw records:", f"{audit['raw_rows']:,}")
print("Exact normalized duplicates:", f"{audit['exact_duplicates_removed']:,}")
print(pd.Series(audit["row_disposition"],name="rows").to_string())
print("Source SHA-256:",audit["source_sha256"])
''')
markdown("""
## 4. Read the actual feature SQL
Each aggregation has one row per customer before the final joins. `LAG` calculates consecutive order gaps. `ROW_NUMBER` selects the dominant country deterministically. Every source CTE respects the historical cutoff.
""")
code('''
print(Path("sql/customer_features.sql").read_text())
''')
markdown("## 5. Inspect raw customer features before modeling transforms")
code('''
customers = pd.read_csv("outputs/customer_segments.csv",dtype={"customer_id":str})
fields = ["customer_id","recency","frequency","monetary","unique_products","avg_gap_days","single_order"]
print(customers[fields].head(10).to_string(index=False))
print("Customer rows:",len(customers),"| Historical invoices:",int(customers.frequency.sum()))
assert customers.customer_id.is_unique
assert (pd.to_datetime(customers.last_purchase) < pd.Timestamp(metrics["cutoff_exclusive"])).all()
print("One row per customer and historical cutoff checks passed.")
''')
figure("rfm_distributions.png","Raw distributions reveal skew; frequency and monetary axes are logarithmic. Reported monetary values remain uncapped.")
markdown("""
## 6. Preprocessing and PCA
Ten features enter a historical-only transform: median imputation for missing gaps; 1st/99th percentile clipping and log1p for seven magnitudes; log1p for refund ratio; and z-score scaling. Rare flags are preserved. PCA retains at least 90% variance. See `src/modeling.py` for the fitted transformer and `docs/DATA_DICTIONARY.md` for definitions.
""")
code('''
print("Model features:",", ".join(metrics["feature_names"]))
print("Retained components:",metrics["pca_components"])
print("Retained variance:",f"{metrics['pca_explained_variance']:.2%}")
print(pd.read_csv("outputs/pca_loadings.csv",index_col=0).round(3).to_string())
''')
markdown("""
## 7. Compare algorithms and cluster counts
Twenty-one fits use identical customer features. Silhouette uses a common fixed 1,500-customer sample. The campaign constraint requires 3+ groups, each ≥2% of customers and the largest ≤80%. A higher raw score with a tiny group does not satisfy that declared constraint. Elbow and BIC remain family-specific diagnostics.
""")
code('''
comparison = pd.read_csv("outputs/model_comparison.csv")
print(comparison[["candidate","silhouette","davies_bouldin","smallest_cluster_share","eligible"]].round(4).to_string(index=False))
print("Selected:",metrics["selected_candidate"])
print("Selection rule:",metrics["selection_rule"])
print("80% subsample ARIs:",[round(v,4) for v in metrics["stability_80pct_subsample_ari"]])
print("Mean ARI:",round(metrics["stability_mean_ari"],4))
''')
figure("model_selection.png","Algorithm comparisons use the same geometry. GMM BIC is not compared with K-Means inertia.")
markdown("""
## 8. Interpret the clusters
Cluster IDs are arbitrary. Inspect original-unit medians and historical value shares. RFM personas are a separate transparent marketing overlay; they are not fabricated cluster labels.
""")
code('''
print(pd.read_csv("outputs/cluster_profiles.csv").round(3).to_string(index=False))
print("\\nPersona counts:")
print(customers.rfm_persona.value_counts().to_string())
''')
figure("executive_summary.png","All measured results are observational. The two-dimensional view is a projection of the six-dimensional fitted representation.")
markdown("""
## 9. Observe the untouched future period
This checks whether historical segments have different future purchasing. It does not measure campaign lift, supervised prediction accuracy or churn. Inactive historical customers remain in the denominator. Future first-time customers are excluded.
""")
code('''
print("Holdout:",metrics["cutoff_exclusive"],"inclusive →",metrics["holdout_end_exclusive"],"exclusive")
print(pd.read_csv("outputs/holdout_by_cluster.csv").round(4).to_string(index=False))
print("\\nRFM persona outcomes:")
print(pd.read_csv("outputs/holdout_by_persona.csv").round(4).to_string(index=False))
''')
markdown("""
## 10. Review monthly cohorts
A cohort means first observed purchase month, not verified acquisition month. Grey cells are unobserved future periods; observed inactivity is zero. The incomplete final month is omitted. This descriptive table includes available full months and is not used to fit or select clusters.
""")
figure("cohort_retention.png","Monthly retention = active customers from the original cohort / original cohort size. Month 0 is 100%.")
markdown("""
## 11. Verify persisted scoring
The selected K-Means artifact contains historical preprocessing and PCA. Native `predict` should reproduce the delivered historical labels. If a future run selects hierarchical clustering, unseen customers use a documented nearest-centroid approximation instead; that approximation need not reproduce Ward labels exactly.
""")
code('''
scored_path = Path("outputs/scoring_check.csv")
scoring = subprocess.run([RUN_PY,"score.py","outputs/customer_segments.csv","--output",str(scored_path)],check=True,capture_output=True,text=True)
print(scoring.stdout)
scored = pd.read_csv(scored_path,dtype={"customer_id":str})
agreement = (customers.cluster.to_numpy() == scored.cluster.to_numpy()).mean()
print("Saved-model assignment agreement:",f"{agreement:.2%}")
if metrics["algorithm"] != "Hierarchical":
    assert agreement == 1.0
scored_path.unlink()
''')
markdown("""
## 12. Turn findings into a testable action
Propose personalized win-back for high-value inactive customers, second-purchase onboarding for new customers, and replenishment/early access for valuable repeat buyers. Randomize within each persona and measure incremental results against a control; track discount cost, returns and opt-outs. This dataset lacks margin and campaign exposure.

Read `outputs/REPORT.md` for the full evidence and limitations. Read `docs/INTERVIEW_GUIDE.md` before using the resume bullets.

**Attribution:** Chen, D. (2012), Online Retail II, UCI Machine Learning Repository, https://doi.org/10.24432/C5CG6D . CC BY 4.0. No simulated data was used for the full-data analysis.
""")
code('''
print(Path("outputs/RESUME_BULLETS.md").read_text())
''')
notebook={"nbformat":4,"nbformat_minor":5,"metadata":{
    "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.12"},
    "title":"RetailLens Customer Segmentation"},"cells":cells}
for i,cell in enumerate(cells):cell["id"]=f"retaillens-{i:03d}"
(ROOT/"RetailLens_Walkthrough.ipynb").write_text(json.dumps(notebook,indent=1))
print(f"Executed {execution} code cells; created {len(cells)}-cell notebook.")
