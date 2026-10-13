"""Same customers, features, and metric sample for every clustering candidate."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin
from threadpoolctl import threadpool_limits
from .features import FEATURES, LOG_FEATURES

class FeatureTransform(BaseEstimator, TransformerMixin):
    """Training-only imputation and 1/99% winsorization, log1p, then z-scores."""
    def fit(self, X, y=None):
        X = X[FEATURES].astype(float)
        self.medians_ = X.median().fillna(0.)
        clean = X.fillna(self.medians_)
        self.low_ = clean.quantile(.01)
        self.high_ = clean.quantile(.99)
        # Do not collapse rare flags or zero-heavy ratios with percentile clipping.
        for c in ["return_value_ratio","recent_order_share","single_order"]:
            self.low_[c],self.high_[c]=0., max(1.,float(clean[c].max()))
        self.scaler_ = StandardScaler().fit(self._prepare(X))
        return self
    def _prepare(self,X):
        out=X[FEATURES].astype(float).fillna(self.medians_).clip(lower=self.low_,upper=self.high_,axis=1)
        out[LOG_FEATURES]=np.log1p(out[LOG_FEATURES])
        # Compress the long refund tail without changing the reported raw ratio.
        out["return_value_ratio"]=np.log1p(out["return_value_ratio"])
        return out
    def transform(self, X):
        return self.scaler_.transform(self._prepare(X))

def fit_family(family, X, k, seed=42):
    if family == "KMeans":
        model=KMeans(n_clusters=k,n_init=20,random_state=seed)
    elif family == "GMM":
        model=GaussianMixture(n_components=k,covariance_type="full",n_init=3,
                              reg_covar=1e-5,max_iter=400,random_state=seed)
    else:
        labels=cut_tree(linkage(X,method="ward"),n_clusters=k).reshape(-1)
        return None,labels
    model.fit(X)
    return model,model.predict(X)

def compare_models(df, seed=42, sample_size=1500):
    prep=FeatureTransform().fit(df)
    scaled=prep.transform(df)
    pca=PCA(n_components=.90,svd_solver="full").fit(scaled)
    X=pca.transform(scaled)
    sample=np.sort(np.random.default_rng(seed).choice(len(X),min(sample_size,len(X)),replace=False))
    distances=pairwise_distances(X[sample])
    tree=linkage(X,method="ward")
    rows,models,labels_by_key=[],{},{}
    with threadpool_limits(limits=2):
        for family in ["KMeans","Hierarchical","GMM"]:
            for k in range(2,min(8,len(df)-1)+1):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always",ConvergenceWarning)
                    if family=="Hierarchical":
                        model=None
                        labels=cut_tree(tree,n_clusters=k).reshape(-1)
                    else:
                        model,labels=fit_family(family,X,k,seed)
                counts=np.bincount(labels,minlength=k)
                sil=silhouette_score(distances,labels[sample],metric="precomputed")
                conv=bool(getattr(model,"converged_",True))
                key=f"{family}_{k}"
                row=dict(candidate=key,algorithm=family,k=k,silhouette=sil,
                         davies_bouldin=davies_bouldin_score(X,labels),
                         calinski_harabasz=calinski_harabasz_score(X,labels),
                         smallest_cluster_share=counts.min()/len(df),
                         largest_cluster_share=counts.max()/len(df),converged=conv,
                         inertia=float(model.inertia_) if family=="KMeans" else np.nan,
                         bic=float(model.bic(X)) if family=="GMM" else np.nan,
                         aic=float(model.aic(X)) if family=="GMM" else np.nan,
                         warning_count=len(caught))
                # Three groups is an explicit campaign-design constraint; k=2 remains a benchmark.
                row["eligible"]=bool(k>=3 and counts.min()/len(df)>=.02 and counts.max()/len(df)<=.80 and conv)
                rows.append(row);models[key]=model;labels_by_key[key]=labels
                print(f"  {key}: silhouette={sil:.3f}; min share={row['smallest_cluster_share']:.1%}",flush=True)
    comparison=pd.DataFrame(rows).sort_values(["eligible","silhouette","k"],ascending=[False,False,True])
    eligible=comparison[comparison.eligible]
    if eligible.empty:
        raise ValueError("No candidate meets the predeclared segment size constraints; inspect model_comparison.csv.")
    winner=eligible.iloc[0]
    selected=labels_by_key[winner.candidate]
    # Initialization sensitivity is separate from bootstrap resampling stability.
    stability=[]
    rng=np.random.default_rng(seed)
    with threadpool_limits(limits=2):
        for run in range(5):
            idx=np.sort(rng.choice(len(X),int(.80*len(X)),replace=False))
            _,lab=fit_family(winner.algorithm,X[idx],int(winner.k),seed+run+1)
            stability.append(adjusted_rand_score(selected[idx],lab))
    # Stability is a diagnostic, not secretly optimized using future data.
    centers=np.vstack([X[selected==c].mean(axis=0) for c in range(int(winner.k))])
    centroid_agreement=float(np.mean(pairwise_distances_argmin(X,centers)==selected))
    bundle={"preprocessor":prep,"pca":pca,"model":models[winner.candidate],
            "centroids":centers,"algorithm":winner.algorithm,"k":int(winner.k),"feature_names":FEATURES,
            "assignment_note":"Nearest-centroid approximation for unseen customers" if winner.algorithm=="Hierarchical" else "Native model predict"}
    metadata={"selected_candidate":winner.candidate,"algorithm":winner.algorithm,"k":int(winner.k),
              "silhouette":float(winner.silhouette),"silhouette_sample_size":len(sample),
              "pca_components":int(pca.n_components_),"pca_explained_variance":float(pca.explained_variance_ratio_.sum()),
              "pca_variance_ratios":pca.explained_variance_ratio_.tolist(),
              "stability_80pct_subsample_ari":stability,"stability_mean_ari":float(np.mean(stability)),
              "centroid_training_agreement":centroid_agreement,
              "selection_rule":"Highest sampled silhouette among converged k>=3 candidates, min share>=2%, max share<=80%; k=2 benchmark only",
              "metric_space":"PCA scores retaining at least 90% of transformed feature variance",
              "feature_names":FEATURES,"seed":seed}
    # RFM-only reference; scores in a different geometry are not directly comparable.
    baseline=StandardScaler().fit_transform(np.log1p(df[["recency","frequency","monetary"]]))
    with threadpool_limits(limits=2):
        rfm=KMeans(n_clusters=int(winner.k),n_init=20,random_state=seed).fit_predict(baseline)
    metadata["rfm_only_reference_ari"]=float(adjusted_rand_score(selected,rfm))
    return comparison,selected,bundle,metadata,scaled,X

def predict(bundle, frame):
    X=bundle["pca"].transform(bundle["preprocessor"].transform(frame))
    if bundle["model"] is None:
        return pairwise_distances_argmin(X,bundle["centroids"])
    return bundle["model"].predict(X)
