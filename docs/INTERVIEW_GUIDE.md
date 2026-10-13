# Explain this project confidently

## 60-second explanation

“I built a reproducible customer-segmentation pipeline on UCI Online Retail II, which contains 1.07 million invoice-line records. I used SQL to clean and aggregate purchases into customer-level RFM and behavioral features, with joins and window functions for order gaps. I excluded returns from purchase frequency and kept their value separately. I fit preprocessing and PCA only on historical data, then compared K-Means, Ward hierarchical clustering and Gaussian mixtures for 2–8 clusters. Under explicit segment-size constraints, three-cluster K-Means was selected with a sampled silhouette of 0.323. Six components retained 92.9% of feature variance, and mean subsample ARI was 0.965. I interpreted the clusters with RFM profiles and checked purchasing over the following 90 days. The output supports retention hypotheses; it does not claim measured campaign uplift.”

Only use this explanation after reviewing the notebook, SQL and actual outputs yourself.

## Questions you should be able to answer

**1. Was it one million customers or orders?** Neither. There are 1,067,371 source invoice-line records. At the historical cutoff the cleaned analysis covers 618,314 purchase lines, 30,019 purchase invoices and 5,256 customers.

**2. What does each RFM feature mean?** Recency is days since the last purchase; frequency is distinct purchase invoice count; monetary is retained gross merchandise spend in GBP. Smaller recency means more recent activity. Larger F/M usually imply more purchases/spend, but wholesalers and observation length matter.

**3. Show the SQL.** In `sql/customer_features.sql`, `orders` aggregates line items into invoices. `ordered` uses `LAG(order_date)` within customer for order gaps. The final SELECT joins customer-level order, product, return and dominant-country aggregates. The join keys are unique at customer level, preventing multiplication of spend.

**4. Why not count transaction rows for frequency?** One invoice can contain many products. Counting lines would mistake basket diversity for repeat purchase activity.

**5. Why handle returns separately?** A cancellation is not a new purchase. It can be useful behavior, so the pipeline retains return magnitude and a return-value ratio without adding return invoices to frequency. Missing links to original purchases mean this is not an accounting net-revenue system.

**6. Why log-transform and scale?** Spend/frequency/basket size are skewed and differently scaled. Log1p compresses large nonnegative magnitudes; standardization makes distance calculations less dominated by units. Historical winsorization limits extreme effects but can compress legitimate large wholesalers. Raw reported values are never capped.

**7. Why PCA?** The 10 features contain correlated signals. Six components retain 92.9% of transformed variance. PCA makes orthogonal combinations, which is not the same as selecting six original features or proving predictive value. The two plotted PCs show only 58.5% of variance; fitting uses all six.

**8. Why compare three algorithms?** K-Means finds centroid-based Euclidean partitions; Ward greedily merges groups to minimize within-cluster variance; GMM models a mixture of multivariate Gaussian densities with soft membership probabilities. Their assumptions and shapes differ.

**9. Why three clusters if K=6 has a higher silhouette?** KMeans_6 has a tiny group of roughly 1.3%. The declared campaign constraint requires every group to contain at least 2% of customers, limits the largest to 80%, and requires at least three actionable groups. KMeans_3 has the highest silhouette among eligible candidates. This is a constrained operational choice, not a claim that three is universally correct.

**10. Explain silhouette.** For each point, a is its mean distance to its own group; b is the smallest mean distance to a different group. The point's score is (b−a)/max(a,b). It ranges from −1 to 1; larger indicates greater separation. The reported 0.323 is an average on the same fixed 1,500-customer sample for every candidate; it is not accuracy.

**11. Explain elbow, Davies–Bouldin, Calinski–Harabasz and BIC.** Elbow inspects diminishing inertia reductions for K-Means. Davies–Bouldin compares within-group scatter with between-centroid separation (lower is preferable). Calinski–Harabasz compares between-group and within-group dispersion (higher is preferable). BIC balances GMM fit and complexity (lower within the comparable GMM family). None substitutes for useful, stable segments.

**12. What is ARI?** Adjusted Rand index measures agreement between two partitions, adjusted for chance; it is unaffected by arbitrary cluster ID permutations. A value near 1 means strong agreement. Five refits on 80% samples give mean 0.965. This procedure fixes the original preprocessing, so it measures conditional clustering stability, not full pipeline or temporal stability.

**13. How did you prevent leakage?** Features and fitted transforms only use timestamps before 10 September 2011. Future purchasing is attached after the selected model is frozen. Holdout behavior does not choose k, feature thresholds or algorithms. New future customers are not smuggled into the historical population.

**14. Is at-risk a churn prediction?** No. It is a 90-day inactivity heuristic, split by historical value. Churn has no observed label or contractual definition here. Seasonal/wholesale buyers can naturally have long gaps. For instance, the at-risk high-value persona still has a 53.3% future purchase rate, illustrating the heuristic's limitations.

**15. What business insight did you measure?** Cluster 2 contains 47.9% of historical customers and 91.4% of their gross spend. Its 90-day purchase rate is 61.9%, compared with 21.3% for the least-active cluster. These are observational differences. They do not imply an outreach campaign caused any change.

**16. What would prove business value?** Randomize outreach within a persona, predefine an outcome and guardrails, and compare treatment/control incremental purchasing or contribution margin. Margin is unavailable in this dataset and needs cost data. Avoid claiming uplift before such an experiment.

**17. How do cohort tables avoid false zeros?** A cell beyond the available observation window is missing, not zero. An observable month with no repeat buyers is zero. Month 0 is 100%. The final incomplete month is excluded; first-observed purchases may not be true acquisition dates.

**18. What happens in production?** Engineer features at a new snapshot using the agreed observation-window policy, apply saved preprocessing/PCA and K-Means predict, monitor input drift and segment shares, and retrain when needed. Keep model and feature versions together. Align new clusters with profiles after retraining because numeric IDs can change. Long-history features also grow with tenure and need a reviewed window policy.

## Suggested study order

1. Follow the source fields and audit waterfall.
2. Read the SQL and work through the 12 fixture tests.
3. Understand preprocessing and why each modeling feature exists.
4. Open the model comparison and explain the constrained selection.
5. Interpret cluster profiles and future-period differences.
6. Read limitations and describe a randomized campaign test.
